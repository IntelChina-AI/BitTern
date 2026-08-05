import unittest

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - minimal documentation environments
    np = None
    torch = None
    nn = None


def _ternary_params(**overrides):
    params = {
        "n_bits": 1,
        "group_size": 128,
        "shift_mu": False,
        "drop_quant_mu": True,
        "ter_scale_type": "absmean",
        "init_scale_from_raw_weights": True,
        "learnable_scale": False,
        "learnable_mu": False,
        "learnable_round": False,
        "learnable_factor_act": "sigmoid",
        "init_round_thd": 0.5,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "disable_zero_point": False,
    }
    params.update(overrides)
    return params


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExtractTernaryTest(unittest.TestCase):
    def _quantizer(self, shape, **overrides):
        from quantize.quantizer import TernaryQuantizer

        return TernaryQuantizer(weight_quant_params=_ternary_params(**overrides), shape=shape)

    def test_codes_are_ternary_and_grouped(self):
        from quantize.ternary_export import extract_ternary

        torch.manual_seed(0)
        weight = torch.randn(64, 256)
        ternary = extract_ternary(self._quantizer(weight.shape), weight)

        self.assertEqual(ternary.group_size, 128)
        self.assertEqual(ternary.codes.dtype, torch.int8)
        self.assertEqual(tuple(ternary.codes.shape), (64, 256))
        self.assertEqual(ternary.scales.numel(), weight.numel() // 128)
        self.assertTrue(bool(torch.isin(ternary.codes, torch.tensor([-1, 0, 1], dtype=torch.int8)).all()))

    def test_codes_and_scales_reproduce_the_fake_quantized_weight(self):
        from quantize.ternary_export import extract_ternary

        torch.manual_seed(1)
        weight = torch.randn(32, 128)
        for overrides in ({}, {"learnable_scale": True}, {"learnable_round": True}):
            with self.subTest(**overrides):
                quantizer = self._quantizer(weight.shape, **overrides)
                ternary = extract_ternary(quantizer, weight)
                self.assertTrue(torch.equal(ternary.dequantize(), quantizer(weight)))

    def test_partial_groups_are_rejected(self):
        from quantize.ternary_export import extract_ternary

        weight = torch.randn(8, 192)
        with self.assertRaises(ValueError):
            extract_ternary(self._quantizer(weight.shape), weight)

    def test_a_retained_group_mean_is_rejected(self):
        from quantize.ternary_export import extract_ternary

        weight = torch.randn(8, 128)
        quantizer = self._quantizer(weight.shape, shift_mu=True, drop_quant_mu=False)
        with self.assertRaises(ValueError):
            extract_ternary(quantizer, weight)

    def test_lora_update_is_applied_before_ternarization(self):
        from quantize.int_linear_lora import LoRAQuantLinear
        from quantize.ternary_export import _merged_weight, extract_ternary

        torch.manual_seed(2)
        linear = nn.Linear(128, 32, bias=False)
        module = LoRAQuantLinear(
            org_module=linear,
            weight_quant_params=_ternary_params(),
            act_quant_params={"n_bits": 16},
            r=4,
            lora_alpha=4,
        )
        with torch.no_grad():
            module.lora_A[0].normal_()
            module.lora_B[0].normal_()

        merged = _merged_weight(module)
        self.assertFalse(torch.equal(merged, module.weight))
        ternary = extract_ternary(module.weight_quantizer, merged)
        self.assertTrue(torch.equal(ternary.dequantize(), module.weight_quantizer(merged)))


@unittest.skipIf(np is None, "NumPy is not installed")
class PackQ2_0Test(unittest.TestCase):
    @staticmethod
    def _pack_module():
        from quantize import q2_0

        return q2_0

    def test_block_layout_matches_the_runtime_struct(self):
        q2_0 = self._pack_module()
        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :4] = [-1, 0, 1, 0]
        packed = q2_0.pack_q2_0(codes, np.array([0.5], dtype=np.float32))

        self.assertEqual(packed.shape, (1, 34))
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(np.float16(packed[0, :2].copy().view(np.float16)[0]), np.float16(0.5))
        # 00 | 01 | 10 | 01 packed low bits first -> 0b01100100
        self.assertEqual(packed[0, 2], 0b01100100)
        # the remaining weights are zero, i.e. level 01 in every slot
        self.assertTrue((packed[0, 3:] == 0b01010101).all())

    def test_round_trip(self):
        q2_0 = self._pack_module()
        rng = np.random.default_rng(0)
        codes = rng.integers(-1, 2, size=(6, 256)).astype(np.int8)
        scales = rng.random(codes.size // 128).astype(np.float32)

        packed = q2_0.pack_q2_0(codes, scales)
        self.assertEqual(packed.shape, (6, 2 * 34))
        restored = q2_0.unpack_q2_0(packed, codes.shape)
        expected = codes.reshape(-1, 128) * scales.astype(np.float16).astype(np.float32).reshape(-1, 1)
        self.assertTrue(np.array_equal(restored, expected.reshape(codes.shape)))

    def test_expert_stacks_keep_their_leading_dimension(self):
        q2_0 = self._pack_module()
        codes = np.zeros((3, 4, 128), dtype=np.int8)
        packed = q2_0.pack_q2_0(codes, np.ones(12, dtype=np.float32))
        self.assertEqual(packed.shape, (3, 4, 34))

    def test_non_ternary_codes_are_rejected(self):
        q2_0 = self._pack_module()
        codes = np.full((1, 128), 2, dtype=np.int8)
        with self.assertRaises(ValueError):
            q2_0.pack_q2_0(codes, np.ones(1, dtype=np.float32))




@unittest.skipIf(np is None, "NumPy is not installed")
class GGUFExportRulesTest(unittest.TestCase):
    """Conversion rules that decide what ends up in the GGUF and how."""

    @staticmethod
    def _exporter(arch="qwen3moe"):
        import gguf

        from quantize.gguf_export import ARCHITECTURES, TernaryGGUFExporter

        exporter = TernaryGGUFExporter.__new__(TernaryGGUFExporter)
        exporter.arch = ARCHITECTURES["Qwen3MoeForCausalLM" if arch == "qwen3moe" else "LlamaForCausalLM"]
        exporter.block_count = 2
        exporter.n_head = 4
        exporter.n_head_kv = 2
        exporter.float_type = gguf.GGMLQuantizationType.F16
        exporter.tensor_map = gguf.get_tensor_name_map(exporter.arch.arch, exporter.block_count)
        return exporter

    def test_tensor_names_follow_the_gguf_convention(self):
        exporter = self._exporter()
        self.assertEqual(exporter._gguf_name("model.embed_tokens.weight"), "token_embd.weight")
        self.assertEqual(
            exporter._gguf_name("model.layers.1.self_attn.q_proj.weight"), "blk.1.attn_q.weight"
        )
        self.assertEqual(
            exporter._gguf_name("model.layers.1.mlp.experts.down_proj.weight"),
            "blk.1.ffn_down_exps.weight",
        )
        with self.assertRaises(ValueError):
            exporter._gguf_name("model.layers.1.mystery.weight")

    def test_norms_and_the_router_stay_in_f32(self):
        import gguf

        exporter = self._exporter()
        f32 = gguf.GGMLQuantizationType.F32
        self.assertEqual(exporter._float_dtype("blk.0.attn_norm.weight", 1), f32)
        self.assertEqual(exporter._float_dtype("blk.0.attn_q_norm.weight", 2), f32)
        self.assertEqual(exporter._float_dtype("blk.0.ffn_gate_inp.weight", 2), f32)
        self.assertEqual(
            exporter._float_dtype("token_embd.weight", 2), gguf.GGMLQuantizationType.F16
        )

    def test_expert_slots_are_grouped_by_projection(self):
        exporter = self._exporter()
        self.assertEqual(
            exporter._expert_slot("model.layers.3.mlp.experts.7.up_proj.weight"),
            ("model.layers.3.mlp.experts.up_proj.weight", 7),
        )
        self.assertIsNone(exporter._expert_slot("model.layers.3.mlp.up_proj.weight"))

    def test_llama_permutation_applies_to_codes_and_scales(self):
        from quantize.gguf_export import _permute
        from quantize.q2_0 import unpack_q2_0

        rng = np.random.default_rng(0)
        codes = rng.integers(-1, 2, size=(8, 256)).astype(np.int8)
        scales = rng.random((8, 2)).astype(np.float16).astype(np.float32)
        weight = codes.astype(np.float32) * np.repeat(scales, 128, axis=1)

        exporter = self._exporter(arch="llama")
        packed = exporter._pack("model.layers.0.self_attn.q_proj.weight", codes, scales)
        restored = unpack_q2_0(packed, codes.shape)
        np.testing.assert_allclose(restored, _permute(weight, 4, 4), rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main()
