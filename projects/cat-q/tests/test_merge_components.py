import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - minimal documentation environments
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MergeComponentTest(unittest.TestCase):
    @staticmethod
    def _ternary_params(**overrides):
        params = {
            "n_bits": 1,
            "group_size": 4,
            "shift_mu": True,
            "drop_quant_mu": False,
            "ter_scale_type": "absmean",
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

    def test_scale_initialization_can_use_raw_grouped_weights(self):
        from quantize.quantizer import TernaryQuantizer

        weight = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
        centered_quantizer = TernaryQuantizer(self._ternary_params(), shape=weight.shape)
        raw_quantizer = TernaryQuantizer(
            self._ternary_params(init_scale_from_raw_weights=True),
            shape=weight.shape,
        )

        torch.testing.assert_close(
            centered_quantizer(weight),
            torch.tensor([[1.0, 1.0, 4.0, 7.0]]),
        )
        torch.testing.assert_close(
            raw_quantizer(weight),
            torch.tensor([[0.0, 4.0, 4.0, 8.0]]),
        )

    def test_lora_and_ternary_checkpoint_key_schema(self):
        from quantize.int_linear_lora import LoRAQuantLinear

        weight_params = {
            "n_bits": 1,
            "group_size": 4,
            "shift_mu": True,
            "drop_quant_mu": True,
            "ter_scale_type": "absmean",
            "learnable_scale": True,
            "learnable_mu": True,
            "learnable_round": True,
            "learnable_factor_act": "sigmoid",
            "init_round_thd": 0.5,
            "per_channel_axes": [0],
            "symmetric": False,
            "dynamic_method": "per_channel",
            "disable_zero_point": False,
        }
        act_params = {
            "n_bits": 16,
            "per_channel_axes": [],
            "symmetric": False,
            "dynamic_method": "per_token",
        }
        layer = LoRAQuantLinear(
            nn.Linear(4, 2, bias=False),
            weight_params,
            act_params,
            r=2,
            lora_attr={"lora_iter_num": 1},
        )
        self.assertEqual(
            set(layer.state_dict()),
            {
                "weight",
                "lora_A.0",
                "lora_B.0",
                "weight_quantizer.generate_scale_factor.bound_factor",
                "weight_quantizer.generate_mu_factor.bound_factor",
                "weight_quantizer.generate_round_factor.bound_factor",
            },
        )


if __name__ == "__main__":
    unittest.main()
