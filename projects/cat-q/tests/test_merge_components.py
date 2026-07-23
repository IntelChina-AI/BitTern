import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - minimal documentation environments
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MergeComponentTest(unittest.TestCase):
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
