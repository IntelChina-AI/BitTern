"""Linear wrapper carrying checkpoint-compatible fake quantizers."""

import torch.nn as nn
import torch.nn.functional as F

from quantize.quantizer import UniformAffineQuantizer


class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module,
        weight_quant_params,
        act_quant_params,
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_func = F.linear
        self.fwd_kwargs = {}
        self.register_buffer("weight", org_module.weight)
        if getattr(org_module, "bias", None) is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.use_weight_quant = False
        self.use_act_quant = False
        self.quant_rate = 1.0
        self.disable_input_quant = disable_input_quant

        if weight_quant_params["n_bits"] > 1:
            self.weight_quantizer = UniformAffineQuantizer(
                **weight_quant_params,
                shape=org_module.weight.shape,
                is_weight_quant=True,
            )
            self.act_quantizer = (
                None
                if disable_input_quant
                else UniformAffineQuantizer(**act_quant_params)
            )

    def forward(self, inputs):
        weight = self.weight_quantizer(self.weight, self.quant_rate) if self.use_weight_quant else self.weight
        if self.use_act_quant and not self.disable_input_quant:
            inputs = self.act_quantizer(inputs, self.quant_rate)
        return self.fwd_func(inputs, weight, self.bias, **self.fwd_kwargs)

    def set_quant_state(self, weight_quant=False, act_quant=False, quant_rate=1.0):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        self.quant_rate = quant_rate
