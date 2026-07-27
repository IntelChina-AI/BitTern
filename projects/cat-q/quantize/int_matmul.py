"""Activation-quantized matrix multiplication for scaled evaluation models."""

import torch
import torch.nn as nn

from quantize.quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params,
        x2_quant_params,
        disable_act_quant=False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        self.use_act_quant = False
        self.quant_rate = 1.0
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func
        self.disable_act_quant = disable_act_quant

    def set_quant_state(self, weight_quant=False, act_quant=False, quant_rate=1.0):
        del weight_quant
        self.use_act_quant = act_quant and not self.disable_act_quant
        self.quant_rate = quant_rate

    def quant_x1(self, value):
        return self.x1_quantizer(value, self.quant_rate) if self.use_act_quant else value

    def quant_x2(self, value):
        return self.x2_quantizer(value, self.quant_rate) if self.use_act_quant else value

    def forward(self, x1, x2):
        return self.matmul_func(x1, x2)
