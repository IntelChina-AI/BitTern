"""Inference-only fake quantizers used while merging released checkpoints."""

import math

import torch
import torch.nn as nn


CLIPMIN = 1e-5


class ScaleSigmoid(nn.Module):
    def __init__(self, dim, alpha=1.0, beta=0.0):
        super().__init__()
        self.scale = alpha
        self.shift = beta
        self.bound_factor = nn.Parameter(torch.zeros((dim, 1)), requires_grad=False)

    def forward(self):
        return torch.sigmoid(self.bound_factor) * self.scale + self.shift


class DoubleScaleSigmoid(nn.Module):
    def __init__(self, dim, alpha=1.0, beta=0.0):
        super().__init__()
        self.scale = alpha
        self.shift = beta
        self.bound_factor_A = nn.Parameter(torch.zeros((dim, 1)), requires_grad=False)
        self.bound_factor_B = nn.Parameter(torch.zeros((dim, 1)), requires_grad=False)

    def forward(self):
        numerator = torch.sigmoid(self.bound_factor_A) * self.scale + self.shift
        denominator = torch.sigmoid(self.bound_factor_B) * self.scale + self.shift
        return numerator / denominator


class ScaleSoftPlus(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bound_factor = nn.Parameter(torch.zeros((dim, 1)), requires_grad=False)

    def forward(self):
        return torch.nn.functional.softplus(self.bound_factor) * 1.4427


class ScaleExp(nn.Module):
    def __init__(self, dim, alpha):
        super().__init__()
        self.scale = alpha
        self.bound_factor = nn.Parameter(torch.zeros((dim, 1)), requires_grad=False)

    def forward(self):
        return torch.exp(self.bound_factor * self.scale)


def _factor_module(kind, dim, *, round_factor=False):
    if kind == "sigmoid" or (kind == "round_double_sigmoid" and not round_factor):
        return ScaleSigmoid(dim, alpha=2.0)
    if kind in {"double_sigmoid", "round_double_sigmoid"}:
        return DoubleScaleSigmoid(dim, alpha=2.0)
    if kind == "softplus":
        return ScaleSoftPlus(dim)
    if kind == "exp":
        return ScaleExp(dim, alpha=0.4)
    raise ValueError(f"Unsupported factor activation: {kind}")


class TernaryQuantizer(nn.Module):
    def __init__(self, weight_quant_params, shape=None):
        super().__init__()
        self.group_size = weight_quant_params["group_size"] or shape[-1]
        self.shift_mu = weight_quant_params["shift_mu"]
        self.drop_quant_mu = weight_quant_params["drop_quant_mu"]
        self.ter_scale_type = weight_quant_params["ter_scale_type"]
        self.init_scale_from_raw_weights = weight_quant_params.get(
            "init_scale_from_raw_weights",
            False,
        )
        self.learnable_scale = weight_quant_params["learnable_scale"]
        self.learnable_mu = weight_quant_params["learnable_mu"]
        self.learnable_round = weight_quant_params["learnable_round"]
        self.init_round_thd = weight_quant_params["init_round_thd"]
        dim = int(shape[0] * math.ceil(shape[1] / self.group_size))
        factor_kind = weight_quant_params["learnable_factor_act"]
        if self.learnable_scale:
            self.generate_scale_factor = _factor_module(factor_kind, dim)
        if self.learnable_mu:
            if not self.shift_mu:
                raise ValueError("shift_mu must be enabled when the checkpoint contains mean factors")
            self.generate_mu_factor = ScaleSigmoid(dim, alpha=2.0, beta=-1.0)
        if self.learnable_round:
            self.generate_round_factor = _factor_module(factor_kind, dim, round_factor=True)

    def forward(self, weight, quant_rate=1.0):
        del quant_rate
        original_shape = weight.shape
        grouped = weight.reshape(-1, self.group_size)
        mean = grouped.mean(dim=-1, keepdim=True) if self.shift_mu else grouped.new_zeros((grouped.shape[0], 1))
        scale_values = grouped if self.init_scale_from_raw_weights else grouped - mean
        if self.ter_scale_type == "absmean":
            scale = scale_values.abs().mean(dim=-1, keepdim=True) + 1e-6
        elif self.ter_scale_type == "variance":
            scale = scale_values.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
        else:
            raise ValueError(f"Unsupported ternary scale type: {self.ter_scale_type}")

        if self.learnable_mu:
            mean = mean + self.generate_mu_factor() * scale
        if self.learnable_scale:
            scale = self.generate_scale_factor() * scale
        threshold = self.init_round_thd
        if self.learnable_round:
            threshold = threshold * self.generate_round_factor()

        quantized = torch.clamp(torch.round(((grouped - mean) / scale) * 0.5 / threshold), -1, 1) * scale
        if self.shift_mu and not self.drop_quant_mu:
            quantized = quantized + mean
        return quantized.reshape(original_shape)


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits=8,
        symmetric=False,
        per_channel_axes=None,
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        disable_zero_point=False,
        is_weight_quant=False,
        **kwargs,
    ):
        super().__init__()
        del per_channel_axes, dynamic, kwargs
        if not 1 <= n_bits <= 16:
            raise ValueError("bitwidth must be between 1 and 16")
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.metric = metric
        self.dynamic_method = dynamic_method
        self.group_size = group_size
        self.disable_zero_point = disable_zero_point
        self.qmin = -(2 ** (n_bits - 1)) if disable_zero_point else 0
        self.qmax = 2 ** (n_bits - 1) - 1 if disable_zero_point else 2**n_bits - 1
        self.deficiency = 0
        self.lwc = is_weight_quant and n_bits < 16
        if self.lwc:
            if group_size:
                dim = int(shape[0] * math.ceil(shape[1] / group_size))
                remainder = shape[-1] % group_size
                self.deficiency = group_size - remainder if remainder else 0
            else:
                dim = shape[0]
            self.upbound_factor = nn.Parameter(torch.full((dim, 1), 4.0), requires_grad=False)
            self.lowbound_factor = nn.Parameter(torch.full((dim, 1), 4.0), requires_grad=False)

    def _group(self, value):
        if not self.group_size:
            return value
        if self.deficiency:
            value = torch.cat((value, value.new_zeros((value.shape[0], self.deficiency))), dim=1)
        return value.reshape(-1, self.group_size)

    def forward(self, value, quant_rate=1.0):
        del quant_rate
        if self.n_bits >= 16:
            return value
        if self.metric == "fix0to1":
            levels = 2**self.n_bits - 1
            return value.mul(levels).round().div(levels)
        if self.dynamic_method == "per_token_bitnet":
            scale = 127.0 / value.abs().amax(dim=-1, keepdim=True).clamp(min=CLIPMIN)
            return (value * scale).round().clamp(-128, 127) / scale
        if self.dynamic_method not in {"per_token", "per_channel"}:
            raise ValueError(f"Unsupported dynamic method: {self.dynamic_method}")

        original_shape = value.shape
        grouped = self._group(value)
        minimum = grouped.amin(dim=-1, keepdim=True)
        maximum = grouped.amax(dim=-1, keepdim=True)
        if self.lwc:
            maximum = torch.sigmoid(self.upbound_factor) * maximum
            minimum = torch.sigmoid(self.lowbound_factor) * minimum
        if self.symmetric:
            scale = torch.maximum(maximum.abs(), minimum.abs()) / (2 ** (self.n_bits - 1) - 1)
            zero_point = None if self.disable_zero_point else (2 ** (self.n_bits - 1) - 1)
        else:
            scale = (maximum - minimum) / (2**self.n_bits - 1)
            zero_point = None if self.disable_zero_point else torch.round(-minimum / scale.clamp(min=CLIPMIN))
        scale = scale.clamp(min=CLIPMIN, max=1e4)
        quantized = torch.round(grouped / scale)
        if zero_point is not None:
            quantized = quantized + zero_point
        quantized = quantized.clamp(self.qmin, self.qmax)
        if zero_point is not None:
            quantized = quantized - zero_point
        quantized = quantized * scale
        if self.group_size:
            padded_columns = original_shape[1] + self.deficiency
            quantized = quantized.reshape(original_shape[0], padded_columns)
            if self.deficiency:
                quantized = quantized[:, : original_shape[1]]
        return quantized.reshape(original_shape)
