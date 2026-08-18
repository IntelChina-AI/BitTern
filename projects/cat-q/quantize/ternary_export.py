"""Packed ternary export for released CAT-Q checkpoints.

`quantize.merge` restores a checkpoint and folds the result back into ordinary
floating-point Hugging Face weights, which is a *fake-quantized* model: the
values are ternary but every weight still occupies 16 bits.

This module taps the same restoration path one step earlier and keeps the two
factors the quantizer works with,

    W_g = s_g * T_g,    T_g in {-1, 0, +1},    |g| = group_size

so that a deployment backend can store `T` at 2 bits per weight next to one
scale per group.  `quantize.gguf_export` consumes the tensors yielded here
and packs them straight into a GGUF file.

Nothing here re-estimates a scale from the merged float weights: the numbers
come from the quantizer itself, and every tensor is checked against the stock
fake-quantized output before it is handed on.
"""

from dataclasses import dataclass

import torch

from quantize.int_linear import QuantLinear
from quantize.int_linear_lora import LoRAQuantLinear


@dataclass
class TernaryTensor:
    """Ternary codes plus one scale per group, as stored by the exporter."""

    codes: torch.Tensor  # int8, weight shape, values in {-1, 0, +1}
    scales: torch.Tensor  # float32, (numel // group_size, 1)
    group_size: int

    def dequantize(self):
        return (
            self.codes.to(torch.float32).reshape(-1, self.group_size) * self.scales
        ).reshape(self.codes.shape)


def extract_ternary(quantizer, weight):
    """Re-run `TernaryQuantizer.forward` and keep its ternary codes and scales.

    The returned factors are verified to reproduce `quantizer(weight)` exactly,
    so the packed model is numerically identical to the fake-quantized one.
    """
    group_size = quantizer.group_size
    if not group_size:
        raise ValueError("Packed ternary export requires a grouped ternary quantizer")
    original_shape = weight.shape
    if original_shape[-1] % group_size != 0:
        raise ValueError(
            f"in_features={original_shape[-1]} is not a multiple of group_size={group_size}; "
            "the packed ternary format requires whole groups"
        )

    grouped = weight.reshape(-1, group_size)
    if quantizer.shift_mu:
        mean = grouped.mean(dim=-1, keepdim=True)
    else:
        mean = grouped.new_zeros((grouped.shape[0], 1))

    if quantizer.ter_scale_type == "absmean":
        absmean_values = grouped if quantizer.init_scale_from_raw_weights else grouped - mean
        scale = absmean_values.abs().mean(dim=-1, keepdim=True) + 1e-6
    elif quantizer.ter_scale_type == "variance":
        scale = grouped.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
    else:
        raise ValueError(f"Unsupported ternary scale type: {quantizer.ter_scale_type}")

    if quantizer.learnable_mu:
        mean = mean + quantizer.generate_mu_factor() * scale
    if quantizer.learnable_scale:
        scale = quantizer.generate_scale_factor() * scale
    threshold = quantizer.init_round_thd
    if quantizer.learnable_round:
        threshold = threshold * quantizer.generate_round_factor()

    if quantizer.shift_mu and not quantizer.drop_quant_mu:
        raise ValueError(
            "drop_quant_mu is disabled: the merged weight keeps a per-group mean offset "
            "and is therefore not representable as scale * ternary"
        )

    codes = torch.clamp(torch.round(((grouped - mean) / scale) * 0.5 / threshold), -1, 1)
    reconstructed = (codes * scale).reshape(original_shape)
    if not torch.equal(reconstructed, quantizer(weight)):
        raise AssertionError(
            "Extracted ternary codes and scales do not reproduce the fake-quantized weight"
        )

    return TernaryTensor(
        codes=codes.reshape(original_shape).to(torch.int8),
        scales=scale.to(torch.float32).contiguous(),
        group_size=group_size,
    )


def _merged_weight(module):
    """Weight seen by the quantizer, i.e. with the LoRA update already applied."""
    weight = module.weight
    if isinstance(module, LoRAQuantLinear) and module.r > 0:
        for index in range(module.lora_iter_num):
            weight = weight + module.lora_B[index] @ module.lora_A[index] * module.scaling
    return weight


@torch.no_grad()
def iter_layer_ternary(layer_id, qlayer):
    """Yield `(weight_name, TernaryTensor)` for one restored decoder layer.

    Called from `merge_catq_checkpoint` while the layer still carries its
    quantizer, i.e. before the ternary weights are folded into float tensors.
    """
    found = False
    for name, module in qlayer.named_modules():
        if not isinstance(module, (QuantLinear, LoRAQuantLinear)):
            continue
        quantizer = getattr(module, "weight_quantizer", None)
        if type(quantizer).__name__ != "TernaryQuantizer":
            continue
        found = True
        yield (
            f"model.layers.{layer_id}.{name}.weight",
            extract_ternary(quantizer, _merged_weight(module)),
        )

    if not found:
        raise RuntimeError(f"Layer {layer_id} produced no ternary tensors")
