"""One-way equivalent transformations used during checkpoint merging."""

import torch


def truncate_number(value, threshold=1e-2):
    return torch.where(value.abs() < threshold, value.sign() * threshold, value)


def smooth_ln_fcs_inplace(norm, linears, scales, shifts):
    if not isinstance(linears, list):
        linears = [linears]
    if getattr(norm, "bias", None) is not None:
        norm.bias.sub_(shifts).div_(scales)
    else:
        if hasattr(norm, "bias"):
            del norm.bias
        norm.register_buffer("bias", -shifts / scales)
    norm.weight.div_(scales)
    for linear in linears:
        if getattr(linear, "bias", None) is not None:
            linear.bias.add_(linear.weight @ shifts)
        else:
            if hasattr(linear, "bias"):
                del linear.bias
            linear.register_buffer("bias", linear.weight @ shifts)
        linear.weight.mul_(scales.reshape(1, -1))


def _gqa_values(scales, shifts, num_key_value_groups, head_dim, args):
    if num_key_value_groups <= 1:
        return scales, shifts, scales, shifts
    if args.gqa_scales == "copy":
        expanded_scales = (
            scales.reshape(-1, head_dim)
            .repeat_interleave(num_key_value_groups, dim=0)
            .reshape(-1)
        )
        expanded_shifts = (
            shifts.reshape(-1, head_dim)
            .repeat_interleave(num_key_value_groups, dim=0)
            .reshape(-1)
        )
        return scales, shifts, expanded_scales, expanded_shifts
    kv_scales = scales.reshape(-1, num_key_value_groups, head_dim).mean(dim=1).reshape(-1)
    kv_shifts = shifts.reshape(-1, num_key_value_groups, head_dim).mean(dim=1).reshape(-1)
    return kv_scales, kv_shifts, scales, shifts


def smooth_fc_fc_inplace(
    fc1,
    fc2,
    scales,
    shifts,
    num_key_value_groups=1,
    head_dim=128,
    args=None,
):
    kv_scales, kv_shifts, expanded_scales, expanded_shifts = _gqa_values(
        scales, shifts, num_key_value_groups, head_dim, args
    )
    if getattr(fc1, "bias", None) is not None:
        fc1.bias.sub_(kv_shifts).div_(kv_scales)
    fc1.weight.div_(kv_scales.reshape(-1, 1))
    if getattr(fc2, "bias", None) is not None:
        fc2.bias.add_(fc2.weight @ expanded_shifts)
    else:
        if hasattr(fc2, "bias"):
            del fc2.bias
        fc2.register_buffer("bias", fc2.weight @ expanded_shifts)
    fc2.weight.mul_(expanded_scales.reshape(1, -1))


def smooth_q_k_inplace(
    q_proj,
    k_proj,
    scales,
    num_key_value_groups=1,
    head_dim=128,
    args=None,
):
    zero_shifts = torch.zeros_like(scales)
    kv_scales, _, expanded_scales, _ = _gqa_values(
        scales, zero_shifts, num_key_value_groups, head_dim, args
    )
    q_proj.weight.div_(expanded_scales.reshape(-1, 1))
    k_proj.weight.mul_(kv_scales.reshape(-1, 1))
    if getattr(q_proj, "bias", None) is not None:
        q_proj.bias.div_(expanded_scales)
    if getattr(k_proj, "bias", None) is not None:
        k_proj.bias.mul_(kv_scales)
