"""Checkpoint restoration and one-way merging for released CAT-Q parameters."""

import torch

from quantize.checkpoint import load_catq_parameters
from quantize.int_linear import QuantLinear
from quantize.int_linear_lora import LoRAQuantLinear


def _set_quantizer_options(args):
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "disable_zero_point": args.disable_zero_point,
        "drop_quant_mu": args.drop_quant_mu,
        "ter_scale_type": args.ter_scale_type,
        "init_scale_from_raw_weights": args.init_scale_from_raw_weights,
        "shift_mu": args.shift_mu,
        "learnable_scale": args.learnable_scale,
        "learnable_mu": args.learnable_mu,
        "learnable_round": args.learnable_round,
        "learnable_factor_act": args.learnable_factor_act,
        "init_round_thd": args.init_round_thd,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": args.act_symmetric,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = dict(args.act_quant_params, symmetric=False)
    args.k_quant_params = dict(args.act_quant_params, symmetric=False)
    args.v_quant_params = dict(args.act_quant_params, symmetric=False)
    args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}


def _model_adapter(lm, args):
    name = args.net.lower()
    layers = lm.model.model.layers
    if any(family in name for family in ("llama", "vicuna", "qwen", "mixtral")):
        from models.int_llama_layer import QuantLlamaDecoderLayer

        pairs = {"q_proj": "qkv", "o_proj": "out", "up_proj": "fc1"}
        if args.use_down_scale:
            pairs["down_proj"] = "fc2"
        return layers, QuantLlamaDecoderLayer, pairs
    if "ring" in name:
        from models.int_ring_layer import QuantBailingMoeV2DecoderLayer

        if args.use_scaling:
            raise ValueError("Equivalent scaling is not supported by the Ring adapter")
        return layers, QuantBailingMoeV2DecoderLayer, {}
    raise ValueError("Supported model families are LLaMA, Qwen, Mixtral, and Bailing/Ring")


def _register_scaling(qlayer, args, pairs, dtype):
    if not args.use_scaling:
        return
    if args.gqa_scales == "mean":
        model_dim = qlayer.self_attn.q_proj.out_features
    else:
        model_dim = qlayer.self_attn.k_proj.out_features
    qlayer.register_parameter(
        "qkt_smooth_scale",
        torch.nn.Parameter(torch.ones(model_dim, dtype=dtype), requires_grad=False),
    )
    for name, module in qlayer.named_modules():
        if not isinstance(module, (QuantLinear, LoRAQuantLinear)):
            continue
        for fragment, prefix in pairs.items():
            if fragment not in name:
                continue
            scale = torch.ones(module.in_features, dtype=dtype)
            shift = torch.zeros(module.in_features, dtype=dtype)
            if prefix == "out" and args.gqa_scales == "copy":
                shape = (
                    -1,
                    qlayer.self_attn.num_key_value_groups,
                    qlayer.self_attn.head_dim,
                )
                scale = scale.view(*shape).mean(dim=1).reshape(-1)
                shift = shift.view(*shape).mean(dim=1).reshape(-1)
            qlayer.register_parameter(
                f"{prefix}_smooth_shift",
                torch.nn.Parameter(shift, requires_grad=False),
            )
            qlayer.register_parameter(
                f"{prefix}_smooth_scale",
                torch.nn.Parameter(scale, requires_grad=False),
            )


def _native_layer_state(qlayer, native_layer):
    expected = native_layer.state_dict()
    merged = qlayer.state_dict()
    return {name: value for name, value in merged.items() if name in expected}


def _filter_inert_quantizer_bounds(layer_parameters, qlayer):
    expected = qlayer.state_dict()
    filtered = {}
    ignored = []
    for name, value in layer_parameters.items():
        if (
            name not in expected
            and name.endswith(
                (
                    "weight_quantizer.upbound_factor",
                    "weight_quantizer.lowbound_factor",
                )
            )
        ):
            ignored.append(name)
            continue
        filtered[name] = value
    return filtered, ignored


@torch.no_grad()
def merge_catq_checkpoint(lm, args, logger, ternary_sink=None):
    """Restore released per-layer parameters and merge them into native HF weights.

    `ternary_sink(layer_id, qlayer)` is called for every restored layer just
    before its weights are folded into floating point, which is the only point
    where the ternary codes and per-group scales are still available (see
    `quantize.ternary_export`).
    """
    if args.use_scaling and (args.export_model_path or ternary_sink is not None):
        raise ValueError(
            "Model export currently requires use_scaling: false; "
            "equivalent-scaling checkpoints remain supported for evaluation."
        )
    _set_quantizer_options(args)
    layers, decoder_cls, pairs = _model_adapter(lm, args)
    parameters = load_catq_parameters(args.checkpoint)
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16

    if args.quant_layer_list is None:
        quant_layers = set(range(len(layers)))
    else:
        quant_layers = {int(index) for index in args.quant_layer_list.split(",")}
    invalid = sorted(index for index in quant_layers if index < 0 or index >= len(layers))
    if invalid:
        raise ValueError(f"Invalid layer indices: {invalid}")

    for layer_id, native_layer in enumerate(layers):
        if layer_id not in quant_layers:
            continue
        if layer_id not in parameters:
            raise KeyError(f"Checkpoint does not contain parameters for layer {layer_id}")

        layer_parameters = parameters[layer_id]
        if not isinstance(layer_parameters, dict) or not layer_parameters:
            raise ValueError(f"Checkpoint parameters for layer {layer_id} are empty or invalid")

        qlayer = decoder_cls(
            config=lm.model.config,
            ori_layer=native_layer,
            layer_id=layer_id,
            args=args,
            quant_mode="fp16",
            use_lora=True,
            lora_attr={"lora_iter_num": 1},
        )
        _register_scaling(qlayer, args, pairs, torch.float32)
        layer_parameters, ignored_keys = _filter_inert_quantizer_bounds(
            layer_parameters,
            qlayer,
        )
        if ignored_keys:
            logger.info(
                "Ignored inert 16-bit quantizer bound keys in layer %d: %s",
                layer_id,
                ignored_keys[:10],
            )
        incompatible = qlayer.load_state_dict(layer_parameters, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Layer {layer_id} checkpoint has unsupported keys: {incompatible.unexpected_keys[:10]}"
            )

        qlayer.float()
        if ternary_sink is not None:
            ternary_sink(layer_id, qlayer)
        qlayer.update_quant_mode("weight_merge", args=args)
        if args.use_scaling:
            layers[layer_id] = qlayer.to(dtype=dtype)
        else:
            native_layer.load_state_dict(
                _native_layer_state(qlayer, native_layer),
                strict=False,
            )
            layers[layer_id] = native_layer.to(dtype=dtype)
        logger.info("Merged CAT-Q parameters into layer %d", layer_id)

    lm.model.to(dtype=dtype)
    lm.model.eval()
    for parameter in lm.model.parameters():
        parameter.requires_grad_(False)
    logger.info("Checkpoint merge complete")
    return lm
