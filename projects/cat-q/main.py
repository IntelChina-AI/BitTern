import argparse
import random
from pathlib import Path

import yaml


def build_parser():
    parser = argparse.ArgumentParser(
        description="Load released CAT-Q parameters for evaluation or fake-quantized Hugging Face export."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML file describing the base model and checkpoint format",
    )
    parser.add_argument("--model", type=str, help="Hugging Face model name or local model directory")
    parser.add_argument(
        "--checkpoint",
        "--resume",
        dest="checkpoint",
        type=str,
        help="CAT-Q parameters.pth",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument(
        "--export_model_path",
        type=str,
        default=None,
        help="Directory for the fake-quantized HF model",
    )
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument(
        "--quant_layer_list",
        type=str,
        default=None,
        help="Comma-separated layer indices; defaults to all layers",
    )

    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--lm_eval_batch_size", type=str, default="auto:4")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--parallelize", action="store_true")

    parser.add_argument("--wbits", type=int, default=1)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--use_scaling", action="store_true")
    parser.add_argument("--gqa_scales", choices=["copy", "mean"], default="copy")
    parser.add_argument("--use_down_scale", action="store_true", default=True)
    parser.add_argument("--quant_gate", action="store_true", default=False)
    parser.add_argument("--only_lora_gate", action="store_true")
    parser.add_argument("--drop_quant_mu", action="store_true")
    parser.add_argument("--ter_scale_type", choices=["absmean", "variance"], default="absmean")
    parser.add_argument("--shift_mu", action="store_true")
    parser.add_argument("--learnable_scale", action="store_true")
    parser.add_argument("--learnable_mu", action="store_true")
    parser.add_argument("--learnable_round", action="store_true")
    parser.add_argument(
        "--learnable_factor_act",
        choices=["sigmoid", "double_sigmoid", "round_double_sigmoid", "softplus", "exp"],
        default="sigmoid",
    )
    parser.add_argument("--init_round_thd", type=float, default=0.5)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--disable_zero_point", action="store_true")
    parser.add_argument("--act_symmetric", action="store_true")
    parser.add_argument(
        "--a_dynamic_method",
        choices=["per_token", "per_token_bitnet"],
        default="per_token",
    )
    parser.add_argument("--w_dynamic_method", choices=["per_channel"], default="per_channel")
    parser.add_argument("--disable_quant", action="store_true")
    parser.add_argument("--use_bfloat16", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom Hugging Face model code; disabled by default",
    )
    return parser


def parse_arguments(argv=None):
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    ignored_config_keys = []
    if preliminary.config:
        with open(preliminary.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            parser.error("--config must contain a YAML mapping")
        known = {action.dest for action in parser._actions}
        evaluation_defaults = {key: value for key, value in config.items() if key in known}
        ignored_config_keys = sorted(set(config) - known)
        parser.set_defaults(**evaluation_defaults)

    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required (directly or through --config)")
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if not args.tasks and not args.export_model_path:
        parser.error("select at least one action: --tasks or --export_model_path")
    args.ignored_config_keys = ignored_config_keys
    return args


def main(argv=None):
    args = parse_arguments(argv)

    import numpy as np
    import torch

    from models.LMClass import LMClass
    from quantize.merge import merge_catq_checkpoint
    from quantize.utils import evaluate
    import utils

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(Path(args.output_dir))
    logger.info(args)
    if args.ignored_config_keys:
        logger.info(
            "Ignored training-only config keys: %s",
            ", ".join(args.ignored_config_keys),
        )

    if args.net is None:
        args.net = args.model.rstrip("/").split("/")[-1]
    args.quant_rate = 1.0

    lm = LMClass(args)
    merge_catq_checkpoint(lm, args, logger)

    if args.export_model_path:
        export_dir = Path(args.export_model_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        lm.model.save_pretrained(export_dir)
        lm.tokenizer.save_pretrained(export_dir)
        logger.info("Saved fake-quantized Hugging Face model to %s", export_dir)

    if args.tasks:
        evaluate(lm, args, logger)


if __name__ == "__main__":
    main()
