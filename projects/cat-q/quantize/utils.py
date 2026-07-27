"""lm-eval entry point for merged CAT-Q models."""

import csv
import os
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from parallel_utils import get_max_memory_map


AVG5_TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande")


def _add_downstream_average(metrics):
    if all(task in metrics for task in AVG5_TASKS):
        metrics["avg-5"] = round(
            sum(metrics[task] for task in AVG5_TASKS) / len(AVG5_TASKS),
            4,
        )


def _place_model(lm, args, logger):
    if args.parallelize:
        no_split = [
            "LlamaDecoderLayer",
            "Qwen3MoeDecoderLayer",
            "BailingMoeV2DecoderLayer",
            "QuantLlamaDecoderLayer",
            "QuantBailingMoeV2DecoderLayer",
        ]
        balanced = get_balanced_memory(
            lm.model,
            max_memory=get_max_memory_map(),
            no_split_module_classes=no_split,
        )
        logger.info("Device memory plan: %s", balanced)
        device_map = infer_auto_device_map(
            lm.model,
            max_memory=balanced,
            no_split_module_classes=no_split,
        )
        lm.model = dispatch_model(lm.model, device_map=device_map)
    else:
        lm.model = lm.model.to(lm.device)


def _evaluate_tasks(lm, args, logger):
    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    eval_tasks = []
    for task in tasks:
        if task == "piqa":
            task_config_path = os.path.join(
                os.path.dirname(lm_eval.__file__),
                "tasks",
                "piqa",
                "piqa.yaml",
            )
            task_config = lm_eval_utils.load_yaml_config(task_config_path, mode="full")
            task_config["dataset_name"] = "plain_text"
            eval_tasks.append(task_config)
        else:
            eval_tasks.append(task)
    harness_model = HFLM(
        pretrained=lm.model,
        tokenizer=lm.tokenizer,
        batch_size=args.lm_eval_batch_size,
    )
    raw = lm_eval.simple_evaluate(
        harness_model,
        tasks=eval_tasks,
        batch_size=args.lm_eval_batch_size,
        limit=args.limit + 1 if args.limit >= 0 else None,
        task_manager=lm_eval.tasks.TaskManager(include_defaults=True),
    )["results"]
    metrics = {
        task: round(result.get("acc_norm,none", result.get("acc,none")), 4)
        for task, result in raw.items()
    }
    _add_downstream_average(metrics)
    logger.info("Downstream metrics: %s", metrics)
    return metrics


@torch.no_grad()
def evaluate(lm, args, logger):
    _place_model(lm, args, logger)
    results = _evaluate_tasks(lm, args, logger)
    result_path = os.path.join(args.output_dir, "results.csv")
    with open(result_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results)
        writer.writeheader()
        writer.writerow(results)
    return results
