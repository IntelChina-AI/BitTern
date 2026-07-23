"""Small helpers for Accelerate-based evaluation device placement."""

import torch


def get_max_memory_map(ratio=0.95):
    """Return a conservative per-GPU memory budget for Accelerate."""
    return {
        index: f"{int(torch.cuda.get_device_properties(index).total_memory * ratio / 1024**3)}GiB"
        for index in range(torch.cuda.device_count())
    }
