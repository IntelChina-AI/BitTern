"""Logging helpers for evaluation commands."""

import logging
import sys
import time
from pathlib import Path


def create_logger(output_dir, name="catq"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    output_dir = Path(output_dir)
    file_handler = logging.FileHandler(
        output_dir / f"eval_{int(time.time())}.log",
        mode="a",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
