"""Logging helpers for the VCT project."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a process-wide configured logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
