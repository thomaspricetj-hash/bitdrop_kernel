"""
SyntheticMind Autotuner Package
Provides:
    - autotune()
    - load_best(), save_best()
    - generate_configs()
    - load_build_metrics()
    - print_insights()
"""

from .tuner import autotune, print_insights
from .cache import load_best, save_best
from .search import generate_configs
from .build import METRICS_FILE as _METRICS_FILE
import json
import os


def load_build_metrics():
    if os.path.exists(_METRICS_FILE):
        try:
            with open(_METRICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


__all__ = [
    "autotune",
    "load_best",
    "save_best",
    "generate_configs",
    "load_build_metrics",
    "print_insights",
]
