"""Shared immutable target definitions without training/provider dependencies."""
from __future__ import annotations

ROUND_TRIP_COST = 0.003
HORIZONS = (21, 63, 126, 252)
THRESHOLD_GRIDS = {
    21: (3, 5, 10),
    63: (5, 10, 20),
    126: (10, 15, 25),
    252: (10, 20, 30),
}
CLASS_NAMES = ("down", "middle", "up")
ORDERED_CLASS_NAMES = tuple(f"bin_{index}" for index in range(7))
INDEPENDENT_MODEL_FAMILY = "independent-threshold-v1"
ORDERED_MODEL_FAMILY = "ordered-vector-v1"


def model_key(horizon: int, threshold_pct: int) -> str:
    return f"h{int(horizon)}_x{int(threshold_pct)}"


def label_column(horizon: int, threshold_pct: int) -> str:
    return f"label_{model_key(horizon, threshold_pct)}"


def ordered_label_column(horizon: int) -> str:
    return f"ordered_label_h{int(horizon)}"


def ordered_model_key(horizon: int) -> str:
    return f"h{int(horizon)}_ordered"


__all__ = [
    "CLASS_NAMES",
    "HORIZONS",
    "INDEPENDENT_MODEL_FAMILY",
    "ORDERED_CLASS_NAMES",
    "ORDERED_MODEL_FAMILY",
    "ROUND_TRIP_COST",
    "THRESHOLD_GRIDS",
    "label_column",
    "model_key",
    "ordered_label_column",
    "ordered_model_key",
]
