"""Leak-safe public progress contract for prospective probability validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import DATA, OUTPUT
from .persistence import atomic_write_json, load_json
from .probability_contract import HORIZONS, ORDERED_MODEL_FAMILY
from .probability_model import canonical_hash

PUBLIC_STATUS_SCHEMA = "stock-radar-probability-forward-status"
PUBLIC_STATUS_SCHEMA_VERSION = 1
PUBLIC_STATUS_PATH = DATA / "probability_forward_status.json"
LATEST_PATH = OUTPUT / "latest.json"
DEFAULT_COHORT_ID = "ordered-vector-v1-forward-2026-08-18"
FROZEN_TRAINING_CUTOFF = "2026-07-10"
PUBLIC_STATES = ("collecting", "evaluating", "eligible_for_review")
_COUNT_FIELDS = (
    "eligible_prediction_counts",
    "matured_outcomes",
    "unresolved_outcomes",
)
_FORBIDDEN_DETAIL_KEYS = {
    "raw_ordered_probabilities",
    "derived_probabilities",
    "probabilities",
    "probability",
    "feature_vector",
    "coefficient",
    "intercept",
    "preprocessor",
    "vector_scaling",
    "entry_open",
    "exit_close",
    "gross_return",
    "net_return",
}
_PUBLIC_FIELDS = {
    "schema",
    "schema_version",
    "cohort_id",
    "frozen_at",
    "implementation_state",
    "training_cutoff",
    "weeks_captured",
    "eligible_prediction_counts",
    "matured_outcomes",
    "unresolved_outcomes",
    "next_maturity_dates",
    "minimum_required_dates",
    "status",
    "model_family",
    "retrospective_status",
    "shadow_classification",
    "actionable",
    "shadow_values_published",
    "detail_store",
    "integrity_status",
    "first_anchor_date",
    "latest_anchor_date",
    "schedule",
    "last_updated_at",
    "status_hash",
}
_SCHEDULE_FIELDS = {
    "first_weekly_anchor_not_before",
    "first_1m_maturity_estimate",
    "meaningful_1m_assessment_not_before",
    "final_12m_assessment_not_before",
    "schedule_note",
}


def _horizon_zeros() -> dict[str, int]:
    return {str(horizon): 0 for horizon in HORIZONS}


def initial_forward_validation_status() -> dict[str, Any]:
    value = {
        "schema": PUBLIC_STATUS_SCHEMA,
        "schema_version": PUBLIC_STATUS_SCHEMA_VERSION,
        "cohort_id": DEFAULT_COHORT_ID,
        "frozen_at": None,
        "implementation_state": "awaiting_local_freeze",
        "training_cutoff": FROZEN_TRAINING_CUTOFF,
        "weeks_captured": 0,
        "eligible_prediction_counts": _horizon_zeros(),
        "matured_outcomes": _horizon_zeros(),
        "unresolved_outcomes": _horizon_zeros(),
        "next_maturity_dates": {str(horizon): None for horizon in HORIZONS},
        "minimum_required_dates": {
            "weekly_anchors": 104,
            "distinct_forecast_dates": 26,
            "quarter_blocks": 8,
            "issuers": 200,
            "outcomes_per_class": 200,
        },
        "status": "collecting",
        "model_family": ORDERED_MODEL_FAMILY,
        "retrospective_status": "rejected",
        "shadow_classification": "REJECTED_SHADOW_NOT_FORECAST",
        "actionable": False,
        "shadow_values_published": False,
        "detail_store": "machine_local_only",
        "integrity_status": "not_initialized",
        "first_anchor_date": None,
        "latest_anchor_date": None,
        "schedule": {
            "first_weekly_anchor_not_before": "2026-08-21",
            "first_1m_maturity_estimate": "2026-09-22",
            "meaningful_1m_assessment_not_before": "2028-09-12",
            "final_12m_assessment_not_before": "2029-08-14",
            "schedule_note": (
                "Frozen NYSE weekday/holiday schedule v1; actual maturity still "
                "uses completed symbol sessions and cannot be labeled early."
            ),
        },
        "last_updated_at": None,
    }
    value["status_hash"] = canonical_hash(value)
    return value


def _walk_forbidden_detail(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_DETAIL_KEYS:
                raise ValueError(f"forward public status contains private detail at {path}.{key}")
            _walk_forbidden_detail(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_forbidden_detail(item, f"{path}[{index}]")


def validate_forward_validation_status(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("forward validation status root must be an object")
    if (
        value.get("schema") != PUBLIC_STATUS_SCHEMA
        or value.get("schema_version") != PUBLIC_STATUS_SCHEMA_VERSION
        or set(value) != _PUBLIC_FIELDS
    ):
        raise ValueError("unsupported forward validation status schema")
    stored_hash = value.get("status_hash")
    unhashed = {key: item for key, item in value.items() if key != "status_hash"}
    if not isinstance(stored_hash, str) or canonical_hash(unhashed) != stored_hash:
        raise ValueError("forward validation status hash mismatch")
    if (
        value.get("model_family") != ORDERED_MODEL_FAMILY
        or value.get("retrospective_status") != "rejected"
        or value.get("shadow_classification") != "REJECTED_SHADOW_NOT_FORECAST"
        or value.get("actionable") is not False
        or value.get("shadow_values_published") is not False
        or value.get("detail_store") != "machine_local_only"
        or value.get("implementation_state")
        not in {"awaiting_local_freeze", "frozen_shadow_monitoring"}
        or value.get("integrity_status")
        not in {"not_initialized", "signed_hash_chain_verified"}
        or value.get("status") not in PUBLIC_STATES
        or not isinstance(value.get("cohort_id"), str)
        or not value["cohort_id"].strip()
        or not isinstance(value.get("weeks_captured"), int)
        or value["weeks_captured"] < 0
    ):
        raise ValueError("forward validation status policy fields are invalid")
    expected_horizons = {str(horizon) for horizon in HORIZONS}
    for field in _COUNT_FIELDS:
        counts = value.get(field)
        if (
            not isinstance(counts, dict)
            or set(counts) != expected_horizons
            or any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in counts.values()
            )
        ):
            raise ValueError(f"forward validation status {field} is invalid")
    maturity = value.get("next_maturity_dates")
    if (
        not isinstance(maturity, dict)
        or set(maturity) != expected_horizons
        or any(item is not None and not isinstance(item, str) for item in maturity.values())
    ):
        raise ValueError("forward validation next maturity dates are invalid")
    minimums = value.get("minimum_required_dates")
    required_minimums = {
        "weekly_anchors": 104,
        "distinct_forecast_dates": 26,
        "quarter_blocks": 8,
        "issuers": 200,
        "outcomes_per_class": 200,
    }
    if minimums != required_minimums:
        raise ValueError("forward validation minimums differ from the frozen policy")
    schedule = value.get("schedule")
    if (
        not isinstance(schedule, dict)
        or set(schedule) != _SCHEDULE_FIELDS
        or any(not isinstance(item, str) or not item for item in schedule.values())
    ):
        raise ValueError("forward validation schedule is missing")
    _walk_forbidden_detail(value)
    return value


def load_forward_validation_status(
    path: Path = PUBLIC_STATUS_PATH,
    *,
    allow_default: bool = True,
) -> dict[str, Any]:
    value = load_json(path, default=None, expected_type=dict)
    if value is None:
        if not allow_default:
            raise ValueError(f"forward validation status is missing: {path}")
        value = initial_forward_validation_status()
    return validate_forward_validation_status(value)


def finalize_forward_validation_status(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("status_hash", None)
    result["status_hash"] = canonical_hash(result)
    return validate_forward_validation_status(result)


def inject_forward_validation_status(
    snapshot_path: Path = LATEST_PATH,
    status_path: Path = PUBLIC_STATUS_PATH,
) -> dict[str, Any]:
    snapshot = load_json(snapshot_path, required=True, expected_type=dict)
    status = load_forward_validation_status(status_path)
    snapshot["forward_validation_status"] = status
    atomic_write_json(snapshot_path, snapshot)
    return status


def public_status_json_contains_private_values(value: dict[str, Any]) -> bool:
    try:
        validate_forward_validation_status(value)
    except ValueError:
        return True
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).casefold()
    return any(f'"{key}"' in encoded for key in _FORBIDDEN_DETAIL_KEYS)


__all__ = [
    "DEFAULT_COHORT_ID",
    "FROZEN_TRAINING_CUTOFF",
    "LATEST_PATH",
    "PUBLIC_STATUS_PATH",
    "PUBLIC_STATUS_SCHEMA",
    "PUBLIC_STATUS_SCHEMA_VERSION",
    "finalize_forward_validation_status",
    "initial_forward_validation_status",
    "inject_forward_validation_status",
    "load_forward_validation_status",
    "public_status_json_contains_private_values",
    "validate_forward_validation_status",
]
