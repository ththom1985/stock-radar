"""Prospective, shadow-only forward validation for rejected ordered models.

The detailed ledger, fitted artifact, forecasts, outcomes, and reports stay below
the ignored ``data/probability_forward`` tree.  Only count/progress status is
eligible for the public snapshot.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .config import DATA, ROOT
from .fetch import PriceFetchResult, fetch_prices_with_status
from .persistence import atomic_write_json, load_json
from .probability_contract import (
    CLASS_NAMES,
    HORIZONS,
    ORDERED_CLASS_NAMES,
    ORDERED_MODEL_FAMILY,
    ROUND_TRIP_COST,
    THRESHOLD_GRIDS,
    ordered_label_column,
    ordered_model_key,
)
from .probability_dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_CACHE,
    EligibilityResult,
    classify_material_move,
    classify_ordered_move,
    cost_adjusted_material_return,
    dataset_content_hash,
    dataset_content_summary,
    read_exact_dataset_cache,
    select_eligible_universe,
)
from .probability_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    MIN_HISTORY_BARS,
    dependency_versions,
    feature_schema_hash,
    latest_probability_features,
    probability_code_hash,
)
from .probability_inference import PARTITION, ProbabilityArtifactError
from .probability_model import (
    DEFAULT_SEED,
    STRICT_ACCEPTANCE_GATES,
    _binary_calibration_fit,
    assess_ood,
    canonical_hash,
    multiclass_brier,
    multiclass_log_loss,
    two_way_cluster_bootstrap,
)
from .probability_ordered import (
    ADAPTIVE_MIN_SUPPORTED_BINS,
    ORDERED_MODEL_VERSION,
    ORDERED_PUBLISH_TRANSFORM,
    VECTOR_CALIBRATION_VERSION,
    VECTOR_PENALTY_GRID,
    adaptive_classwise_reliability,
    assert_exact_ordered_monotonicity,
    derive_threshold_probabilities,
    fit_ordered_multinomial_model,
    fit_ordered_vector_calibration,
    ordered_labels_to_threshold_labels,
    predict_ordered_probabilities,
    regime_support,
)
from .probability_forward_public import (
    DEFAULT_COHORT_ID,
    FROZEN_TRAINING_CUTOFF,
    LATEST_PATH,
    PUBLIC_STATUS_PATH,
    finalize_forward_validation_status,
    initial_forward_validation_status,
    inject_forward_validation_status,
    load_forward_validation_status,
)
from .probability_forward_store import (
    ForwardLedger,
    ForwardStaleSnapshotError,
    GENESIS_CHAIN_HASH,
    canonical_json_bytes,
    deterministic_gzip_json,
    load_or_create_signing_key,
    make_capture_envelope,
    read_gzip_json,
    sha256_bytes,
    signed_digest,
    signing_key_id,
    validate_capture_envelope,
    write_immutable_bytes,
    write_immutable_json,
)

FORWARD_ROOT = DATA / "probability_forward"
SHADOW_ARTIFACT_SCHEMA = "stock-radar-probability-shadow-artifact"
SHADOW_ARTIFACT_SCHEMA_VERSION = 1
PREREGISTRATION_SCHEMA = "stock-radar-probability-forward-preregistration"
PREREGISTRATION_SCHEMA_VERSION = 1
SHADOW_CLASSIFICATION = "REJECTED_SHADOW_NOT_FORECAST"
SAFE_CAPTURE_UTC = time(23, 0)
IMPLEMENTATION_CREATED_AT_UTC = datetime(
    2026,
    8,
    18,
    8,
    14,
    29,
    tzinfo=timezone.utc,
)
CAPTURE_HISTORY_PERIOD = "3y"
EVALUATION_HISTORY_PERIOD = "5y"
FORWARD_CODE_FILES = (
    "src/probability_forward.py",
    "src/probability_forward_public.py",
    "src/probability_forward_store.py",
)
PROSPECTIVE_MINIMUMS = {
    "weekly_anchors": 104,
    "distinct_forecast_dates": 26,
    "quarter_blocks": 8,
    "issuers": 200,
    "outcomes_per_class": 200,
    "fixed_prediction_bootstrap_repetitions": 1000,
}
CANONICAL_MODELS_PATH = DATA / "probability_models.json"
CANONICAL_VALIDATION_PATH = DATA / "probability_validation.json"
ORDERED_MODELS_PATH = DATA / "probability_experiments" / "ordered-vector-v1_models.json"
ORDERED_VALIDATION_PATH = (
    DATA / "probability_experiments" / "ordered-vector-v1_validation.json"
)
ORDERED_PREREGISTRATION_PATH = (
    DATA / "probability_experiments" / "ordered-vector-v1_preregistration.json"
)
_COHORT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{5,95}$")


@dataclass(frozen=True)
class CohortPaths:
    root: Path
    cohort: Path
    preregistration: Path
    artifact: Path
    ledger: Path
    signing_key: Path
    manifest: Path
    predictions: Path
    checkpoints: Path
    backups: Path
    candidate_report: Path


def cohort_paths(
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
) -> CohortPaths:
    if not _COHORT_PATTERN.fullmatch(str(cohort_id)):
        raise ValueError("cohort id must be a stable lower-case filesystem identifier")
    cohort = Path(root) / "cohorts" / cohort_id
    return CohortPaths(
        root=Path(root),
        cohort=cohort,
        preregistration=cohort / "preregistration.json",
        artifact=cohort / "shadow_artifact.json",
        ledger=cohort / "forward.sqlite3",
        signing_key=cohort / "manifest-signing.key",
        manifest=cohort / "manifest.json",
        predictions=cohort / "predictions",
        checkpoints=cohort / "fit_checkpoints",
        backups=cohort / "backups",
        candidate_report=cohort / "candidate_report.json",
    )


def _parse_utc(value: str | datetime) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _system_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trusted_now(
    *,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
) -> datetime:
    if clock is not None and not test_mode:
        raise RuntimeError("an injected clock is permitted only in development test-mode")
    now = clock() if clock is not None else _system_utc_now()
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise RuntimeError("trusted clock must return a timezone-aware datetime")
    now = now.astimezone(timezone.utc)
    if not test_mode and now < IMPLEMENTATION_CREATED_AT_UTC:
        raise RuntimeError("system UTC time predates the forward implementation")
    return now


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forward_code_binding(root: Path = ROOT) -> dict[str, Any]:
    files: dict[str, str] = {}
    for name in FORWARD_CODE_FILES:
        path = Path(root) / Path(name)
        if not path.exists():
            raise RuntimeError(f"forward source file is missing: {name}")
        canonical = (
            path.read_text(encoding="utf-8-sig")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .encode("utf-8")
        )
        files[name] = sha256_bytes(canonical)
    return {
        "files": files,
        "forward_code_hash": canonical_hash(files),
    }


def _finalize_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = canonical_hash(result)
    return result


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month = (
        date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    )
    result = next_month - timedelta(days=1)
    return result - timedelta(days=(result.weekday() - weekday) % 7)


def _observed_fixed(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nyse_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed_fixed(date(year, 6, 19)))
    return holidays


def _is_projected_us_session(value: date) -> bool:
    return value.weekday() < 5 and value not in (
        _nyse_holidays(value.year - 1)
        | _nyse_holidays(value.year)
        | _nyse_holidays(value.year + 1)
    )


def _project_us_session(feature_date: date, horizon: int) -> date:
    current = feature_date
    completed = 0
    while completed < int(horizon):
        current += timedelta(days=1)
        if _is_projected_us_session(current):
            completed += 1
    return current


def _latest_projected_session_on_or_before(value: date) -> date:
    current = value
    while not _is_projected_us_session(current):
        current -= timedelta(days=1)
    return current


def _prospective_schedule(freeze_date: date) -> dict[str, str]:
    days_to_friday = (4 - freeze_date.weekday()) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    friday = freeze_date + timedelta(days=days_to_friday)
    first_anchor = _latest_projected_session_on_or_before(friday)
    while first_anchor <= freeze_date:
        friday += timedelta(weeks=1)
        first_anchor = _latest_projected_session_on_or_before(friday)
    anchor_104 = _latest_projected_session_on_or_before(
        friday + timedelta(weeks=103)
    )
    return {
        "first_weekly_anchor_not_before": first_anchor.isoformat(),
        "first_1m_maturity_estimate": _project_us_session(
            first_anchor, 21
        ).isoformat(),
        "meaningful_1m_assessment_not_before": _project_us_session(
            anchor_104, 21
        ).isoformat(),
        "final_12m_assessment_not_before": _project_us_session(
            anchor_104, 252
        ).isoformat(),
        "schedule_note": (
            "Frozen NYSE weekday/holiday schedule v1; actual maturity still uses "
            "completed symbol sessions and cannot be labeled early."
        ),
    }


def _verify_hash(value: dict[str, Any], field: str, label: str) -> None:
    stored = value.get(field)
    unhashed = {key: item for key, item in value.items() if key != field}
    if not isinstance(stored, str) or canonical_hash(unhashed) != stored:
        raise RuntimeError(f"{label} hash mismatch")


def _load_retrospective_context() -> dict[str, Any]:
    canonical_models = load_json(
        CANONICAL_MODELS_PATH, required=True, expected_type=dict
    )
    canonical_validation = load_json(
        CANONICAL_VALIDATION_PATH, required=True, expected_type=dict
    )
    ordered_models = load_json(ORDERED_MODELS_PATH, required=True, expected_type=dict)
    ordered_validation = load_json(
        ORDERED_VALIDATION_PATH, required=True, expected_type=dict
    )
    ordered_preregistration = load_json(
        ORDERED_PREREGISTRATION_PATH, required=True, expected_type=dict
    )
    if (
        canonical_models.get("model_family") != ORDERED_MODEL_FAMILY
        or canonical_models.get("production_status") != "withheld"
        or canonical_models.get("accepted_model_keys") != []
        or canonical_models.get("models") != {}
        or canonical_validation.get("status") != "no_model_passed"
        or int(canonical_validation.get("accepted_model_count") or 0) != 0
        or ordered_models.get("accepted_model_keys") != []
        or ordered_models.get("models") != {}
        or ordered_validation.get("status") != "no_model_passed"
        or int(ordered_validation.get("accepted_model_count") or 0) != 0
    ):
        raise RuntimeError(
            "retrospective artifacts no longer describe a fully rejected ordered grid"
        )
    cutoff = str(canonical_models.get("training_cutoff") or "")
    binding = canonical_models.get("dataset_binding") or {}
    if (
        not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff)
        or binding != ordered_validation.get("dataset_binding")
        or not isinstance(binding.get("dataset_content_hash"), str)
    ):
        raise RuntimeError("frozen retrospective dataset binding is inconsistent")
    return {
        "training_cutoff": cutoff,
        "dataset_binding": binding,
        "canonical_artifact_hash": canonical_models.get("artifact_hash"),
        "canonical_artifact_document_hash": canonical_hash(canonical_models),
        "canonical_validation_document_hash": canonical_hash(canonical_validation),
        "ordered_experiment_artifact_hash": ordered_models.get("artifact_hash"),
        "ordered_validation_document_hash": canonical_hash(ordered_validation),
        "ordered_preregistration_document_hash": canonical_hash(
            ordered_preregistration
        ),
        "rejection_reason_hash": canonical_hash(
            {
                key: (report.get("acceptance") or {}).get("reasons") or []
                for key, report in sorted(
                    (ordered_validation.get("models") or {}).items()
                )
            }
        ),
        "retrospective_status": "rejected",
        "accepted_model_count": 0,
    }


def _load_frozen_dataset(
    cache_dir: Path,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path(cache_dir) / "dataset_manifest.json"
    manifest = load_json(manifest_path, required=True, expected_type=dict)
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("feature_version") != FEATURE_VERSION
        or manifest.get("feature_schema_hash") != feature_schema_hash()
        or manifest.get("dataset_content_hash")
        != context["dataset_binding"]["dataset_content_hash"]
    ):
        raise RuntimeError("cached dataset is not the frozen ordered validation dataset")
    dataset_path = Path(cache_dir) / str(manifest.get("file") or "")
    if (
        not dataset_path.is_file()
        or _sha256_file(dataset_path) != manifest.get("sha256")
        or manifest.get("storage_format")
        != "trusted-local-pandas-pickle-protocol5-gzip"
    ):
        raise RuntimeError("frozen dataset cache checksum/format is invalid")
    dataset = read_exact_dataset_cache(dataset_path)
    if dataset_content_hash(dataset) != manifest["dataset_content_hash"]:
        raise RuntimeError("frozen dataset content hash mismatch")
    dates = pd.to_datetime(dataset["feature_date"], errors="coerce")
    if dates.isna().any() or dates.max().date().isoformat() != context["training_cutoff"]:
        raise RuntimeError("frozen dataset cutoff does not match the artifact")
    return dataset, manifest


def _make_preregistration(
    *,
    cohort_id: str,
    frozen_at: datetime,
    cutoff: str,
    dataset: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    context: dict[str, Any],
    key: bytes,
    test_mode: bool,
) -> dict[str, Any]:
    code = forward_code_binding()
    value = {
        "schema": PREREGISTRATION_SCHEMA,
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "classification": SHADOW_CLASSIFICATION,
        "shadow_only": True,
        "actionable": False,
        "development_test_mode": bool(test_mode),
        "implementation_created_at": IMPLEMENTATION_CREATED_AT_UTC.isoformat(
            timespec="seconds"
        ),
        "frozen_at": frozen_at.isoformat(timespec="seconds"),
        "implementation_deployed_at": frozen_at.isoformat(timespec="seconds"),
        "training_data_cutoff": cutoff,
        "first_anchor_policy": {
            "anchor": "final completed session in one ISO week",
            "safe_capture_utc": SAFE_CAPTURE_UTC.isoformat(timespec="minutes"),
            "feature_date_must_be_strictly_after_freeze_date": True,
            "capture_week_must_equal_anchor_iso_week": True,
            "one_anchor_per_iso_week": True,
            "no_backfill": True,
        },
        "frozen_model_specification": {
            "model_family": ORDERED_MODEL_FAMILY,
            "model_version": ORDERED_MODEL_VERSION,
            "class_order": list(ORDERED_CLASS_NAMES),
            "horizons_sessions": list(HORIZONS),
            "threshold_grids_pct": {
                str(horizon): list(THRESHOLD_GRIDS[horizon])
                for horizon in HORIZONS
            },
            "round_trip_cost_bps": int(ROUND_TRIP_COST * 10_000),
            "features": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "c_value": 0.1,
            "seed": DEFAULT_SEED,
            "calibration_version": VECTOR_CALIBRATION_VERSION,
            "vector_penalty_grid": list(VECTOR_PENALTY_GRID),
            "publish_transform": ORDERED_PUBLISH_TRANSFORM,
        },
        "fit_protocol": {
            "selection": "none; all four rejected horizons are fit once",
            "model_train": (
                "all labeled rows before the final one-year calibration interval; "
                "max_exit_date purged seven days before calibration"
            ),
            "calibration": (
                "final labeled feature year; OOD-supported rows only; first nine "
                "months fit each fixed vector penalty, final three months select by "
                "seven-class log loss with Brier tie-break, then full-year refit"
            ),
            "outcomes_used_for_retuning": False,
            "future_retuning_permitted": False,
            "change_policy": (
                "any model, feature, gate, code, dependency, or data change requires "
                "a new separately named preregistration and cohort"
            ),
        },
        "outcome_protocol": {
            "entry": "first adjusted/raw-equivalent open strictly after feature session",
            "exit": "adjusted close at feature session plus H symbol sessions",
            "corporate_actions": "Yahoo adjustment factor convention used by training",
            "early_labels_forbidden": True,
            "missing_or_halted": "remain unresolved with an explicit attempt reason",
        },
        "prospective_schedule": _prospective_schedule(frozen_at.date()),
        "prospective_release_policy": {
            "minimums": PROSPECTIVE_MINIMUMS,
            "metric_gates": STRICT_ACCEPTANCE_GATES,
            "retrospective_metrics": "context_only",
            "automatic_promotion": False,
            "independent_review_required": True,
        },
        "retrospective_validation": {
            "status": "rejected",
            "accepted_model_count": 0,
            "links": {
                "preregistration": (
                    "data/probability_experiments/"
                    "ordered-vector-v1_preregistration.json"
                ),
                "validation": (
                    "data/probability_experiments/ordered-vector-v1_validation.json"
                ),
                "empty_models": (
                    "data/probability_experiments/ordered-vector-v1_models.json"
                ),
                "canonical_validation": "data/probability_validation.json",
                "canonical_models": "data/probability_models.json",
            },
            "hashes": {
                key_name: context[key_name]
                for key_name in (
                    "canonical_artifact_hash",
                    "canonical_artifact_document_hash",
                    "canonical_validation_document_hash",
                    "ordered_experiment_artifact_hash",
                    "ordered_validation_document_hash",
                    "ordered_preregistration_document_hash",
                    "rejection_reason_hash",
                )
            },
        },
        "frozen_hashes": {
            "ordered_core_code_hash": probability_code_hash(),
            "forward_code_hash": code["forward_code_hash"],
            "forward_source_files": code["files"],
            "dependency_versions": dependency_versions(),
            "dependency_versions_hash": canonical_hash(dependency_versions()),
            "dataset_content_hash": dataset_content_hash(dataset),
            "dataset_manifest_hash": canonical_hash(dataset_manifest),
            "dataset_manifest_source_code_hash": dataset_manifest.get("code_hash"),
            "dataset_summary": dataset_content_summary(dataset),
            "retrospective_dataset_binding": context["dataset_binding"],
            "frozen_provider_context": (
                context["dataset_binding"].get("provider") or {}
            ),
        },
        "signing": {
            "algorithm": "HMAC-SHA256",
            "key_id": signing_key_id(key),
            "key_storage": "ignored machine-local cohort file; back it up separately",
        },
    }
    return _finalize_hash(value, "preregistration_hash")


def validate_preregistration(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != PREREGISTRATION_SCHEMA
        or value.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION
        or value.get("classification") != SHADOW_CLASSIFICATION
        or value.get("shadow_only") is not True
        or value.get("actionable") is not False
        or not isinstance(value.get("development_test_mode"), bool)
        or value.get("implementation_created_at")
        != IMPLEMENTATION_CREATED_AT_UTC.isoformat(timespec="seconds")
        or (value.get("retrospective_validation") or {}).get("status") != "rejected"
        or (value.get("retrospective_validation") or {}).get(
            "accepted_model_count"
        )
        != 0
    ):
        raise RuntimeError("invalid forward preregistration policy")
    _verify_hash(value, "preregistration_hash", "forward preregistration")
    frozen = value.get("frozen_hashes") or {}
    specification = value.get("frozen_model_specification") or {}
    current = forward_code_binding()
    if (
        frozen.get("ordered_core_code_hash") != probability_code_hash()
        or frozen.get("forward_code_hash") != current["forward_code_hash"]
        or frozen.get("forward_source_files") != current["files"]
        or frozen.get("dependency_versions") != dependency_versions()
        or specification.get("feature_version") != FEATURE_VERSION
        or specification.get("feature_schema_hash") != feature_schema_hash()
        or specification.get("features") != list(FEATURE_NAMES)
        or specification.get("horizons_sessions") != list(HORIZONS)
        or specification.get("vector_penalty_grid") != list(VECTOR_PENALTY_GRID)
    ):
        raise RuntimeError(
            "forward implementation/specification changed; create a new cohort id"
        )
    return value


def _fit_shadow_model(
    dataset: pd.DataFrame,
    *,
    horizon: int,
    preregistration: dict[str, Any],
    trained_at: str,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    target = ordered_label_column(horizon)
    valid_target = pd.to_numeric(
        dataset[target], errors="coerce"
    ).isin(range(len(ORDERED_CLASS_NAMES)))
    excluded_invalid_labels = int(
        (dataset[target].notna() & ~valid_target).sum()
    )
    eligible = dataset.loc[valid_target].copy()
    if eligible.empty:
        raise RuntimeError(f"frozen dataset has no mature h{horizon} labels")
    dates = pd.to_datetime(eligible["feature_date"]).dt.tz_localize(None)
    max_exit = pd.to_datetime(eligible["max_exit_date"]).dt.tz_localize(None)
    calibration_end = dates.max() + pd.Timedelta(days=1)
    calibration_start = calibration_end - pd.DateOffset(years=1)
    train_mask = (dates < calibration_start) & (
        max_exit < calibration_start - pd.Timedelta(days=7)
    )
    calibration_mask = (dates >= calibration_start) & (dates < calibration_end)
    train = eligible.loc[train_mask].copy()
    calibration = eligible.loc[calibration_mask].copy()
    labels = train[target].to_numpy(dtype=int)
    if set(np.unique(labels)) != set(range(len(ORDERED_CLASS_NAMES))):
        raise RuntimeError(f"frozen h{horizon} training segment misses an ordered bin")
    model = fit_ordered_multinomial_model(
        train.loc[:, FEATURE_NAMES],
        labels,
        c_value=0.1,
        seed=seed,
    )
    calibrated = fit_ordered_vector_calibration(
        model,
        calibration,
        target=target,
        calibration_interval_start=calibration_start,
        penalty_grid=VECTOR_PENALTY_GRID,
    )
    model = calibrated["model"]
    thresholds = THRESHOLD_GRIDS[horizon]
    baseline_rates: dict[str, dict[str, float]] = {}
    for threshold_index, threshold in enumerate(thresholds):
        threshold_labels = ordered_labels_to_threshold_labels(labels, threshold_index)
        counts = np.bincount(threshold_labels, minlength=3).astype(float)
        baseline_rates[str(threshold)] = {
            name: float(counts[index] / counts.sum())
            for index, name in enumerate(CLASS_NAMES)
        }
    volatility = train["spy_vol_60"].to_numpy(dtype=float)
    volatility = volatility[np.isfinite(volatility)]
    regime_terciles = np.quantile(volatility, [1 / 3, 2 / 3]).astype(float)
    model.update(
        {
            "model_family": ORDERED_MODEL_FAMILY,
            "model_key": ordered_model_key(horizon),
            "model_role": "shadow_only_rejected_retrospective",
            "classification": SHADOW_CLASSIFICATION,
            "shadow_only": True,
            "actionable": False,
            "accepted": False,
            "retrospective_validation_accepted": False,
            "horizon_sessions": int(horizon),
            "thresholds_pct": list(thresholds),
            "round_trip_cost_bps": int(ROUND_TRIP_COST * 10_000),
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
            "trained_at": trained_at,
            "training_data_cutoff": preregistration["training_data_cutoff"],
            "label_feature_cutoff": dates.max().date().isoformat(),
            "excluded_invalid_ordered_labels": excluded_invalid_labels,
            "release_train_start": pd.to_datetime(train["feature_date"])
            .min()
            .date()
            .isoformat(),
            "release_train_end": pd.to_datetime(train["feature_date"])
            .max()
            .date()
            .isoformat(),
            "release_calibration_start": pd.Timestamp(calibration_start)
            .date()
            .isoformat(),
            "release_calibration_end": pd.Timestamp(
                calibration_end - pd.Timedelta(days=1)
            )
            .date()
            .isoformat(),
            "release_calibration_candidate_count": int(len(calibration)),
            "release_calibration_scored_count": int(calibrated["scored_count"]),
            "release_calibration_coverage": float(calibrated["coverage"]),
            "baseline_rates_by_threshold": baseline_rates,
            "forward_regime_definition": {
                "trend": "spy_price_sma200 >= 0 is trend_up, otherwise trend_down",
                "volatility_feature": "spy_vol_60",
                "volatility_terciles": regime_terciles.tolist(),
                "fit_scope": "frozen final model training segment",
            },
            "preregistration_hash": preregistration["preregistration_hash"],
            "fit_protocol_hash": canonical_hash(preregistration["fit_protocol"]),
        }
    )
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    return model


def validate_shadow_artifact(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != SHADOW_ARTIFACT_SCHEMA
        or value.get("schema_version") != SHADOW_ARTIFACT_SCHEMA_VERSION
        or value.get("classification") != SHADOW_CLASSIFICATION
        or value.get("shadow_only") is not True
        or value.get("actionable") is not False
        or value.get("production_loader_compatible") is not False
        or value.get("model_family") != ORDERED_MODEL_FAMILY
        or not isinstance(value.get("development_test_mode"), bool)
        or "accepted_model_keys" in value
        or value.get("retrospective_validation", {}).get("accepted_model_count") != 0
    ):
        raise RuntimeError("invalid shadow artifact isolation contract")
    _verify_hash(value, "artifact_hash", "shadow artifact")
    if value.get("forward_code_hash") != forward_code_binding()["forward_code_hash"]:
        raise RuntimeError(
            "shadow artifact forward code changed; start a separately named cohort"
        )
    if value.get("ordered_core_code_hash") != probability_code_hash():
        raise RuntimeError(
            "shadow artifact ordered engine changed; start a separately named cohort"
        )
    models = value.get("models")
    expected = {ordered_model_key(horizon) for horizon in HORIZONS}
    if not isinstance(models, dict) or set(models) != expected:
        raise RuntimeError("shadow artifact must contain all four frozen horizons")
    for key, model in models.items():
        if (
            not isinstance(model, dict)
            or model.get("model_key") != key
            or model.get("shadow_only") is not True
            or model.get("actionable") is not False
            or model.get("accepted") is not False
            or model.get("retrospective_validation_accepted") is not False
            or model.get("preregistration_hash") != value.get("preregistration_hash")
        ):
            raise RuntimeError(f"shadow model isolation is invalid for {key}")
        _verify_hash(model, "model_hash", f"shadow model {key}")
    return value


def load_shadow_artifact(path: Path) -> dict[str, Any]:
    return validate_shadow_artifact(
        load_json(path, required=True, expected_type=dict)
    )


def freeze_shadow_cohort(
    dataset: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    context: dict[str, Any],
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    signing_key: bytes | None = None,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Preregister first, fit each horizon once/checkpointed, then seal an artifact."""
    paths = cohort_paths(cohort_id, root)
    trusted_now = _trusted_now(clock=clock, test_mode=test_mode)
    cutoff = str(context["training_cutoff"])
    if date.fromisoformat(cutoff) >= trusted_now.date():
        raise RuntimeError("training cutoff must be strictly earlier than trusted freeze")
    dates = pd.to_datetime(dataset["feature_date"], errors="coerce")
    if (
        dates.isna().any()
        or dates.max().date().isoformat() != cutoff
        or (dates.dt.date > date.fromisoformat(cutoff)).any()
    ):
        raise RuntimeError("freeze dataset is not bounded by the exact frozen cutoff")
    key = signing_key or load_or_create_signing_key(paths.signing_key)
    if signing_key is not None:
        write_immutable_bytes(paths.signing_key, signing_key)
    if paths.preregistration.exists():
        preregistration = validate_preregistration(
            load_json(paths.preregistration, required=True, expected_type=dict)
        )
        now = _parse_utc(preregistration["frozen_at"])
        if now > trusted_now:
            raise RuntimeError("existing cohort freeze time is later than trusted UTC")
        if not test_mode and now < IMPLEMENTATION_CREATED_AT_UTC:
            raise RuntimeError(
                "existing cohort freeze predates implementation creation"
            )
        if (
            preregistration.get("cohort_id") != cohort_id
            or preregistration.get("development_test_mode") is not bool(test_mode)
            or preregistration.get("training_data_cutoff") != cutoff
            or preregistration.get("signing", {}).get("key_id")
            != signing_key_id(key)
            or preregistration.get("frozen_hashes", {}).get(
                "dataset_content_hash"
            )
            != dataset_content_hash(dataset)
        ):
            raise RuntimeError(
                "existing cohort preregistration differs; use a new cohort id"
            )
    else:
        now = trusted_now
        preregistration = _make_preregistration(
            cohort_id=cohort_id,
            frozen_at=now,
            cutoff=cutoff,
            dataset=dataset,
            dataset_manifest=dataset_manifest,
            context=context,
            key=key,
            test_mode=test_mode,
        )
        write_immutable_json(paths.preregistration, preregistration)

    models: dict[str, Any] = {}
    for horizon in HORIZONS:
        key_name = ordered_model_key(horizon)
        checkpoint_path = paths.checkpoints / f"{key_name}.json"
        if checkpoint_path.exists():
            checkpoint = load_json(
                checkpoint_path, required=True, expected_type=dict
            )
            model = checkpoint.get("model")
            if (
                checkpoint.get("preregistration_hash")
                != preregistration["preregistration_hash"]
                or checkpoint.get("dataset_content_hash")
                != preregistration["frozen_hashes"]["dataset_content_hash"]
                or not isinstance(model, dict)
            ):
                raise RuntimeError(f"fit checkpoint mismatch for {key_name}")
            _verify_hash(model, "model_hash", f"fit checkpoint {key_name}")
        else:
            model = _fit_shadow_model(
                dataset,
                horizon=horizon,
                preregistration=preregistration,
                trained_at=now.isoformat(timespec="seconds"),
            )
            checkpoint = {
                "schema": "stock-radar-probability-shadow-fit-checkpoint",
                "schema_version": 1,
                "classification": SHADOW_CLASSIFICATION,
                "preregistration_hash": preregistration["preregistration_hash"],
                "dataset_content_hash": preregistration["frozen_hashes"][
                    "dataset_content_hash"
                ],
                "model": model,
            }
            write_immutable_json(checkpoint_path, checkpoint)
        models[key_name] = model

    artifact = _finalize_hash(
        {
            "schema": SHADOW_ARTIFACT_SCHEMA,
            "schema_version": SHADOW_ARTIFACT_SCHEMA_VERSION,
            "classification": SHADOW_CLASSIFICATION,
            "artifact_role": "prospective_validation_shadow_only",
            "shadow_only": True,
            "actionable": False,
            "development_test_mode": bool(test_mode),
            "production_loader_compatible": False,
            "cohort_id": cohort_id,
            "created_at": now.isoformat(timespec="seconds"),
            "training_data_cutoff": cutoff,
            "model_family": ORDERED_MODEL_FAMILY,
            "model_keys": [ordered_model_key(horizon) for horizon in HORIZONS],
            "horizons_sessions": list(HORIZONS),
            "threshold_grids_pct": {
                str(horizon): list(THRESHOLD_GRIDS[horizon])
                for horizon in HORIZONS
            },
            "class_order": list(ORDERED_CLASS_NAMES),
            "supported_partition": PARTITION,
            "preregistration_hash": preregistration["preregistration_hash"],
            "ordered_core_code_hash": probability_code_hash(),
            "forward_code_hash": forward_code_binding()["forward_code_hash"],
            "dataset_content_hash": preregistration["frozen_hashes"][
                "dataset_content_hash"
            ],
            "fit_protocol": preregistration["fit_protocol"],
            "retrospective_validation": {
                "status": "rejected",
                "accepted_model_count": 0,
                "validation_document_hash": context[
                    "ordered_validation_document_hash"
                ],
                "rejection_reason_hash": context["rejection_reason_hash"],
                "link": (
                    "data/probability_experiments/ordered-vector-v1_validation.json"
                ),
            },
            "canonical_production_artifact": {
                "path": "data/probability_models.json",
                "artifact_hash": context["canonical_artifact_hash"],
                "overwritten": False,
            },
            "signing_key_id": signing_key_id(key),
            "models": models,
        },
        "artifact_hash",
    )
    validate_shadow_artifact(artifact)
    write_immutable_json(paths.artifact, artifact)
    if load_shadow_artifact(paths.artifact) != artifact:
        raise RuntimeError("persisted shadow artifact differs from fitted artifact")

    # A normal production loader must fail on the deliberately distinct schema.
    from .probability_inference import validate_probability_artifact

    try:
        validate_probability_artifact(artifact)
    except ProbabilityArtifactError:
        pass
    else:
        raise AssertionError("production probability loader accepted a shadow artifact")

    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.export_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        ledger.backup(
            paths.backups,
            manifest_path=paths.manifest,
            candidate_report_path=paths.candidate_report,
        )
    return artifact


def freeze_from_cache(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    now = _trusted_now()
    context = _load_retrospective_context()
    if context["training_cutoff"] != FROZEN_TRAINING_CUTOFF:
        raise RuntimeError(
            "canonical cutoff differs from the expected frozen cutoff; inspect before "
            "creating a separately named cohort"
        )
    dataset, manifest = _load_frozen_dataset(cache_dir, context)
    if date.fromisoformat(context["training_cutoff"]) >= now.date():
        raise RuntimeError("frozen cutoff is not earlier than trusted system UTC")
    return freeze_shadow_cohort(
        dataset,
        manifest,
        context,
        cohort_id=cohort_id,
        root=root,
        test_mode=False,
    )


def _normalized_history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    frame = history.copy().sort_index()
    index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    valid = ~index.isna()
    frame = frame.loc[valid].copy()
    frame.index = index[valid].tz_convert(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    for name in required | {"RawOpen", "RawClose"}:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _history_through(history: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    frame = _normalized_history(history)
    if frame.empty:
        return frame
    return frame.loc[frame.index.date <= cutoff].copy()


def _bar_checksum(frame: pd.DataFrame, timestamp: pd.Timestamp) -> str:
    row = frame.loc[timestamp]
    fields = {}
    for name in (
        "Open",
        "High",
        "Low",
        "Close",
        "RawOpen",
        "RawClose",
        "Volume",
        "Dividends",
        "Stock Splits",
    ):
        if name in row:
            value = float(row[name])
            fields[name] = value if math.isfinite(value) else None
    return canonical_hash({"timestamp": timestamp.isoformat(), "bar": fields})


def _record_signature(value: dict[str, Any], key: bytes) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("record_hash", None)
    result.pop("record_signature", None)
    result["record_hash"] = canonical_hash(result)
    result["record_signature"] = signed_digest(result["record_hash"], key)
    return result


def _regime_for_vector(model: dict[str, Any], vector: dict[str, float]) -> str:
    definition = model["forward_regime_definition"]
    cutoffs = np.asarray(definition["volatility_terciles"], dtype=float)
    trend = "trend_up" if vector["spy_price_sma200"] >= 0 else "trend_down"
    volatility = vector["spy_vol_60"]
    bucket = (
        "vol_low"
        if volatility <= cutoffs[0]
        else "vol_mid"
        if volatility <= cutoffs[1]
        else "vol_high"
    )
    return f"{trend}|{bucket}"


def _scheduled_maturity(feature_date: date, horizon: int) -> dict[str, Any]:
    return {
        "sessions_after_feature": int(horizon),
        "scheduled_exit_session": _project_us_session(
            feature_date,
            int(horizon),
        ).isoformat(),
        "schedule_version": "nyse-weekday-holidays-v1",
    }


def _expected_completed_us_session(
    histories: dict[str, pd.DataFrame],
    *,
    us_session_symbols: Iterable[str],
    now: datetime,
) -> date:
    candidates: list[date] = []
    current_iso = now.date().isocalendar()
    for symbol in sorted(set(us_session_symbols) | {"SPY"}):
        frame = _history_through(histories.get(symbol, pd.DataFrame()), now.date())
        if frame.empty:
            continue
        latest = pd.Timestamp(frame.index[-1]).date()
        latest_iso = latest.isocalendar()
        if (latest_iso.year, latest_iso.week) == (
            current_iso.year,
            current_iso.week,
        ):
            candidates.append(latest)
    if not candidates:
        raise RuntimeError(
            "no retrospective backfill: cannot establish expected completed US "
            "session from the current ISO week"
        )
    friday = now.date() + timedelta(days=4 - now.date().weekday())
    if now.date().weekday() > 4:
        friday = now.date() - timedelta(days=now.date().weekday() - 4)
    scheduled = _latest_projected_session_on_or_before(friday)
    observed = max(candidates)
    if observed != scheduled:
        raise RuntimeError(
            "current US histories are stale or disagree with the frozen NYSE "
            f"session schedule: observed {observed}, expected {scheduled}"
        )
    return scheduled


def _ensure_capture_window(
    *,
    now: datetime,
    anchor_date: date,
    frozen_at: datetime,
) -> None:
    if anchor_date <= frozen_at.date():
        raise RuntimeError("no retrospective backfill: anchor is not after the freeze date")
    anchor_iso = anchor_date.isocalendar()
    now_iso = now.date().isocalendar()
    if (anchor_iso.year, anchor_iso.week) != (now_iso.year, now_iso.week):
        raise RuntimeError(
            "no retrospective backfill: capture must occur in the anchor ISO week"
        )
    if now.weekday() < 4 or (now.weekday() == 4 and now.time() < SAFE_CAPTURE_UTC):
        raise RuntimeError(
            "weekly capture is before the configured Friday 23:00 UTC safe cutoff"
        )
    if now.weekday() > 6:
        raise AssertionError("invalid weekday")


def _load_cohort(
    *,
    cohort_id: str,
    root: Path,
    test_mode: bool = False,
) -> tuple[CohortPaths, dict[str, Any], dict[str, Any], bytes]:
    paths = cohort_paths(cohort_id, root)
    preregistration = validate_preregistration(
        load_json(paths.preregistration, required=True, expected_type=dict)
    )
    artifact = load_shadow_artifact(paths.artifact)
    key = paths.signing_key.read_bytes()
    development = preregistration["development_test_mode"]
    if (
        artifact["cohort_id"] != cohort_id
        or artifact["preregistration_hash"]
        != preregistration["preregistration_hash"]
        or artifact["signing_key_id"] != signing_key_id(key)
        or artifact["development_test_mode"] is not development
    ):
        raise RuntimeError("cohort preregistration/artifact/signing binding mismatch")
    if development is not bool(test_mode):
        raise RuntimeError(
            "development test-mode cohort cannot be used by production commands"
        )
    trusted = _trusted_now(test_mode=test_mode)
    frozen = _parse_utc(preregistration["frozen_at"])
    if frozen > trusted:
        raise RuntimeError("cohort freeze is later than trusted UTC")
    if not test_mode and frozen < IMPLEMENTATION_CREATED_AT_UTC:
        raise RuntimeError("cohort freeze predates implementation creation")
    if date.fromisoformat(preregistration["training_data_cutoff"]) >= frozen.date():
        raise RuntimeError("cohort cutoff is not strictly earlier than its freeze")
    return paths, preregistration, artifact, key


def capture_from_histories(
    histories: dict[str, pd.DataFrame],
    eligibility: EligibilityResult,
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    bar_info: dict[str, dict[str, Any]] | None = None,
    provider_failures: dict[str, str] | None = None,
    us_session_symbols: Iterable[str] | None = None,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    captured_at = _trusted_now(clock=clock, test_mode=test_mode)
    spy = _normalized_history(histories.get("SPY", pd.DataFrame()))
    if spy.empty:
        raise RuntimeError("capture requires completed SPY history")
    expected_session = _expected_completed_us_session(
        histories,
        us_session_symbols=us_session_symbols or eligibility.symbols,
        now=captured_at,
    )
    spy = _history_through(spy, captured_at.date())
    selected_spy_session = pd.Timestamp(spy.index[-1]).date()
    if selected_spy_session != expected_session:
        raise RuntimeError(
            "SPY is stale: selected session does not equal expected latest "
            "completed US session"
        )
    anchor_date = expected_session
    frozen_at = _parse_utc(preregistration["frozen_at"])
    _ensure_capture_window(
        now=captured_at,
        anchor_date=anchor_date,
        frozen_at=frozen_at,
    )
    iso = anchor_date.isocalendar()
    prediction_path = paths.predictions / f"{anchor_date.isoformat()}.json.gz"

    existing_envelope = None
    if prediction_path.exists():
        existing_envelope = validate_capture_envelope(
            read_gzip_json(prediction_path),
            key,
        )
        captured_at_text = existing_envelope["core"]["captured_at"]
    else:
        captured_at_text = captured_at.isoformat(timespec="seconds")

    info = bar_info or {}
    exclusions = []
    for symbol, reason in sorted(eligibility.excluded.items()):
        exclusion = {
            "exclusion_id": canonical_hash(
                {
                    "cohort_id": cohort_id,
                    "anchor_date": anchor_date.isoformat(),
                    "symbol": symbol,
                    "reason": reason,
                }
            ),
            "symbol": symbol,
            "issuer_key": None,
            "reason": reason,
        }
        exclusions.append(_record_signature(exclusion, key))

    predictions = []
    spy_through_anchor = _history_through(spy, anchor_date)
    provider_failure_map = {
        symbol: str(reason)[:240]
        for symbol, reason in sorted((provider_failures or {}).items())
        if symbol in eligibility.symbols
    }
    successful_provider_symbols = []
    for symbol in eligibility.symbols:
        provider_frame = _history_through(
            histories.get(symbol, pd.DataFrame()),
            anchor_date,
        )
        if not provider_frame.empty and len(provider_frame) >= 30:
            successful_provider_symbols.append(symbol)
            provider_failure_map.pop(symbol, None)
        else:
            provider_failure_map.setdefault(
                symbol,
                "no usable completed provider history",
            )
    requested_issuer_count = len(eligibility.symbols)
    successful_issuer_count = len(successful_provider_symbols)
    provider_coverage = (
        successful_issuer_count / requested_issuer_count
        if requested_issuer_count
        else 0.0
    )
    for symbol in eligibility.symbols:
        issuer_key = eligibility.issuer_keys[symbol]
        history = _history_through(histories.get(symbol, pd.DataFrame()), anchor_date)
        if history.empty or len(history) < MIN_HISTORY_BARS + 1:
            exclusion = {
                "exclusion_id": canonical_hash(
                    {
                        "cohort_id": cohort_id,
                        "anchor_date": anchor_date.isoformat(),
                        "symbol": symbol,
                        "reason": "insufficient completed history for frozen features",
                    }
                ),
                "symbol": symbol,
                "issuer_key": issuer_key,
                "reason": "insufficient completed history for frozen features",
            }
            exclusions.append(_record_signature(exclusion, key))
            continue
        feature_date = pd.Timestamp(history.index[-1])
        if feature_date.date() != expected_session:
            exclusion = {
                "exclusion_id": canonical_hash(
                    {
                        "cohort_id": cohort_id,
                        "anchor_date": anchor_date.isoformat(),
                        "symbol": symbol,
                        "reason": (
                            "latest stock session does not equal expected completed "
                            "US cohort session"
                        ),
                    }
                ),
                "symbol": symbol,
                "issuer_key": issuer_key,
                "reason": (
                    "latest stock session does not equal expected completed "
                    "US cohort session"
                ),
            }
            exclusions.append(_record_signature(exclusion, key))
            continue
        spy_asof_rows = spy_through_anchor.loc[
            spy_through_anchor.index <= feature_date
        ]
        if spy_asof_rows.empty:
            raise RuntimeError(f"strict SPY as-of history is unavailable for {symbol}")
        spy_asof = pd.Timestamp(spy_asof_rows.index[-1])
        if spy_asof.date() != expected_session or spy_asof != feature_date:
            raise RuntimeError(
                "stock feature session and SPY as-of must both equal the expected "
                "completed US cohort session"
            )
        try:
            feature_timestamp, vector = latest_probability_features(
                history,
                spy_asof_rows,
                as_of=feature_date,
            )
        except Exception as exc:
            exclusion = {
                "exclusion_id": canonical_hash(
                    {
                        "cohort_id": cohort_id,
                        "anchor_date": anchor_date.isoformat(),
                        "symbol": symbol,
                        "reason": f"feature gate failed: {str(exc)[:180]}",
                    }
                ),
                "symbol": symbol,
                "issuer_key": issuer_key,
                "reason": f"feature gate failed: {str(exc)[:180]}",
            }
            exclusions.append(_record_signature(exclusion, key))
            continue
        if pd.Timestamp(feature_timestamp) != feature_date:
            raise AssertionError("feature extraction did not use the final stock session")
        feature_hash = canonical_hash(
            {
                "feature_names": list(FEATURE_NAMES),
                "values": [vector[name] for name in FEATURE_NAMES],
            }
        )
        source_checksum = _bar_checksum(history, feature_date)
        completed_info = info.get(symbol) or {}
        if completed_info and completed_info.get("completed_bars_only") is not True:
            raise RuntimeError(f"provider did not certify completed bars for {symbol}")
        for horizon in HORIZONS:
            model = artifact["models"][ordered_model_key(horizon)]
            matrix = pd.DataFrame([vector], columns=FEATURE_NAMES)
            ood = assess_ood(matrix, model["preprocessor"])[0]
            ordered = predict_ordered_probabilities(
                model,
                matrix,
                require_complete=True,
            )[0]
            derived = derive_threshold_probabilities(
                ordered,
                THRESHOLD_GRIDS[horizon],
            )
            assert_exact_ordered_monotonicity(derived)
            reason = "; ".join(ood["reasons"]) if ood["withhold"] else None
            record = {
                "prediction_id": canonical_hash(
                    {
                        "cohort_id": cohort_id,
                        "artifact_hash": artifact["artifact_hash"],
                        "anchor_date": anchor_date.isoformat(),
                        "symbol": symbol,
                        "issuer_key": issuer_key,
                        "horizon": horizon,
                    }
                ),
                "classification": SHADOW_CLASSIFICATION,
                "shadow_only": True,
                "actionable": False,
                "cohort_id": cohort_id,
                "artifact_hash": artifact["artifact_hash"],
                "model_key": ordered_model_key(horizon),
                "horizon_sessions": horizon,
                "symbol": symbol,
                "issuer_key": issuer_key,
                "asset_type": "company_equity",
                "currency": "USD",
                "partition": PARTITION,
                "feature_date": feature_date.date().isoformat(),
                "feature_timestamp": feature_timestamp.isoformat(),
                "expected_us_session": expected_session.isoformat(),
                "spy_asof": spy_asof.date().isoformat(),
                "feature_hash": feature_hash,
                "source_bar_checksum": source_checksum,
                "raw_ordered_probabilities": [float(value) for value in ordered],
                "derived_probabilities": {
                    str(threshold): {
                        name: float(values[index])
                        for index, name in enumerate(CLASS_NAMES)
                    }
                    for threshold, values in derived.items()
                },
                "baseline_rates": model["baseline_rates_by_threshold"],
                "ood": ood,
                "regime": _regime_for_vector(model, vector),
                "eligible_for_evaluation": not ood["withhold"],
                "exclusion_reason": reason,
                "entry_status": "pending_first_open_after_feature_session",
                "maturity_target": _scheduled_maturity(
                    feature_date.date(),
                    horizon,
                ),
            }
            predictions.append(_record_signature(record, key))

    predictions.sort(
        key=lambda item: (
            item["symbol"],
            item["issuer_key"],
            item["horizon_sessions"],
        )
    )
    exclusions = list(
        {
            (item["symbol"], item["reason"]): item
            for item in exclusions
        }.values()
    )
    exclusions.sort(key=lambda item: (item["symbol"], item["reason"]))
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.verify_sealed_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        existing_anchor = ledger.anchor_for_week(iso.year, iso.week)
        previous_chain = (
            existing_anchor["previous_chain_hash"]
            if existing_anchor is not None
            else (
                ledger.latest_anchor()["chain_hash"]
                if ledger.latest_anchor() is not None
                else GENESIS_CHAIN_HASH
            )
        )
        core = {
            "cohort_id": cohort_id,
            "artifact_hash": artifact["artifact_hash"],
            "classification": SHADOW_CLASSIFICATION,
            "shadow_only": True,
            "actionable": False,
            "anchor_date": anchor_date.isoformat(),
            "iso_year": iso.year,
            "iso_week": iso.week,
            "captured_at": captured_at_text,
            "expected_us_session": expected_session.isoformat(),
            "spy_asof": anchor_date.isoformat(),
            "provider": {
                "requested_issuer_count": requested_issuer_count,
                "successful_issuer_count": successful_issuer_count,
                "success_coverage": provider_coverage,
                "failures": provider_failure_map,
            },
            "previous_chain_hash": previous_chain,
            "predictions": predictions,
            "exclusions": exclusions,
        }
        envelope = make_capture_envelope(core, key)
        if existing_envelope is not None and envelope != existing_envelope:
            raise RuntimeError(
                "immutable weekly file differs from current input; refusing rewrite"
            )
        encoded = deterministic_gzip_json(envelope)
        created = write_immutable_bytes(prediction_path, encoded)
        file_digest = sha256_bytes(encoded)
        try:
            inserted = ledger.insert_capture(envelope, file_digest=file_digest)
        except Exception:
            if created:
                prediction_path.unlink(missing_ok=True)
            raise
        ledger.verify_integrity(key)
        ledger.export_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        ledger.backup(
            paths.backups,
            manifest_path=paths.manifest,
            candidate_report_path=paths.candidate_report,
        )
        counts = ledger.aggregate_counts()
    return {
        "anchor_date": anchor_date.isoformat(),
        "prediction_count": len(predictions),
        "eligible_count": sum(
            int(bool(item["eligible_for_evaluation"])) for item in predictions
        ),
        "withheld_count": sum(
            int(not item["eligible_for_evaluation"]) for item in predictions
        ),
        "exclusion_count": len(exclusions),
        "created": bool(created and inserted),
        "idempotent": not bool(created and inserted),
        "aggregate_counts": counts,
    }


def _load_universe() -> tuple[EligibilityResult, list[dict[str, Any]]]:
    with (DATA / "tickers.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        universe = list(csv.DictReader(handle))
    metadata = load_json(
        DATA / "fundamentals.json", required=True, expected_type=dict
    )
    return select_eligible_universe(universe, metadata), universe


def _download(
    symbols: Iterable[str],
    *,
    period: str,
    now: datetime,
) -> PriceFetchResult:
    return fetch_prices_with_status(
        sorted(set(symbols)),
        period=period,
        now=now,
        verbose=False,
    )


def capture_latest(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    period: str = CAPTURE_HISTORY_PERIOD,
    publish: bool = True,
) -> dict[str, Any]:
    capture_time = _trusted_now()
    eligibility, universe = _load_universe()
    us_session_symbols = {
        str(item.get("symbol") or "").strip().upper()
        for item in universe
        if str(item.get("exchange") or "").strip().upper()
        in {"NYSE", "NASDAQ", "AMEX", "NYSE ARCA"}
    } & set(eligibility.symbols)
    fetched = _download(
        [*eligibility.symbols, "SPY"],
        period=period,
        now=capture_time,
    )
    result = capture_from_histories(
        fetched.prices,
        eligibility,
        cohort_id=cohort_id,
        root=root,
        bar_info=fetched.bar_info,
        provider_failures=fetched.failed_symbols,
        us_session_symbols=us_session_symbols,
    )
    if publish:
        publish_local_status(
            cohort_id=cohort_id,
            root=root,
        )
    return result


def _session_position(frame: pd.DataFrame, value: str) -> int | None:
    matches = np.flatnonzero(frame.index == pd.Timestamp(value))
    return int(matches[0]) if len(matches) == 1 else None


def _label_value(
    prediction: Any,
    frame: pd.DataFrame,
    *,
    labeled_at: datetime,
    key: bytes,
) -> dict[str, Any]:
    horizon = int(prediction["horizon_sessions"])
    feature_position = _session_position(frame, prediction["feature_date"])
    if feature_position is None:
        raise ValueError("feature_session_missing")
    entry_position = feature_position + 1
    exit_position = feature_position + horizon
    if exit_position >= len(frame):
        raise ValueError("not_matured")
    entry = frame.iloc[entry_position]
    exit_row = frame.iloc[exit_position]
    entry_adjusted = float(entry["Open"])
    exit_adjusted = float(exit_row["Close"])
    entry_raw = float(
        entry["RawOpen"] if "RawOpen" in frame.columns else entry_adjusted
    )
    exit_raw = float(
        exit_row["RawClose"] if "RawClose" in frame.columns else exit_adjusted
    )
    prices = (entry_adjusted, exit_adjusted, entry_raw, exit_raw)
    if any(not math.isfinite(value) or value <= 0 for value in prices):
        raise ValueError("nonfinite_or_nonpositive_maturity_price")
    gross = exit_adjusted / entry_adjusted - 1.0
    ordered = int(
        classify_ordered_move(
            np.asarray([gross]),
            (
                threshold / 100.0
                for threshold in THRESHOLD_GRIDS[horizon]
            ),
        )[0]
    )
    threshold_labels = {
        str(threshold): int(
            classify_material_move(
                np.asarray([gross]),
                threshold / 100.0,
            )[0]
        )
        for threshold in THRESHOLD_GRIDS[horizon]
    }
    entry_timestamp = pd.Timestamp(frame.index[entry_position])
    exit_timestamp = pd.Timestamp(frame.index[exit_position])
    checksum_rows = {
        "feature": _bar_checksum(frame, pd.Timestamp(frame.index[feature_position])),
        "entry": _bar_checksum(frame, entry_timestamp),
        "exit": _bar_checksum(frame, exit_timestamp),
    }
    core = {
        "prediction_id": prediction["prediction_id"],
        "entry_timestamp": entry_timestamp.isoformat(),
        "exit_timestamp": exit_timestamp.isoformat(),
        "entry_open_adjusted": entry_adjusted,
        "entry_open_raw": entry_raw,
        "exit_close_adjusted": exit_adjusted,
        "exit_close_raw": exit_raw,
        "gross_return": float(gross),
        "long_net_return": float(gross - ROUND_TRIP_COST),
        "material_net_return": float(cost_adjusted_material_return(gross)),
        "ordered_label": ordered,
        "threshold_labels": threshold_labels,
        "source_checksum": canonical_hash(checksum_rows),
        "convention": (
            "first adjusted/raw-equivalent open at t+1; adjusted close at t+H; "
            "Yahoo adjustment factor; round-trip cost 30 bps"
        ),
        "labeled_at": labeled_at.isoformat(timespec="seconds"),
    }
    core["label_hash"] = canonical_hash(core)
    core["label_signature"] = signed_digest(core["label_hash"], key)
    return core


def evaluate_from_histories(
    histories: dict[str, pd.DataFrame],
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
    recompute_report: bool = True,
) -> dict[str, Any]:
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    evaluated_at = _trusted_now(clock=clock, test_mode=test_mode)
    normalized = {
        symbol: _normalized_history(frame) for symbol, frame in histories.items()
    }
    spy = normalized.get("SPY", pd.DataFrame())
    labeled = 0
    unresolved = 0
    pending = 0
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.verify_sealed_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        for prediction in ledger.pending_predictions():
            symbol = prediction["symbol"]
            frame = normalized.get(symbol, pd.DataFrame())
            observed_through = (
                frame.index[-1].date().isoformat() if not frame.empty else None
            )
            feature_position = (
                _session_position(frame, prediction["feature_date"])
                if not frame.empty
                else None
            )
            horizon = int(prediction["horizon_sessions"])
            symbol_mature = bool(
                feature_position is not None
                and feature_position + horizon < len(frame)
            )
            spy_position = (
                _session_position(spy, prediction["feature_date"])
                if not spy.empty
                else None
            )
            market_mature = bool(
                spy_position is not None and spy_position + horizon < len(spy)
            )
            if not symbol_mature:
                if market_mature:
                    reason = (
                        "feature_session_missing"
                        if feature_position is None
                        else "missing_or_halted_symbol_sessions_after_market_maturity"
                    )
                    if ledger.record_resolution_attempt(
                        prediction["prediction_id"],
                        attempted_at=evaluated_at.isoformat(timespec="seconds"),
                        observed_through=observed_through,
                        reason=reason,
                    ):
                        unresolved += 1
                else:
                    pending += 1
                continue
            try:
                label = _label_value(
                    prediction,
                    frame,
                    labeled_at=evaluated_at,
                    key=key,
                )
                if ledger.insert_label(label):
                    labeled += 1
            except ValueError as exc:
                reason = str(exc)
                if reason == "not_matured":
                    pending += 1
                elif ledger.record_resolution_attempt(
                    prediction["prediction_id"],
                    attempted_at=evaluated_at.isoformat(timespec="seconds"),
                    observed_through=observed_through,
                    reason=reason,
                ):
                    unresolved += 1
        integrity = ledger.verify_integrity(key)
        ledger.export_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        ledger.backup(
            paths.backups,
            manifest_path=paths.manifest,
            candidate_report_path=paths.candidate_report,
        )
        counts = ledger.aggregate_counts()
    report = (
        build_candidate_report(
            cohort_id=cohort_id,
            root=root,
            bootstrap_repetitions=(0 if test_mode else 1000),
            clock=(clock if test_mode else None),
            test_mode=test_mode,
        )
        if recompute_report
        else {"status": "evaluating"}
    )
    return {
        "newly_labeled": labeled,
        "newly_unresolved": unresolved,
        "still_pending_checked": pending,
        "integrity": integrity,
        "aggregate_counts": counts,
        "candidate_status": report["status"],
    }


def evaluate_latest(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    period: str = EVALUATION_HISTORY_PERIOD,
    publish: bool = True,
) -> dict[str, Any]:
    evaluation_time = _trusted_now()
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
    )
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.verify_sealed_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        symbols = {row["symbol"] for row in ledger.pending_predictions()}
    fetched = _download(
        [*symbols, "SPY"],
        period=period,
        now=evaluation_time,
    )
    result = evaluate_from_histories(
        fetched.prices,
        cohort_id=cohort_id,
        root=root,
        recompute_report=not publish,
    )
    if publish:
        status = publish_local_status(
            cohort_id=cohort_id,
            root=root,
        )
        result["candidate_status"] = status["status"]
    return result


def _prospective_support(rows: list[Any], labels: np.ndarray) -> dict[str, Any]:
    dates = pd.to_datetime([row["feature_date"] for row in rows], utc=True)
    quarters = set(zip(dates.year.astype(int), dates.quarter.astype(int)))
    class_counts = {
        name: int((labels == index).sum())
        for index, name in enumerate(CLASS_NAMES)
    }
    values = {
        "distinct_forecast_dates": int(dates.normalize().nunique()),
        "quarter_blocks": len(quarters),
        "issuers": len({row["issuer_key"] for row in rows}),
        "class_counts": class_counts,
    }
    reasons = []
    if values["distinct_forecast_dates"] < PROSPECTIVE_MINIMUMS[
        "distinct_forecast_dates"
    ]:
        reasons.append("prospective distinct forecast dates below 26")
    if values["quarter_blocks"] < PROSPECTIVE_MINIMUMS["quarter_blocks"]:
        reasons.append("prospective quarter blocks below 8")
    if values["issuers"] < PROSPECTIVE_MINIMUMS["issuers"]:
        reasons.append("prospective issuers below 200")
    if min(class_counts.values(), default=0) < PROSPECTIVE_MINIMUMS[
        "outcomes_per_class"
    ]:
        reasons.append("prospective outcomes per class below 200")
    return {**values, "passed": not reasons, "reasons": reasons}


def _point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    baselines: np.ndarray,
) -> dict[str, Any]:
    reliability = adaptive_classwise_reliability(labels, probabilities)
    brier = multiclass_brier(labels, probabilities)
    baseline_brier = multiclass_brier(labels, baselines)
    log_loss = multiclass_log_loss(labels, probabilities)
    baseline_log_loss = multiclass_log_loss(labels, baselines)
    return {
        "count": int(len(labels)),
        "class_counts": {
            name: int((labels == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "brier": brier,
        "climatology_brier": baseline_brier,
        "brier_skill": (
            1.0 - brier / baseline_brier if baseline_brier > 0 else None
        ),
        "log_loss": log_loss,
        "climatology_log_loss": baseline_log_loss,
        "log_loss_improvement": (
            1.0 - log_loss / baseline_log_loss
            if baseline_log_loss > 0
            else None
        ),
        "classwise_ece": {
            name: reliability["classes"][name]["ece"] for name in CLASS_NAMES
        },
        "maximum_gap": reliability["maximum_gap"],
        "calibration": {
            name: _binary_calibration_fit(
                (labels == index).astype(float),
                probabilities[:, index],
            )
            for index, name in enumerate(CLASS_NAMES)
        },
        "adaptive_reliability_supported": bool(
            reliability["all_classes_supported"]
        ),
        "adaptive_supported_bins": {
            name: reliability["classes"][name]["supported_bin_count"]
            for name in CLASS_NAMES
        },
    }


def _threshold_gate_reasons(
    *,
    metrics: dict[str, Any],
    support: dict[str, Any],
    bootstrap: dict[str, Any] | None,
    regime: list[dict[str, Any]],
    coverage: float,
    weeks_captured: int,
    provider_support_value: dict[str, Any],
) -> list[str]:
    gates = STRICT_ACCEPTANCE_GATES
    reasons = list(support["reasons"])
    if weeks_captured < PROSPECTIVE_MINIMUMS["weekly_anchors"]:
        reasons.append("prospective weekly anchors below 104")
    if coverage < gates["inference_coverage_min"]:
        reasons.append(
            f"prospective non-OOD coverage {coverage:.6f} below "
            f"{gates['inference_coverage_min']}"
        )
    if (
        provider_support_value["minimum_anchor_success_coverage"]
        < gates["provider_success_coverage_min"]
    ):
        reasons.append("prospective provider success coverage gate failed")
    if (
        provider_support_value["minimum_anchor_successful_issuers"]
        < gates["provider_successful_issuer_count_min"]
    ):
        reasons.append("prospective provider successful-issuer count gate failed")
    if metrics.get("brier_skill") is None or metrics["brier_skill"] < gates[
        "brier_skill_min"
    ]:
        reasons.append("prospective Brier skill gate failed")
    if (
        metrics.get("log_loss_improvement") is None
        or metrics["log_loss_improvement"] < gates["log_loss_improvement_min"]
    ):
        reasons.append("prospective log-loss improvement gate failed")
    for name in CLASS_NAMES:
        ece = (metrics.get("classwise_ece") or {}).get(name)
        calibration = (metrics.get("calibration") or {}).get(name) or {}
        slope = calibration.get("slope")
        intercept = calibration.get("intercept")
        if ece is None or ece > gates["classwise_ece_max"]:
            reasons.append(f"prospective {name} ECE gate failed")
        if (
            slope is None
            or not gates["calibration_slope_min"]
            <= slope
            <= gates["calibration_slope_max"]
        ):
            reasons.append(f"prospective {name} calibration slope gate failed")
        if (
            intercept is None
            or not gates["calibration_intercept_min"]
            <= intercept
            <= gates["calibration_intercept_max"]
        ):
            reasons.append(f"prospective {name} calibration intercept gate failed")
    if (
        metrics.get("maximum_gap") is None
        or metrics["maximum_gap"] > gates["maximum_gap_max"]
    ):
        reasons.append("prospective maximum reliability gap gate failed")
    if not metrics.get("adaptive_reliability_supported"):
        reasons.append(
            "adaptive reliability has fewer than "
            f"{ADAPTIVE_MIN_SUPPORTED_BINS} supported bins for a class"
        )
    if bootstrap is None:
        reasons.append("prospective 1000-repetition fixed-prediction bootstrap not run")
    else:
        repetitions = int(bootstrap.get("completed_repetitions") or 0)
        ci = bootstrap.get("brier_skill_ci95") or [None, None]
        if (
            repetitions
            < PROSPECTIVE_MINIMUMS["fixed_prediction_bootstrap_repetitions"]
            or not isinstance(ci, list)
            or len(ci) != 2
            or not isinstance(ci[0], (int, float))
            or ci[0] <= gates["brier_skill_ci_low_strict_min"]
        ):
            reasons.append("prospective Brier-skill bootstrap gate failed")
    supported_regimes = [row for row in regime if row["support"]["available"]]
    if not supported_regimes:
        reasons.append("no prospectively supported market regime")
    for row in supported_regimes:
        if (
            row["max_class_ece"] is None
            or row["max_class_ece"] > gates["regime_ece_max"]
        ):
            reasons.append(f"prospective regime {row['regime']} ECE gate failed")
    return list(dict.fromkeys(reasons))


def build_candidate_report(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    bootstrap_repetitions: int = 1000,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
    _retry_count: int = 0,
) -> dict[str, Any]:
    if (
        int(bootstrap_repetitions)
        != PROSPECTIVE_MINIMUMS["fixed_prediction_bootstrap_repetitions"]
        and not test_mode
    ):
        raise RuntimeError(
            "production forward reports require exactly 1000 bootstrap repetitions"
        )
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    generated = _trusted_now(clock=clock, test_mode=test_mode)
    horizons: dict[str, Any] = {}
    all_passed = True
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.verify_sealed_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        ledger.connection.execute("BEGIN")
        ledger.connection.execute("SELECT COUNT(*) FROM events").fetchone()
        report_snapshot = ledger.snapshot_identity()
        counts = ledger.aggregate_counts()
        provider_support_value = ledger.provider_support()
        for horizon in HORIZONS:
            rows = ledger.labels_for_metrics(horizon)
            threshold_reports: dict[str, Any] = {}
            horizon_passed = bool(rows)
            if rows:
                raw_ordered = np.asarray(
                    [json.loads(row["raw_ordered_json"]) for row in rows],
                    dtype=float,
                )
                derived_all = derive_threshold_probabilities(
                    raw_ordered,
                    THRESHOLD_GRIDS[horizon],
                )
                exact = assert_exact_ordered_monotonicity(derived_all)
                eligible_count = counts["eligible_prediction_counts"][str(horizon)]
                requested_count = provider_support_value[
                    "requested_issuer_observations"
                ]
                coverage = (
                    eligible_count / requested_count if requested_count else 0.0
                )
                for threshold in THRESHOLD_GRIDS[horizon]:
                    labels = np.asarray(
                        [
                            json.loads(row["threshold_labels_json"])[str(threshold)]
                            for row in rows
                        ],
                        dtype=int,
                    )
                    probabilities = np.asarray(
                        [
                            [
                                json.loads(row["derived_json"])[str(threshold)][name]
                                for name in CLASS_NAMES
                            ]
                            for row in rows
                        ],
                        dtype=float,
                    )
                    baselines = np.asarray(
                        [
                            [
                                json.loads(row["baseline_json"])[str(threshold)][name]
                                for name in CLASS_NAMES
                            ]
                            for row in rows
                        ],
                        dtype=float,
                    )
                    support = _prospective_support(rows, labels)
                    metrics = _point_metrics(labels, probabilities, baselines)
                    bootstrap = (
                        two_way_cluster_bootstrap(
                            labels,
                            probabilities,
                            baselines,
                            [row["feature_date"] for row in rows],
                            [row["issuer_key"] for row in rows],
                            repetitions=int(bootstrap_repetitions),
                            seed=DEFAULT_SEED + horizon + threshold,
                        )
                        if bootstrap_repetitions
                        >= PROSPECTIVE_MINIMUMS[
                            "fixed_prediction_bootstrap_repetitions"
                        ]
                        and support["passed"]
                        else None
                    )
                    regimes = []
                    regime_values = np.asarray(
                        [row["regime"] for row in rows], dtype=object
                    )
                    for regime_name in sorted(set(regime_values)):
                        mask = regime_values == regime_name
                        regime_labels = labels[mask]
                        regime_probabilities = probabilities[mask]
                        regime_rows = [
                            row for index, row in enumerate(rows) if mask[index]
                        ]
                        support_value = regime_support(
                            regime_labels,
                            [row["feature_date"] for row in regime_rows],
                            [row["issuer_key"] for row in regime_rows],
                        )
                        reliability = (
                            adaptive_classwise_reliability(
                                regime_labels,
                                regime_probabilities,
                            )
                            if support_value["available"]
                            else None
                        )
                        adaptive_supported = bool(
                            reliability and reliability["all_classes_supported"]
                        )
                        max_ece = (
                            max(
                                reliability["classes"][name]["ece"]
                                for name in CLASS_NAMES
                            )
                            if adaptive_supported
                            else None
                        )
                        regimes.append(
                            {
                                "regime": str(regime_name),
                                "count": int(mask.sum()),
                                "support": support_value,
                                "adaptive_reliability_supported": adaptive_supported,
                                "max_class_ece": max_ece,
                            }
                        )
                    reasons = _threshold_gate_reasons(
                        metrics=metrics,
                        support=support,
                        bootstrap=bootstrap,
                        regime=regimes,
                        coverage=coverage,
                        weeks_captured=counts["weeks_captured"],
                        provider_support_value=provider_support_value,
                    )
                    threshold_reports[str(threshold)] = {
                        "threshold_pct": threshold,
                        "support": support,
                        "coverage": coverage,
                        "metrics": metrics,
                        "bootstrap": bootstrap,
                        "regimes": regimes,
                        "passed": not reasons,
                        "reasons": reasons,
                    }
                    horizon_passed = horizon_passed and not reasons
            else:
                exact = {
                    "passed": False,
                    "reason": "no prospectively matured outcomes",
                }
                horizon_passed = False
            horizons[str(horizon)] = {
                "horizon_sessions": horizon,
                "matured_outcome_count": len(rows),
                "exact_monotonicity": exact,
                "thresholds": threshold_reports,
                "passed": horizon_passed,
            }
            all_passed = all_passed and horizon_passed
    if test_mode:
        all_passed = False
    report = _finalize_hash(
        {
            "schema": "stock-radar-probability-forward-candidate-report",
            "schema_version": 1,
            "classification": SHADOW_CLASSIFICATION,
            "shadow_only": True,
            "actionable": False,
            "automatic_promotion": False,
            "independent_review_required": True,
            "development_test_mode": bool(test_mode),
            "cohort_id": cohort_id,
            "artifact_hash": artifact["artifact_hash"],
            "preregistration_hash": preregistration["preregistration_hash"],
            "generated_at": generated.isoformat(timespec="seconds"),
            "source_snapshot": report_snapshot,
            "retrospective_metrics_role": "context_only",
            "minimums": PROSPECTIVE_MINIMUMS,
            "metric_gates": STRICT_ACCEPTANCE_GATES,
            "provider_support": provider_support_value,
            "status": "eligible_for_review" if all_passed else "evaluating",
            "all_gates_passed": all_passed,
            "horizons": horizons,
        },
        "report_hash",
    )
    try:
        with ForwardLedger(
            paths.ledger,
            cohort_id=cohort_id,
            artifact_hash=artifact["artifact_hash"],
            preregistration_hash=preregistration["preregistration_hash"],
            signing_key=key,
        ) as ledger:
            ledger.verify_sealed_manifest(
                paths.manifest,
                key,
                candidate_report_path=paths.candidate_report,
            )
            ledger.record_candidate_report(
                report,
                expected_snapshot=report_snapshot,
            )
            ledger.export_manifest(
                paths.manifest,
                key,
                candidate_report_path=paths.candidate_report,
            )
            ledger.backup(
                paths.backups,
                manifest_path=paths.manifest,
                candidate_report_path=paths.candidate_report,
            )
    except ForwardStaleSnapshotError:
        if _retry_count >= 2:
            raise
        return build_candidate_report(
            cohort_id=cohort_id,
            root=root,
            bootstrap_repetitions=bootstrap_repetitions,
            clock=clock,
            test_mode=test_mode,
            _retry_count=_retry_count + 1,
        )
    return report


def _public_snapshot_token(
    ledger: ForwardLedger,
    manifest: dict[str, Any],
    counts: dict[str, Any],
    provider_support_value: dict[str, Any],
) -> dict[str, Any]:
    identity = ledger.snapshot_identity()
    seal = ledger.connection.execute(
        "SELECT * FROM seal_state WHERE singleton = 1"
    ).fetchone()
    if seal is None or seal["pending_event_count"] is not None:
        raise ForwardStaleSnapshotError(
            "aggregate snapshot is not at a finalized external seal"
        )
    if (
        int(seal["finalized_event_count"]) != identity["event_count"]
        or seal["finalized_event_head_hash"] != identity["event_head_hash"]
        or seal["finalized_table_snapshot_root"]
        != identity["table_snapshot_root"]
        or seal["finalized_manifest_hash"] != manifest.get("manifest_hash")
        or int(manifest["event_count"]) != identity["event_count"]
        or manifest["event_head_hash"] != identity["event_head_hash"]
        or manifest["table_snapshot_root"] != identity["table_snapshot_root"]
    ):
        raise ForwardStaleSnapshotError(
            "aggregate snapshot, database seal, and external manifest differ"
        )
    return {
        **identity,
        "manifest_hash": manifest["manifest_hash"],
        "finalized_event_count": int(seal["finalized_event_count"]),
        "finalized_event_head_hash": seal["finalized_event_head_hash"],
        "finalized_table_snapshot_root": seal[
            "finalized_table_snapshot_root"
        ],
        "counts_hash": canonical_hash(counts),
        "provider_support_hash": canonical_hash(provider_support_value),
    }


def build_public_status(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
    _return_binding: bool = False,
    _snapshot_hook: Callable[[], None] | None = None,
) -> Any:
    updated = _trusted_now(clock=clock, test_mode=test_mode)
    build_candidate_report(
        cohort_id=cohort_id,
        root=root,
        bootstrap_repetitions=(0 if test_mode else 1000),
        clock=(clock if test_mode else None),
        test_mode=test_mode,
    )
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        manifest = ledger.verify_sealed_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        with ledger.read_snapshot():
            counts = ledger.aggregate_counts()
            provider_support_value = ledger.provider_support()
            if _snapshot_hook is not None:
                _snapshot_hook()
            candidate = ledger.latest_candidate_report()
            latest_event = ledger.connection.execute(
                """
                SELECT sequence, event_type, entity_key, previous_event_hash
                FROM events ORDER BY sequence DESC LIMIT 1
                """
            ).fetchone()
            status_binding = _public_snapshot_token(
                ledger,
                manifest,
                counts,
                provider_support_value,
            )
            source = (candidate or {}).get("source_snapshot") or {}
            candidate_is_current = bool(
                candidate is not None
                and latest_event is not None
                and latest_event["event_type"] == "candidate_report"
                and latest_event["entity_key"] == candidate["report_hash"]
                and int(latest_event["sequence"])
                == int(status_binding["event_count"])
                and int(source.get("event_count") or -1) + 1
                == int(status_binding["event_count"])
                and source.get("event_head_hash")
                == latest_event["previous_event_hash"]
                and source.get("table_snapshot_root")
                == status_binding["table_snapshot_root"]
                and candidate.get("provider_support")
                == provider_support_value
            )
    matured_total = sum(counts["matured_outcomes"].values())
    status = (
        "eligible_for_review"
        if candidate.get("all_gates_passed") is True and candidate_is_current
        else "evaluating"
        if matured_total
        else "collecting"
    )
    value = initial_forward_validation_status()
    value.update(
        {
            "cohort_id": cohort_id,
            "frozen_at": preregistration["frozen_at"],
            "implementation_state": "frozen_shadow_monitoring",
            "training_cutoff": preregistration["training_data_cutoff"],
            "weeks_captured": counts["weeks_captured"],
            "eligible_prediction_counts": counts["eligible_prediction_counts"],
            "matured_outcomes": counts["matured_outcomes"],
            "unresolved_outcomes": counts["unresolved_outcomes"],
            "next_maturity_dates": counts["next_maturity_dates"],
            "status": status,
            "integrity_status": "signed_hash_chain_verified",
            "first_anchor_date": counts["first_anchor_date"],
            "latest_anchor_date": counts["latest_anchor_date"],
            "schedule": preregistration["prospective_schedule"],
            "last_updated_at": updated.isoformat(timespec="seconds"),
        }
    )
    finalized = finalize_forward_validation_status(value)
    return (finalized, status_binding) if _return_binding else finalized


def publish_local_status(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    status_path: Path = PUBLIC_STATUS_PATH,
    snapshot_path: Path = LATEST_PATH,
    clock: Callable[[], datetime] | None = None,
    test_mode: bool = False,
    _retry_count: int = 0,
    _snapshot_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    status, expected_binding = build_public_status(
        cohort_id=cohort_id,
        root=root,
        clock=clock,
        test_mode=test_mode,
        _return_binding=True,
        _snapshot_hook=_snapshot_hook,
    )
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    try:
        with ForwardLedger(
            paths.ledger,
            cohort_id=cohort_id,
            artifact_hash=artifact["artifact_hash"],
            preregistration_hash=preregistration["preregistration_hash"],
            signing_key=key,
        ) as ledger:
            manifest = ledger.verify_sealed_manifest(
                paths.manifest,
                key,
                candidate_report_path=paths.candidate_report,
            )
            with ledger.transaction():
                current_counts = ledger.aggregate_counts()
                current_provider = ledger.provider_support()
                current_binding = _public_snapshot_token(
                    ledger,
                    manifest,
                    current_counts,
                    current_provider,
                )
                if current_binding != expected_binding:
                    raise ForwardStaleSnapshotError(
                        "ledger changed before aggregate status publication"
                    )
                atomic_write_json(status_path, status)
                if Path(snapshot_path).exists():
                    inject_forward_validation_status(snapshot_path, status_path)
    except ForwardStaleSnapshotError:
        if _retry_count >= 2:
            raise
        return publish_local_status(
            cohort_id=cohort_id,
            root=root,
            status_path=status_path,
            snapshot_path=snapshot_path,
            clock=clock,
            test_mode=test_mode,
            _retry_count=_retry_count + 1,
            _snapshot_hook=_snapshot_hook,
        )
    return status


def publish_aggregate_only(
    *,
    status_path: Path = PUBLIC_STATUS_PATH,
    snapshot_path: Path = LATEST_PATH,
) -> dict[str, Any]:
    # Used by GitHub Actions. It republishes only an already-committed aggregate;
    # it cannot create forecasts, labels, or unseen history without the local ledger.
    status = load_forward_validation_status(status_path, allow_default=True)
    inject_forward_validation_status(snapshot_path, status_path)
    return status


def verify_cohort(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    test_mode: bool = False,
) -> dict[str, Any]:
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.verify_sealed_manifest(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        integrity = ledger.verify_integrity(key)
        anchors = list(
            ledger.connection.execute("SELECT * FROM anchors ORDER BY anchor_date")
        )
        for row in anchors:
            path = paths.predictions / f"{row['anchor_date']}.json.gz"
            envelope = validate_capture_envelope(read_gzip_json(path), key)
            if (
                sha256_bytes(path.read_bytes()) != row["file_digest"]
                or envelope["record_digest"] != row["record_digest"]
                or envelope["chain_hash"] != row["chain_hash"]
            ):
                raise RuntimeError(f"weekly file/ledger mismatch for {row['anchor_date']}")
        latest_report_hash = integrity["latest_candidate_report_hash"]
        if latest_report_hash is not None:
            report = load_json(
                paths.candidate_report,
                required=True,
                expected_type=dict,
            )
            _verify_hash(report, "report_hash", "forward candidate report")
            if report["report_hash"] != latest_report_hash:
                raise RuntimeError(
                    "candidate report file does not match the sealed event head"
                )
    return integrity


def backup_cohort(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    test_mode: bool = False,
) -> Path:
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        return ledger.backup(
            paths.backups,
            manifest_path=paths.manifest,
            candidate_report_path=paths.candidate_report,
        )


def recover_cohort(
    *,
    cohort_id: str = DEFAULT_COHORT_ID,
    root: Path = FORWARD_ROOT,
    test_mode: bool = False,
) -> dict[str, Any]:
    paths, preregistration, artifact, key = _load_cohort(
        cohort_id=cohort_id,
        root=root,
        test_mode=test_mode,
    )
    with ForwardLedger(
        paths.ledger,
        cohort_id=cohort_id,
        artifact_hash=artifact["artifact_hash"],
        preregistration_hash=preregistration["preregistration_hash"],
        signing_key=key,
    ) as ledger:
        ledger.recover_seal(
            paths.manifest,
            key,
            candidate_report_path=paths.candidate_report,
        )
        return ledger.verify_integrity(key)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective shadow-only probability validation"
    )
    parser.add_argument("--root", type=Path, default=FORWARD_ROOT)
    parser.add_argument("--cohort-id", default=DEFAULT_COHORT_ID)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--period", default=CAPTURE_HISTORY_PERIOD)
    capture_parser.add_argument("--no-publish", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--period", default=EVALUATION_HISTORY_PERIOD)
    evaluate_parser.add_argument("--no-publish", action="store_true")

    subparsers.add_parser("report")

    publish_parser = subparsers.add_parser("publish-status")
    publish_parser.add_argument("--aggregate-only", action="store_true")
    publish_parser.add_argument("--status-path", type=Path, default=PUBLIC_STATUS_PATH)
    publish_parser.add_argument("--snapshot-path", type=Path, default=LATEST_PATH)

    subparsers.add_parser("verify")
    subparsers.add_parser("recover")
    subparsers.add_parser("backup")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    common = {"cohort_id": args.cohort_id, "root": args.root}
    if args.command == "freeze":
        artifact = freeze_from_cache(
            **common,
            cache_dir=args.cache_dir,
        )
        print(
            "Forward freeze complete: 4 shadow-only rejected horizons; "
            f"cohort={artifact['cohort_id']}; no probabilities published."
        )
    elif args.command == "capture":
        result = capture_latest(
            **common,
            period=args.period,
            publish=not args.no_publish,
        )
        print(
            f"Forward capture {result['anchor_date']}: "
            f"{result['eligible_count']} eligible, "
            f"{result['withheld_count']} withheld, "
            f"{result['exclusion_count']} excluded; "
            f"{'idempotent' if result['idempotent'] else 'stored'}."
        )
    elif args.command == "evaluate":
        result = evaluate_latest(
            **common,
            period=args.period,
            publish=not args.no_publish,
        )
        print(
            "Forward evaluation: "
            f"{result['newly_labeled']} newly matured, "
            f"{result['newly_unresolved']} unresolved, "
            f"{result['still_pending_checked']} pending; "
            f"status={result['candidate_status']}."
        )
    elif args.command == "report":
        status = publish_local_status(
            **common,
        )
        print(
            "Forward candidate report: "
            f"status={status['status']}; automatic promotion disabled."
        )
    elif args.command == "publish-status":
        if args.aggregate_only:
            status = publish_aggregate_only(
                status_path=args.status_path,
                snapshot_path=args.snapshot_path,
            )
        else:
            status = publish_local_status(
                **common,
                status_path=args.status_path,
                snapshot_path=args.snapshot_path,
            )
        print(
            "Forward aggregate published: "
            f"{status['weeks_captured']} weeks; status={status['status']}; "
            "no shadow values included."
        )
    elif args.command == "verify":
        result = verify_cohort(**common)
        print(
            "Forward integrity verified: "
            f"{result['anchors']} anchors, {result['predictions']} predictions, "
            f"{result['labels']} labels."
        )
    elif args.command == "backup":
        path = backup_cohort(**common)
        print(f"Forward SQLite backup created: {path.name}")
    elif args.command == "recover":
        result = recover_cohort(**common)
        print(
            "Forward seal recovered/verified: "
            f"{result['events']} events at {result['event_head_hash'][:12]}."
        )


if __name__ == "__main__":
    main()


__all__ = [
    "CAPTURE_HISTORY_PERIOD",
    "CohortPaths",
    "FORWARD_ROOT",
    "PROSPECTIVE_MINIMUMS",
    "SAFE_CAPTURE_UTC",
    "SHADOW_ARTIFACT_SCHEMA",
    "SHADOW_ARTIFACT_SCHEMA_VERSION",
    "SHADOW_CLASSIFICATION",
    "backup_cohort",
    "build_candidate_report",
    "build_public_status",
    "capture_from_histories",
    "capture_latest",
    "cohort_paths",
    "evaluate_from_histories",
    "evaluate_latest",
    "forward_code_binding",
    "freeze_from_cache",
    "freeze_shadow_cohort",
    "load_shadow_artifact",
    "publish_aggregate_only",
    "publish_local_status",
    "recover_cohort",
    "validate_preregistration",
    "validate_shadow_artifact",
    "verify_cohort",
]
