"""Fail-closed pure-NumPy inference and deep forecast output contract."""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DATA, PROBABILITY_HISTORY_START
from .persistence import load_json
from .probability_contract import (
    CLASS_NAMES,
    HORIZONS,
    INDEPENDENT_MODEL_FAMILY,
    ORDERED_CLASS_NAMES,
    ORDERED_MODEL_FAMILY,
    ROUND_TRIP_COST,
    THRESHOLD_GRIDS,
    model_key,
    ordered_model_key,
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
from .probability_model import (
    GRID_MONOTONICITY_GATES,
    PUBLISH_TRANSFORM,
    apply_publish_transform,
    assess_ood,
    canonical_hash,
    predict_probabilities,
    whole_identity_percentages,
)
from .probability_ordered import (
    ORDERED_MODEL_VERSION,
    ORDERED_MONOTONICITY_EPSILON,
    ORDERED_PUBLISH_TRANSFORM,
    RELEASE_REFIT_BOOTSTRAP_REPS,
    VECTOR_CALIBRATION_VERSION,
    assert_exact_ordered_monotonicity,
    derive_threshold_probabilities,
    predict_ordered_probabilities,
)

ARTIFACT_SCHEMA = "stock-radar-probability-models"
ARTIFACT_SCHEMA_VERSION = 2
VALIDATION_SCHEMA = "stock-radar-probability-validation"
VALIDATION_SCHEMA_VERSION = 2
FORECAST_SCHEMA_VERSION = 1
MODEL_MAX_AGE_DAYS = 45
MAX_BAR_AGE_DAYS = 4
MAX_SPY_CALENDAR_LAG_DAYS = 4
MAX_SPY_BUSINESS_LAG_DAYS = 2
MAX_INTERVAL_WIDTH = 0.20
CURRENT_THRESHOLD_TOLERANCE = 0.005
MODELS_PATH = DATA / "probability_models.json"
VALIDATION_PATH = DATA / "probability_validation.json"
PARTITION = "USD_company_equity"
WITHHELD_MESSAGE = "No validated stock-specific probability edge"
SURVIVOR_WARNING = (
    "Validation uses the currently observable eligible company universe and is "
    "subject to current-universe survivorship bias."
)
HORIZON_LABELS = {21: "1 Monat", 63: "3 Monate", 126: "6 Monate", 252: "12 Monate"}
_FORBIDDEN_FORECAST_CLAIMS = (
    "will rise",
    "guaranteed",
    "expected price",
    "confidence",
    "wird steigen",
    "garantiert",
    "erwarteter preis",
)


class ProbabilityArtifactError(RuntimeError):
    pass


def finalize_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(artifact)
    result.pop("artifact_hash", None)
    result["artifact_hash"] = canonical_hash(result)
    return result


def empty_probability_artifact(
    *,
    created_at: str | None = None,
    reason: str = "Release validation has not been run.",
    model_family: str | None = None,
) -> dict[str, Any]:
    if model_family not in (
        None,
        INDEPENDENT_MODEL_FAMILY,
        ORDERED_MODEL_FAMILY,
    ):
        raise ValueError("unsupported empty probability artifact model family")
    publish_transform = (
        ORDERED_PUBLISH_TRANSFORM
        if model_family == ORDERED_MODEL_FAMILY
        else PUBLISH_TRANSFORM
    )
    created_at = created_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return finalize_artifact(
        {
            "schema": ARTIFACT_SCHEMA,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "engine_version": "probability-mvp-v1",
            "model_family": model_family,
            "created_at": created_at,
            "training_cutoff": None,
            "default_history_start": PROBABILITY_HISTORY_START,
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "code_hash": probability_code_hash(),
            "dependency_versions": dependency_versions(),
            "publish_transform": dict(publish_transform),
            "dataset_binding": None,
            "supported_partition": PARTITION,
            "class_order": list(CLASS_NAMES),
            "round_trip_cost_bps": int(ROUND_TRIP_COST * 10_000),
            "horizons_sessions": list(HORIZONS),
            "threshold_grids_pct": {
                str(horizon): list(THRESHOLD_GRIDS[horizon])
                for horizon in HORIZONS
            },
            "production_status": "withheld",
            "production_reasons": [reason],
            "models": {},
            "baselines": {},
            "accepted_model_keys": [],
            "positive_net_return_model": None,
            "positive_net_return_note": (
                "No separately modeled/calibrated positive-net-return target; it "
                "must not be inferred from material-threshold classes."
            ),
        }
    )


def empty_validation_artifact(
    *,
    created_at: str | None = None,
    reason: str = "Release validation has not been run.",
) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "default_history_start": PROBABILITY_HISTORY_START,
        "model_family": None,
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "code_hash": probability_code_hash(),
        "dependency_versions": dependency_versions(),
        "publish_transform": dict(PUBLISH_TRANSFORM),
        "dataset_binding": None,
        "status": "not_run",
        "accepted_model_count": 0,
        "tested_model_count": 0,
        "reasons": [reason],
        "models": {},
        "bootstrap_method": (
            "fixed OOS prediction two-way issuer/calendar-quarter block bootstrap; "
            "ordered release additionally requires 200 full model+calibrator refits"
        ),
    }


def validate_probability_artifact(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ProbabilityArtifactError("probability artifact root must be an object")
    if (
        artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise ProbabilityArtifactError("unsupported probability artifact schema")
    stored_hash = artifact.get("artifact_hash")
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    if not isinstance(stored_hash, str) or canonical_hash(unhashed) != stored_hash:
        raise ProbabilityArtifactError("probability artifact hash mismatch")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ProbabilityArtifactError("probability feature version mismatch")
    if artifact.get("default_history_start") != PROBABILITY_HISTORY_START:
        raise ProbabilityArtifactError("probability default history boundary mismatch")
    if artifact.get("feature_schema_hash") != feature_schema_hash():
        raise ProbabilityArtifactError("probability feature schema hash mismatch")
    if artifact.get("code_hash") != probability_code_hash():
        raise ProbabilityArtifactError("probability engine code hash mismatch")
    if artifact.get("dependency_versions") != dependency_versions():
        raise ProbabilityArtifactError("probability dependency fingerprint mismatch")
    if artifact.get("supported_partition") != PARTITION:
        raise ProbabilityArtifactError("unsupported probability artifact partition")
    if artifact.get("class_order") != list(CLASS_NAMES):
        raise ProbabilityArtifactError("probability class order mismatch")
    artifact_family = artifact.get("model_family")
    if artifact_family not in (
        None,
        INDEPENDENT_MODEL_FAMILY,
        ORDERED_MODEL_FAMILY,
    ):
        raise ProbabilityArtifactError("unsupported artifact model family")
    effective_artifact_family = (
        artifact_family
        if artifact_family is not None
        else INDEPENDENT_MODEL_FAMILY
    )
    expected_top_transform = (
        ORDERED_PUBLISH_TRANSFORM
        if effective_artifact_family == ORDERED_MODEL_FAMILY
        else PUBLISH_TRANSFORM
    )
    if artifact.get("publish_transform") != expected_top_transform:
        raise ProbabilityArtifactError("probability publish transform mismatch")
    if not isinstance(artifact.get("models"), dict) or not isinstance(
        artifact.get("baselines"), dict
    ):
        raise ProbabilityArtifactError("probability artifact model maps are invalid")
    binding = artifact.get("dataset_binding")
    if artifact["models"]:
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("binding_hash"), str)
            or canonical_hash(
                {key: value for key, value in binding.items() if key != "binding_hash"}
            )
            != binding["binding_hash"]
        ):
            raise ProbabilityArtifactError("probability dataset binding is invalid")
    accepted_keys = artifact.get("accepted_model_keys")
    if not isinstance(accepted_keys, list) or any(
        key not in artifact["models"] for key in accepted_keys
    ):
        raise ProbabilityArtifactError("accepted probability model index is invalid")
    for key, model in artifact["models"].items():
        if not isinstance(model, dict):
            raise ProbabilityArtifactError(f"accepted model {key} is malformed")
        family = model.get(
            "model_family",
            INDEPENDENT_MODEL_FAMILY,
        )
        if family != effective_artifact_family:
            raise ProbabilityArtifactError(
                f"accepted model {key} family is inconsistent with artifact"
            )
        horizon = model.get("horizon_sessions")
        common_invalid = (
            model.get("model_key") != key
            or model.get("accepted") is not True
            or model.get("acceptance_reasons") != []
            or horizon not in HORIZONS
            or (model.get("preprocessor") or {}).get("feature_names")
            != list(FEATURE_NAMES)
            or (
                isinstance(binding, dict)
                and model.get("dataset_binding_hash") != binding.get("binding_hash")
            )
            or not isinstance(model.get("checkpoint_key"), str)
            or not isinstance(model.get("history_years"), (int, float))
            or model["history_years"] < 8
            or not isinstance(model.get("full_test_fold_count"), int)
            or model["full_test_fold_count"] < 5
            or not isinstance(model.get("min_usable_train_years"), (int, float))
            or model["min_usable_train_years"] < 5
        )
        if common_invalid:
            raise ProbabilityArtifactError(f"accepted model {key} is malformed")
        if family == INDEPENDENT_MODEL_FAMILY:
            threshold = model.get("threshold_pct")
            if (
                threshold not in THRESHOLD_GRIDS[horizon]
                or key != model_key(horizon, threshold)
                or not isinstance(
                    (model.get("temperature") or {}).get("value"),
                    (int, float),
                )
                or model.get("publish_transform") != PUBLISH_TRANSFORM
                or not isinstance(model.get("grid_monotonicity"), dict)
                or model["grid_monotonicity"].get("passed") is not True
                or model["grid_monotonicity"].get("gates")
                != GRID_MONOTONICITY_GATES
            ):
                raise ProbabilityArtifactError(
                    f"accepted independent model {key} is malformed"
                )
            class_count = len(CLASS_NAMES)
        elif family == ORDERED_MODEL_FAMILY:
            vector = model.get("vector_scaling") or {}
            exact = model.get("exact_monotonicity") or {}
            refit = model.get("refit_acceptance") or {}
            fixed_bootstraps = model.get("bootstrap_by_threshold") or {}
            fixed_bootstrap_complete = (
                set(fixed_bootstraps)
                == {
                    str(value)
                    for value in THRESHOLD_GRIDS[horizon]
                }
                and all(
                    isinstance(item, dict)
                    for item in fixed_bootstraps.values()
                )
                and all(
                    (
                        requested := int(
                            item.get("requested_repetitions") or 0
                        )
                    )
                    >= 1000
                    and (
                        attempted := int(
                            item.get("attempted_repetitions") or 0
                        )
                    )
                    >= (
                        completed := int(
                        item.get("completed_repetitions")
                        if item.get("completed_repetitions") is not None
                        else item.get("repetitions")
                        or 0
                        )
                    )
                    >= requested
                    and int(item.get("skipped_repetitions") or 0)
                    == attempted - completed
                    and item.get("complete") is True
                    for item in fixed_bootstraps.values()
                )
            )
            if (
                key != ordered_model_key(horizon)
                or model.get("model_version") != ORDERED_MODEL_VERSION
                or model.get("class_names") != list(ORDERED_CLASS_NAMES)
                or model.get("class_order") != list(
                    range(len(ORDERED_CLASS_NAMES))
                )
                or model.get("thresholds_pct")
                != list(THRESHOLD_GRIDS[horizon])
                or model.get("publish_transform")
                != ORDERED_PUBLISH_TRANSFORM
                or vector.get("version") != VECTOR_CALIBRATION_VERSION
                or vector.get("penalty_grid")
                != [0.01, 0.1, 1.0, 10.0, 100.0]
                or vector.get("converged") is not True
                or exact.get("passed") is not True
                or exact.get("tolerance")
                != ORDERED_MONOTONICITY_EPSILON
                or refit.get("production_release_eligible") is not True
                or refit.get("dev_override") is not False
                or int(refit.get("completed_repetitions") or 0)
                < RELEASE_REFIT_BOOTSTRAP_REPS
                or not fixed_bootstrap_complete
            ):
                raise ProbabilityArtifactError(
                    f"accepted ordered model {key} is malformed"
                )
            class_count = len(ORDERED_CLASS_NAMES)
        else:
            raise ProbabilityArtifactError(
                f"accepted model {key} has unsupported family"
            )
        try:
            preprocessor = model["preprocessor"]
            output_dimension = int(preprocessor["output_dimension"])
            coefficient = np.asarray(model["coefficient"], dtype=float)
            intercept = np.asarray(model["intercept"], dtype=float)
            feature_vectors = (
                "winsor_lower_005",
                "winsor_upper_995",
                "median",
                "mean",
                "scale",
            )
            if (
                coefficient.shape != (class_count, output_dimension)
                or intercept.shape != (class_count,)
                or not np.isfinite(coefficient).all()
                or not np.isfinite(intercept).all()
                or any(
                    np.asarray(preprocessor[name], dtype=float).shape
                    != (len(FEATURE_NAMES),)
                    for name in feature_vectors
                )
                or any(
                    not np.isfinite(np.asarray(preprocessor[name], dtype=float)).all()
                    for name in feature_vectors
                )
            ):
                raise ValueError("numeric model/preprocessor shape is invalid")
            if family == INDEPENDENT_MODEL_FAMILY:
                temperature = float(model["temperature"]["value"])
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError("temperature is invalid")
            else:
                scales = np.asarray(
                    model["vector_scaling"]["scales"],
                    dtype=float,
                )
                biases = np.asarray(
                    model["vector_scaling"]["biases"],
                    dtype=float,
                )
                if (
                    scales.shape != (class_count,)
                    or biases.shape != (class_count,)
                    or not np.isfinite(scales).all()
                    or not np.isfinite(biases).all()
                    or (scales <= 0).any()
                    or abs(float(biases.sum())) > 1e-10
                ):
                    raise ValueError("vector scaling parameters are invalid")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProbabilityArtifactError(
                f"accepted model {key} numeric contract is invalid: {exc}"
            ) from exc
        stored_model_hash = model.get("model_hash")
        unhashed_model = {
            name: value for name, value in model.items() if name != "model_hash"
        }
        if (
            not isinstance(stored_model_hash, str)
            or canonical_hash(unhashed_model) != stored_model_hash
        ):
            raise ProbabilityArtifactError(f"model hash mismatch for {key}")
    return artifact


def load_probability_artifact(path: Path = MODELS_PATH) -> dict[str, Any]:
    try:
        artifact = load_json(path, required=True, expected_type=dict)
        return validate_probability_artifact(artifact)
    except Exception as exc:
        if isinstance(exc, ProbabilityArtifactError):
            detail = str(exc)
        else:
            detail = f"{type(exc).__name__}: {exc}"
        raise ProbabilityArtifactError(
            f"invalid_artifact: {detail[:500]}"
        ) from exc


def load_probability_validation_summary(
    path: Path = VALIDATION_PATH,
) -> dict[str, Any]:
    try:
        value = load_json(path, required=True, expected_type=dict)
        if (
            value.get("schema") != VALIDATION_SCHEMA
            or value.get("schema_version") != VALIDATION_SCHEMA_VERSION
            or value.get("feature_version") != FEATURE_VERSION
            or value.get("default_history_start") != PROBABILITY_HISTORY_START
            or value.get("feature_schema_hash") != feature_schema_hash()
            or value.get("code_hash") != probability_code_hash()
            or value.get("dependency_versions") != dependency_versions()
            or value.get("publish_transform")
            not in (PUBLISH_TRANSFORM, ORDERED_PUBLISH_TRANSFORM)
        ):
            raise ValueError("unsupported validation schema")
        models = value.get("models") or {}
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": value.get("status"),
            "model_family": value.get(
                "model_family",
                INDEPENDENT_MODEL_FAMILY,
            ),
            "created_at": value.get("created_at"),
            "default_history_start": value.get("default_history_start"),
            "accepted_model_count": int(value.get("accepted_model_count") or 0),
            "tested_model_count": int(value.get("tested_model_count") or 0),
            "reasons": list(value.get("reasons") or []),
            "provider_coverage": (
                (value.get("dataset_binding") or {}).get("provider")
            ),
            "models": {
                key: {
                    "accepted": bool((report.get("acceptance") or {}).get("accepted")),
                    "model_family": report.get(
                        "model_family",
                        value.get(
                            "model_family",
                            INDEPENDENT_MODEL_FAMILY,
                        ),
                    ),
                    "reasons": list((report.get("acceptance") or {}).get("reasons") or []),
                    "brier_skill": (report.get("aggregate") or {}).get("brier_skill"),
                    "log_loss_improvement": (report.get("aggregate") or {}).get(
                        "log_loss_improvement"
                    ),
                    "classwise_ece": (report.get("aggregate") or {}).get(
                        "classwise_ece"
                    ),
                    "fold_count": report.get("fold_count"),
                    "full_test_fold_count": report.get(
                        "full_test_fold_count"
                    ),
                    "history_years": report.get("history_years"),
                    "min_usable_train_years": report.get(
                        "min_usable_train_years"
                    ),
                    "publish_transform": report.get("publish_transform"),
                    "grid_monotonicity": report.get("grid_monotonicity"),
                    "provider_success_coverage": report.get(
                        "provider_success_coverage"
                    ),
                    "provider_successful_issuer_count": report.get(
                        "provider_successful_issuer_count"
                    ),
                    "threshold_validation": {
                        threshold: {
                            "accepted": bool(
                                (
                                    threshold_report.get("acceptance")
                                    or {}
                                ).get("accepted")
                            ),
                            "reasons": list(
                                (
                                    threshold_report.get("acceptance")
                                    or {}
                                ).get("reasons")
                                or []
                            ),
                            "brier_skill": (
                                threshold_report.get("aggregate") or {}
                            ).get("brier_skill"),
                            "classwise_ece": (
                                threshold_report.get("aggregate") or {}
                            ).get("classwise_ece"),
                            "fixed_oos_bootstrap": {
                                "requested": (
                                    threshold_report.get("bootstrap")
                                    or {}
                                ).get("requested_repetitions"),
                                "attempted": (
                                    threshold_report.get("bootstrap")
                                    or {}
                                ).get("attempted_repetitions"),
                                "completed": (
                                    threshold_report.get("bootstrap")
                                    or {}
                                ).get("completed_repetitions"),
                                "skipped": (
                                    threshold_report.get("bootstrap")
                                    or {}
                                ).get("skipped_repetitions"),
                            },
                        }
                        for threshold, threshold_report in (
                            report.get("threshold_validation") or {}
                        ).items()
                        if isinstance(threshold_report, dict)
                    },
                }
                for key, report in sorted(models.items())
                if isinstance(report, dict)
            },
        }
    except Exception as exc:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "unavailable",
            "created_at": None,
            "default_history_start": PROBABILITY_HISTORY_START,
            "accepted_model_count": 0,
            "tested_model_count": 0,
            "reasons": [f"probability validation unavailable: {str(exc)[:240]}"],
            "provider_coverage": None,
            "models": {},
        }


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _select_spy_asof_history(
    spy_history: pd.DataFrame,
    signal_timestamp: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(spy_history, pd.DataFrame) or spy_history.empty:
        raise ValueError("SPY history is unavailable")
    signal = pd.Timestamp(signal_timestamp)
    if pd.isna(signal):
        raise ValueError("signal timestamp is invalid")
    if signal.tzinfo is not None:
        signal = signal.tz_convert(None)
    signal = signal.normalize()
    index = pd.to_datetime(
        spy_history.index,
        errors="coerce",
        utc=True,
    ).tz_convert(None).normalize()
    valid = ~index.isna()
    eligible = valid & (index <= signal)
    if not eligible.any():
        raise ValueError(
            "future SPY history: no completed SPY session at or before signal"
        )
    selected = pd.Timestamp(index[eligible].max())
    calendar_lag = int((signal - selected).days)
    business_lag = int(
        np.busday_count(
            np.datetime64(selected.date()),
            np.datetime64(signal.date()),
        )
    )
    if (
        calendar_lag > MAX_SPY_CALENDAR_LAG_DAYS
        or business_lag > MAX_SPY_BUSINESS_LAG_DAYS
    ):
        raise ValueError(
            "stale SPY bar: latest as-of session "
            f"{selected.date()} lags signal {signal.date()} by "
            f"{calendar_lag} calendar/{business_lag} business days; maxima are "
            f"{MAX_SPY_CALENDAR_LAG_DAYS}/{MAX_SPY_BUSINESS_LAG_DAYS}"
        )
    selected_history = spy_history.loc[np.asarray(eligible)].copy()
    selected_history = selected_history.sort_index()
    return selected_history, {
        "selected_session": selected.date().isoformat(),
        "signal_session": signal.date().isoformat(),
        "calendar_lag_days": calendar_lag,
        "business_lag_days": business_lag,
        "maximum_calendar_lag_days": MAX_SPY_CALENDAR_LAG_DAYS,
        "maximum_business_lag_days": MAX_SPY_BUSINESS_LAG_DAYS,
        "alignment": "latest completed SPY session <= stock signal session",
    }


def _artifact_age_reason(model: dict[str, Any], now: datetime) -> str | None:
    trained = _parse_time(model.get("trained_at"))
    if trained is None:
        return "model training timestamp is missing/invalid"
    age = (now - trained).total_seconds() / 86400.0
    if age < -1:
        return "model training timestamp is in the future"
    if age > MODEL_MAX_AGE_DAYS:
        return f"model age {age:.1f} days exceeds {MODEL_MAX_AGE_DAYS} days"
    return None


def _baseline_grid(
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[int, list[float]]] = {}
    records: dict[tuple[int, int], dict[str, Any]] = {}
    sources = dict(artifact.get("baselines") or {})
    for key, model in (artifact.get("models") or {}).items():
        sources.setdefault(
            key,
            {
                "model_key": key,
                "horizon_sessions": model.get("horizon_sessions"),
                "threshold_pct": model.get("threshold_pct"),
                "rates": model.get("baseline_rates"),
                "sample_size": model.get("oos_sample_size"),
                "validation": {
                    "accepted": True,
                    "brier_skill": (model.get("oos_metrics") or {}).get(
                        "brier_skill"
                    ),
                    "classwise_ece": (model.get("oos_metrics") or {}).get(
                        "classwise_ece"
                    ),
                    "fold_count": model.get("fold_count"),
                    "full_test_fold_count": model.get(
                        "full_test_fold_count"
                    ),
                    "history_years": model.get("history_years"),
                    "min_usable_train_years": model.get(
                        "min_usable_train_years"
                    ),
                },
                "event_counts_calibration_test": model.get(
                    "event_counts_calibration_test"
                ),
            },
        )
    for key, source in sorted(sources.items()):
        try:
            horizon = int(source["horizon_sessions"])
            threshold = int(source["threshold_pct"])
            rates = source.get("rates") or source.get("baseline_rates")
            row = [float(rates[name]) for name in CLASS_NAMES]
            if not np.isfinite(row).all() or not np.isclose(sum(row), 1.0, atol=1e-6):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(horizon, {})[threshold] = row
        records[(horizon, threshold)] = source
    output = []
    for horizon in sorted(grouped):
        for threshold in sorted(grouped[horizon]):
            source = records[(horizon, threshold)]
            raw_rates = grouped[horizon][threshold]
            percentages = whole_identity_percentages(raw_rates)
            validation = source.get("validation") or {}
            output.append(
                {
                    "model_key": model_key(horizon, threshold),
                    "model_family": source.get(
                        "model_family",
                        INDEPENDENT_MODEL_FAMILY,
                    ),
                    "source_model_key": source.get("source_model_key"),
                    "horizon_sessions": horizon,
                    "horizon_label": HORIZON_LABELS[horizon],
                    "threshold_pct": threshold,
                    "rates": dict(zip(CLASS_NAMES, raw_rates)),
                    "rates_pct": dict(zip(CLASS_NAMES, percentages)),
                    "sample_size": source.get("sample_size"),
                    "accepted_stock_specific_model": bool(
                        validation.get("accepted")
                    ),
                    "brier_skill": validation.get("brier_skill"),
                    "classwise_ece": validation.get("classwise_ece"),
                    "fold_count": validation.get("fold_count"),
                    "full_test_fold_count": source.get(
                        "full_test_fold_count"
                    ),
                    "history_years": source.get("history_years"),
                    "min_usable_train_years": source.get(
                        "min_usable_train_years"
                    ),
                }
            )
    return output


def _base_forecast(
    row: dict[str, Any],
    artifact: dict[str, Any] | None,
    reasons: Iterable[str],
) -> dict[str, Any]:
    reason_list = list(dict.fromkeys(str(reason) for reason in reasons if reason))
    supported_baseline_partition = (
        row.get("asset_type") == "company_equity"
        and row.get("currency") == "USD"
    )
    baselines = (
        _baseline_grid(artifact)
        if artifact and supported_baseline_partition
        else []
    )
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "status": "withheld",
        "message": WITHHELD_MESSAGE,
        "actionable": False,
        "separate_from_radar_score": True,
        "separate_from_insight_ranking": True,
        "separate_from_sweet_spot": True,
        "listing_currency": row.get("currency"),
        "supported_partition": PARTITION,
        "signal_timestamp": row.get("bar_timestamp") or row.get("bar_date"),
        "entry_assumption": (
            "first adjusted/raw-equivalent open strictly after completed signal close"
        ),
        "cost_assumption_bps_round_trip": 30,
        "outcome_definition": (
            "DOWN gross return <= -(X+0.30%), MIDDLE between boundaries, "
            "UP gross return >= +(X+0.30%); exit is adjusted close H sessions after t"
        ),
        "positive_net_return_probability": None,
        "positive_net_return_note": (
            "Not modeled/calibrated separately and not inferred from threshold classes."
        ),
        "reasons": reason_list,
        "ood": [],
        "baselines": baselines,
        "forecasts": [],
        "artifact_created_at": artifact.get("created_at") if artifact else None,
        "training_cutoff": artifact.get("training_cutoff") if artifact else None,
        "survivorship_warning": SURVIVOR_WARNING,
    }


def _eligibility_reasons(
    row: dict[str, Any],
    history: pd.DataFrame | None,
    *,
    duplicate_listing: bool,
    require_history: bool = True,
) -> list[str]:
    reasons = []
    if row.get("asset_type") != "company_equity":
        reasons.append("unsupported partition: instrument is not a company equity")
    if row.get("currency") != "USD":
        reasons.append("unsupported partition: listing currency is not USD")
    if duplicate_listing:
        reasons.append("duplicate issuer listing excluded")
    if require_history and (
        history is None or len(history) - 1 < MIN_HISTORY_BARS
    ):
        reasons.append(
            f"insufficient history: at least {MIN_HISTORY_BARS} bars before t required"
        )
    if row.get("completed_bars_only") is not True:
        reasons.append("latest bar is not verified as completed")
    if row.get("source_interval") not in (None, "1d"):
        reasons.append("source interval is not completed daily")
    bar_age = row.get("bar_age_days")
    if not isinstance(bar_age, (int, float)) or not math.isfinite(bar_age):
        reasons.append("completed-bar age is unavailable")
    elif bar_age < 0 or bar_age > MAX_BAR_AGE_DAYS:
        reasons.append(
            f"stale/incomplete bar: age {bar_age} days outside 0-{MAX_BAR_AGE_DAYS}"
        )
    return reasons


def _probability_interval(
    probability: float,
    model: dict[str, Any],
    class_name: str,
    *,
    threshold: int | None = None,
) -> tuple[float, float]:
    bootstrap = (
        (model.get("bootstrap_by_threshold") or {}).get(str(threshold), {})
        if model.get("model_family") == ORDERED_MODEL_FAMILY
        else model.get("bootstrap") or {}
    )
    offsets = (
        bootstrap
        .get("probability_error_offsets_ci95", {})
        .get(class_name)
    )
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, (int, float)) for value in offsets)
    ):
        raise ValueError("95% probability interval offsets are unavailable")
    candidates = [probability, probability + offsets[0], probability + offsets[1]]
    return max(0.0, min(candidates)), min(1.0, max(candidates))


def evaluate_current_threshold_grid(
    probabilities_by_threshold: dict[int, np.ndarray],
) -> dict[str, Any]:
    thresholds = sorted(probabilities_by_threshold)
    displays = {
        threshold: whole_identity_percentages(
            probabilities_by_threshold[threshold]
        )
        for threshold in thresholds
    }
    up_magnitudes = []
    down_magnitudes = []
    for easier, harder in zip(thresholds, thresholds[1:]):
        up_magnitudes.append(
            max(
                0.0,
                float(
                    probabilities_by_threshold[harder][2]
                    - probabilities_by_threshold[easier][2]
                ),
            )
        )
        down_magnitudes.append(
            max(
                0.0,
                float(
                    probabilities_by_threshold[harder][0]
                    - probabilities_by_threshold[easier][0]
                ),
            )
        )
    max_up = max(up_magnitudes, default=0.0)
    max_down = max(down_magnitudes, default=0.0)
    display_monotonic = all(
        displays[harder][2] <= displays[easier][2]
        and displays[harder][0] <= displays[easier][0]
        for easier, harder in zip(thresholds, thresholds[1:])
    )
    permitted = (
        max_up <= CURRENT_THRESHOLD_TOLERANCE
        and max_down <= CURRENT_THRESHOLD_TOLERANCE
        and display_monotonic
    )
    tolerated = permitted and max(max_up, max_down) > 1e-12
    return {
        "permitted": permitted,
        "reason_code": None if permitted else "current_threshold_non_monotonic",
        "thresholds": thresholds,
        "max_up_inversion": max_up,
        "max_down_inversion": max_down,
        "tolerance": CURRENT_THRESHOLD_TOLERANCE,
        "whole_percent_display_monotonic": display_monotonic,
        "tolerated_independent_threshold_inversion": tolerated,
        "disclosure": (
            "Independent-threshold tolerance applied: raw inversion is at most "
            "0.5 percentage point, whole-percent display is monotonic, and raw "
            "probabilities are unchanged."
            if tolerated
            else "Raw independent-threshold probabilities are monotonic."
            if permitted
            else None
        ),
        "action": "never project; withhold horizon when not permitted",
    }


def _accepted_forecasts(
    artifact: dict[str, Any],
    feature_vector: dict[str, float],
    now: datetime,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    frame = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
    raw_by_horizon: dict[int, dict[int, np.ndarray]] = {}
    models_by_key: dict[str, dict[str, Any]] = {}
    artifact_keys_by_key: dict[str, str] = {}
    exact_by_horizon: dict[int, dict[str, Any]] = {}
    ordered_horizons: set[int] = set()
    model_reasons: list[str] = []
    ood_records = []
    for artifact_key in sorted(artifact.get("accepted_model_keys") or []):
        model = artifact["models"][artifact_key]
        family = model.get(
            "model_family",
            INDEPENDENT_MODEL_FAMILY,
        )
        if model.get("accepted") is not True or model.get("acceptance_reasons"):
            model_reasons.append(f"{artifact_key}: acceptance failed")
            continue
        age_reason = _artifact_age_reason(model, now)
        if age_reason:
            model_reasons.append(f"{artifact_key}: {age_reason}")
            continue
        ood = assess_ood(frame, model["preprocessor"])[0]
        ood_record = {"model_key": artifact_key, **ood}
        ood_records.append(ood_record)
        if ood["withhold"]:
            model_reasons.extend(
                f"{artifact_key}: {reason}" for reason in ood["reasons"]
            )
            continue
        try:
            horizon = int(model["horizon_sessions"])
            if family == ORDERED_MODEL_FAMILY:
                ordered = predict_ordered_probabilities(
                    model,
                    frame,
                    require_complete=True,
                )[0]
                derived = derive_threshold_probabilities(
                    ordered,
                    THRESHOLD_GRIDS[horizon],
                )
                exact_by_horizon[horizon] = (
                    assert_exact_ordered_monotonicity(derived)
                )
                ordered_horizons.add(horizon)
                for threshold, probability in derived.items():
                    key = model_key(horizon, threshold)
                    raw_by_horizon.setdefault(horizon, {})[
                        threshold
                    ] = probability
                    models_by_key[key] = model
                    artifact_keys_by_key[key] = artifact_key
            else:
                probability = predict_probabilities(
                    model,
                    frame,
                    require_complete=True,
                )[0]
                probability = apply_publish_transform(
                    probability,
                    model.get("publish_transform"),
                )
                threshold = int(model["threshold_pct"])
                key = model_key(horizon, threshold)
                raw_by_horizon.setdefault(horizon, {})[
                    threshold
                ] = probability
                models_by_key[key] = model
                artifact_keys_by_key[key] = artifact_key
        except Exception as exc:
            model_reasons.append(
                f"{artifact_key}: inference/model hash failure: "
                f"{str(exc)[:200]}"
            )
            continue

    forecasts = []
    for horizon in sorted(raw_by_horizon):
        thresholds = sorted(raw_by_horizon[horizon])
        is_ordered = horizon in ordered_horizons
        if is_ordered and thresholds != list(THRESHOLD_GRIDS[horizon]):
            model_reasons.append(
                f"h{horizon}: ordered horizon is incomplete; baseline only"
            )
            continue
        current_grid = (
            exact_by_horizon[horizon]
            if is_ordered
            else evaluate_current_threshold_grid(raw_by_horizon[horizon])
        )
        if not current_grid["permitted"]:
            model_reasons.append(
                f"h{horizon}: current_threshold_non_monotonic; max UP inversion "
                f"{current_grid['max_up_inversion']:.6f}, max DOWN inversion "
                f"{current_grid['max_down_inversion']:.6f}; entire horizon "
                "withheld without projection"
            )
            continue
        horizon_forecasts = []
        horizon_failures = []
        for threshold in thresholds:
            key = model_key(horizon, threshold)
            model = models_by_key[key]
            probability = raw_by_horizon[horizon][threshold]
            intervals: dict[str, list[int]] = {}
            widest = 0.0
            interval_error = None
            for index, name in enumerate(CLASS_NAMES):
                try:
                    lower, upper = _probability_interval(
                        float(probability[index]),
                        model,
                        name,
                        threshold=threshold,
                    )
                    widest = max(widest, upper - lower)
                    intervals[name] = [
                        int(math.floor(lower * 100)),
                        int(math.ceil(upper * 100)),
                    ]
                except ValueError as exc:
                    interval_error = str(exc)
                    break
            if interval_error:
                horizon_failures.append(f"{key}: {interval_error}")
                continue
            if widest > MAX_INTERVAL_WIDTH + 1e-12:
                horizon_failures.append(
                    f"{key}: 95% model interval width {widest:.3f} exceeds "
                    f"{MAX_INTERVAL_WIDTH:.2f}"
                )
                continue
            percentages = whole_identity_percentages(probability)
            for index, name in enumerate(CLASS_NAMES):
                intervals[name][0] = min(intervals[name][0], percentages[index])
                intervals[name][1] = max(intervals[name][1], percentages[index])
            if any(
                intervals[name][1] - intervals[name][0]
                > int(MAX_INTERVAL_WIDTH * 100)
                for name in CLASS_NAMES
            ):
                horizon_failures.append(
                    f"{key}: rounded 95% model interval exceeds "
                    f"{int(MAX_INTERVAL_WIDTH * 100)} points"
                )
                continue
            baseline_source = (
                (model.get("baseline_rates_by_threshold") or {}).get(
                    str(threshold)
                )
                if is_ordered
                else model.get("baseline_rates")
            )
            baseline_rates = [
                float(baseline_source[name]) for name in CLASS_NAMES
            ]
            baseline_pct = whole_identity_percentages(baseline_rates)
            metrics = (
                (model.get("oos_metrics_by_threshold") or {}).get(
                    str(threshold),
                    {},
                )
                if is_ordered
                else model["oos_metrics"]
            )
            bootstrap_report = (
                (model.get("bootstrap_by_threshold") or {}).get(
                    str(threshold),
                    {},
                )
                if is_ordered
                else model.get("bootstrap") or {}
            )
            gross_boundary = threshold + ROUND_TRIP_COST * 100
            horizon_forecasts.append(
                {
                    "model_key": key,
                    "artifact_model_key": artifact_keys_by_key[key],
                    "model_family": (
                        ORDERED_MODEL_FAMILY
                        if is_ordered
                        else INDEPENDENT_MODEL_FAMILY
                    ),
                    "horizon_sessions": horizon,
                    "horizon_label": HORIZON_LABELS[horizon],
                    "threshold_pct": threshold,
                    "definition": (
                        f"gross UP >= +{gross_boundary:.2f}%; gross DOWN <= "
                        f"-{gross_boundary:.2f}%; otherwise MIDDLE"
                    ),
                    "probabilities": {
                        name: float(probability[index])
                        for index, name in enumerate(CLASS_NAMES)
                    },
                    "probabilities_pct": dict(zip(CLASS_NAMES, percentages)),
                    "sum_pct": int(sum(percentages)),
                    "model_interval_95_pct": intervals,
                    "model_interval_method": (
                        "95% aggregate calibration-error interval approximation "
                        "from fixed OOS predictions; not an individual stock "
                        "outcome interval"
                    ),
                    "publish_transform": dict(
                        model.get("publish_transform")
                    ),
                    "threshold_monotonicity": current_grid,
                    "sample_size": model.get("oos_sample_size"),
                    "baseline_rates_pct": dict(
                        zip(CLASS_NAMES, baseline_pct)
                    ),
                    "brier_skill": metrics.get("brier_skill"),
                    "log_loss_improvement": metrics.get(
                        "log_loss_improvement"
                    ),
                    "classwise_ece": metrics.get("classwise_ece"),
                    "maximum_gap": metrics.get("maximum_gap"),
                    "fixed_oos_bootstrap": {
                        "requested": bootstrap_report.get(
                            "requested_repetitions"
                        ),
                        "attempted": bootstrap_report.get(
                            "attempted_repetitions"
                        ),
                        "completed": bootstrap_report.get(
                            "completed_repetitions",
                            bootstrap_report.get("repetitions"),
                        ),
                        "skipped": bootstrap_report.get(
                            "skipped_repetitions"
                        ),
                    },
                    "fold_count": model.get("fold_count"),
                    "full_test_fold_count": model.get(
                        "full_test_fold_count"
                    ),
                    "history_years": model.get("history_years"),
                    "min_usable_train_years": model.get(
                        "min_usable_train_years"
                    ),
                    "artifact_trained_at": model.get("trained_at"),
                    "training_cutoff": model.get("training_cutoff"),
                }
            )
        if is_ordered and (
            horizon_failures
            or len(horizon_forecasts) != len(THRESHOLD_GRIDS[horizon])
        ):
            model_reasons.extend(horizon_failures)
            model_reasons.append(
                f"h{horizon}: ordered horizon failure; all stock-specific "
                "thresholds withheld and baselines retained"
            )
            continue
        model_reasons.extend(horizon_failures)
        forecasts.extend(horizon_forecasts)
    return forecasts, model_reasons, ood_records


def score_probability_row(
    row: dict[str, Any],
    history: pd.DataFrame | None,
    spy_history: pd.DataFrame | None,
    artifact: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    duplicate_listing: bool = False,
    artifact_error: str | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    has_accepted_models = bool(
        artifact and artifact.get("accepted_model_keys")
    )
    fatal_reasons = _eligibility_reasons(
        row,
        history,
        duplicate_listing=duplicate_listing,
        require_history=has_accepted_models,
    )
    if artifact_error:
        fatal_reasons.append(artifact_error)
    if artifact is None:
        fatal_reasons.append("probability artifact is unavailable")
        return _base_forecast(row, None, fatal_reasons)
    production_reasons = list(artifact.get("production_reasons") or [])
    base = _base_forecast(
        row,
        artifact,
        [*fatal_reasons, *production_reasons],
    )
    if fatal_reasons:
        return validate_probability_forecast(base)
    if not artifact.get("accepted_model_keys"):
        if not base["reasons"]:
            base["reasons"] = ["no horizon/threshold model passed release acceptance"]
        return validate_probability_forecast(base)
    if spy_history is None:
        base["reasons"].append("missing/insufficient completed SPY history")
        return validate_probability_forecast(base)
    try:
        spy_asof_history, spy_asof = _select_spy_asof_history(
            spy_history,
            row.get("bar_date")
            or row.get("bar_timestamp"),
        )
    except (TypeError, ValueError, IndexError) as exc:
        base["reasons"].append(str(exc))
        return validate_probability_forecast(base)
    if len(spy_asof_history) - 1 < MIN_HISTORY_BARS:
        base["reasons"].append("missing/insufficient completed SPY history")
        return validate_probability_forecast(base)
    try:
        timestamp, feature_vector = latest_probability_features(
            history,
            spy_asof_history,
            as_of=row.get("bar_date"),
        )
        if row.get("bar_date") and timestamp.date().isoformat() != str(
            row["bar_date"]
        ):
            raise ValueError(
                f"feature session {timestamp.date()} does not match row bar "
                f"{row.get('bar_date')}"
            )
    except Exception as exc:
        base["reasons"].append(f"current feature coverage failed: {str(exc)[:300]}")
        return validate_probability_forecast(base)
    forecasts, model_reasons, ood = _accepted_forecasts(
        artifact, feature_vector, now
    )
    ood_by_model = {item["model_key"]: item for item in ood}
    for item in forecasts:
        item.update(
            {
                "listing_currency": row.get("currency"),
                "signal_timestamp": row.get("bar_timestamp")
                or row.get("bar_date"),
                "entry_assumption": base["entry_assumption"],
                "exit_assumption": (
                    f"adjusted close {item['horizon_sessions']} sessions after t"
                ),
                "cost_assumption_bps_round_trip": 30,
                "artifact_created_at": artifact.get("created_at"),
                "survivorship_warning": SURVIVOR_WARNING,
                "ood": ood_by_model.get(
                    item.get("artifact_model_key")
                    or item["model_key"]
                ),
                "acceptance_passed": True,
                "withholding_reasons": [],
            }
        )
    base["forecasts"] = forecasts
    base["spy_asof"] = spy_asof
    base["ood"] = ood
    base["reasons"].extend(model_reasons)
    base["reasons"] = list(dict.fromkeys(base["reasons"]))
    if forecasts:
        expected_count = sum(
            len(THRESHOLD_GRIDS[int(model["horizon_sessions"])])
            if model.get("model_family") == ORDERED_MODEL_FAMILY
            else 1
            for key, model in (artifact.get("models") or {}).items()
            if key in set(artifact.get("accepted_model_keys") or [])
        )
        base["status"] = (
            "accepted"
            if len(forecasts) == expected_count == sum(
                len(THRESHOLD_GRIDS[horizon]) for horizon in HORIZONS
            )
            else "partial"
        )
        base["message"] = (
            "Strictly validated calibrated material-move probabilities"
            if base["status"] == "accepted"
            else "Only the listed horizon/threshold models passed all current gates"
        )
    return validate_probability_forecast(base)


def _duplicate_listing_symbols(rows: list[dict[str, Any]]) -> set[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("asset_type") != "company_equity" or row.get("currency") != "USD":
            continue
        key = str(row.get("issuer_uuid") or row.get("issuer_key") or "").strip()
        if key and not key.startswith("symbol:"):
            groups.setdefault(key, []).append(row)
    duplicates = set()
    for members in groups.values():
        if len(members) <= 1:
            continue
        ordered = sorted(
            members,
            key=lambda item: (
                0
                if str(item.get("listing_market") or "").upper() in {"NYSE", "NASDAQ"}
                else 1,
                "." in str(item.get("symbol") or ""),
                str(item.get("symbol") or ""),
            ),
        )
        duplicates.update(str(row.get("symbol")) for row in ordered[1:])
    return duplicates


def attach_probability_forecasts(
    rows: list[dict[str, Any]],
    histories: dict[str, pd.DataFrame],
    *,
    spy_history: pd.DataFrame | None = None,
    artifact_path: Path = MODELS_PATH,
    now: datetime | None = None,
    embed_baselines: bool = True,
) -> list[dict[str, Any]]:
    """Attach forecasts without reading or mutating any score/ranking field."""
    try:
        artifact = load_probability_artifact(artifact_path)
        artifact_error = None
    except ProbabilityArtifactError as exc:
        artifact = None
        artifact_error = str(exc)
    duplicates = _duplicate_listing_symbols(rows)
    spy = spy_history if spy_history is not None else histories.get("SPY")
    for row in rows:
        symbol = str(row.get("symbol") or "")
        row["probability_forecast"] = score_probability_row(
            row,
            histories.get(symbol),
            spy,
            artifact,
            now=now,
            duplicate_listing=symbol in duplicates,
            artifact_error=artifact_error,
        )
        if not embed_baselines:
            row["probability_forecast"]["baselines"] = []
    return rows


def load_probability_baselines(
    path: Path = MODELS_PATH,
) -> list[dict[str, Any]]:
    """Load the shared descriptive baseline catalog, failing closed."""
    try:
        artifact = load_probability_artifact(path)
    except ProbabilityArtifactError:
        return []
    return _baseline_grid(artifact)


def validate_probability_forecast(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("probability forecast must be an object")
    if value.get("schema_version") != FORECAST_SCHEMA_VERSION:
        raise ValueError("unsupported probability forecast schema")
    if value.get("status") not in {"accepted", "partial", "withheld"}:
        raise ValueError("invalid probability forecast status")
    if value.get("actionable") is not False:
        raise ValueError("probability forecast must remain non-actionable")
    if not all(
        value.get(key) is True
        for key in (
            "separate_from_radar_score",
            "separate_from_insight_ranking",
            "separate_from_sweet_spot",
        )
    ):
        raise ValueError("probability/ranking separation contract failed")
    if not isinstance(value.get("reasons"), list) or not isinstance(
        value.get("baselines"), list
    ) or not isinstance(value.get("forecasts"), list):
        raise ValueError("probability forecast lists are invalid")
    if value.get("positive_net_return_probability") is not None:
        raise ValueError("positive-net-return probability was not separately modeled")
    for forecast in value["forecasts"]:
        probabilities = forecast.get("probabilities_pct")
        raw_probabilities = forecast.get("probabilities")
        intervals = forecast.get("model_interval_95_pct")
        family = forecast.get(
            "model_family",
            INDEPENDENT_MODEL_FAMILY,
        )
        expected_transform = (
            ORDERED_PUBLISH_TRANSFORM
            if family == ORDERED_MODEL_FAMILY
            else PUBLISH_TRANSFORM
        )
        if (
            forecast.get("acceptance_passed") is not True
            or forecast.get("withholding_reasons") != []
            or forecast.get("listing_currency") != value.get("listing_currency")
            or forecast.get("cost_assumption_bps_round_trip") != 30
            or not isinstance(forecast.get("entry_assumption"), str)
            or not isinstance(forecast.get("exit_assumption"), str)
            or not isinstance(forecast.get("survivorship_warning"), str)
            or not isinstance(forecast.get("training_cutoff"), str)
            or not isinstance(forecast.get("ood"), dict)
            or forecast["ood"].get("withhold") is not False
            or family
            not in (INDEPENDENT_MODEL_FAMILY, ORDERED_MODEL_FAMILY)
            or forecast.get("publish_transform") != expected_transform
            or not isinstance(forecast.get("threshold_monotonicity"), dict)
            or forecast["threshold_monotonicity"].get("permitted") is not True
        ):
            raise ValueError("accepted probability forecast provenance is incomplete")
        if (
            not isinstance(raw_probabilities, dict)
            or set(raw_probabilities) != set(CLASS_NAMES)
            or any(
                not isinstance(raw_probabilities[name], (int, float))
                or not math.isfinite(raw_probabilities[name])
                or not 0 <= raw_probabilities[name] <= 1
                for name in CLASS_NAMES
            )
            or not math.isclose(
                sum(raw_probabilities.values()), 1.0, abs_tol=1e-12
            )
        ):
            raise ValueError("raw published probabilities are not a simplex")
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != set(CLASS_NAMES)
            or any(
                not isinstance(probabilities[name], int)
                or not 0 <= probabilities[name] <= 100
                for name in CLASS_NAMES
            )
            or sum(probabilities.values()) != 100
            or forecast.get("sum_pct") != 100
        ):
            raise ValueError("whole-number probability display violates simplex")
        if not isinstance(intervals, dict) or set(intervals) != set(CLASS_NAMES):
            raise ValueError("published 95% model interval is missing")
        for name in CLASS_NAMES:
            interval = intervals[name]
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(isinstance(point, int) for point in interval)
                or not all(0 <= point <= 100 for point in interval)
                or interval[0] > probabilities[name]
                or interval[1] < probabilities[name]
                or interval[1] - interval[0] > int(MAX_INTERVAL_WIDTH * 100)
            ):
                raise ValueError("published model interval is invalid/too wide")
    grids: dict[int, list[dict[str, Any]]] = {}
    for forecast in value["forecasts"]:
        grids.setdefault(int(forecast["horizon_sessions"]), []).append(forecast)
    for items in grids.values():
        items.sort(key=lambda item: int(item["threshold_pct"]))
        probability_grid = {
            int(item["threshold_pct"]): np.asarray(
                [
                    item["probabilities"]["down"],
                    item["probabilities"]["middle"],
                    item["probabilities"]["up"],
                ],
                dtype=float,
            )
            for item in items
        }
        diagnostic = (
            assert_exact_ordered_monotonicity(probability_grid)
            if any(
                item.get("model_family") == ORDERED_MODEL_FAMILY
                for item in items
            )
            else evaluate_current_threshold_grid(probability_grid)
        )
        if not diagnostic["permitted"]:
            raise ValueError("current_threshold_non_monotonic")
    if value["status"] == "withheld" and value["forecasts"]:
        raise ValueError("withheld forecast cannot contain instrument probabilities")
    text = json.dumps(value, ensure_ascii=False).casefold()
    forbidden = [claim for claim in _FORBIDDEN_FORECAST_CLAIMS if claim in text]
    if forbidden:
        raise ValueError(f"forbidden probability claim language: {forbidden}")
    return value


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_SCHEMA_VERSION",
    "CURRENT_THRESHOLD_TOLERANCE",
    "FORECAST_SCHEMA_VERSION",
    "MAX_INTERVAL_WIDTH",
    "MAX_SPY_BUSINESS_LAG_DAYS",
    "MAX_SPY_CALENDAR_LAG_DAYS",
    "MODEL_MAX_AGE_DAYS",
    "MODELS_PATH",
    "PARTITION",
    "ProbabilityArtifactError",
    "SURVIVOR_WARNING",
    "VALIDATION_PATH",
    "WITHHELD_MESSAGE",
    "attach_probability_forecasts",
    "empty_probability_artifact",
    "empty_validation_artifact",
    "evaluate_current_threshold_grid",
    "finalize_artifact",
    "load_probability_artifact",
    "load_probability_validation_summary",
    "load_probability_baselines",
    "score_probability_row",
    "validate_probability_artifact",
    "validate_probability_forecast",
]
