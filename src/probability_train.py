"""CLI for probability panel construction, walk-forward validation, and release."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DATA, PROBABILITY_HISTORY_START
from .persistence import atomic_write_json, load_json
from .probability_dataset import (
    DEFAULT_CACHE,
    DEFAULT_START,
    HORIZONS,
    THRESHOLD_GRIDS,
    build_weekly_dataset,
    dataset_content_hash,
    dataset_content_summary,
    download_probability_panel,
    label_column,
    load_probability_panel,
    load_training_universe,
    load_weekly_dataset,
    make_purged_expanding_folds,
    model_key,
    ordered_label_column,
)
from .probability_contract import (
    INDEPENDENT_MODEL_FAMILY,
    ORDERED_CLASS_NAMES,
    ORDERED_MODEL_FAMILY,
    ordered_model_key,
)
from .probability_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    dependency_versions,
    feature_schema_hash,
    probability_code_hash,
)
from .probability_inference import (
    ARTIFACT_SCHEMA,
    ARTIFACT_SCHEMA_VERSION,
    MODELS_PATH,
    PARTITION,
    VALIDATION_PATH,
    VALIDATION_SCHEMA,
    VALIDATION_SCHEMA_VERSION,
    empty_probability_artifact,
    finalize_artifact,
    validate_probability_artifact,
)
from .probability_model import (
    CALIBRATION_VERSION,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_SEED,
    GRID_MONOTONICITY_GATES,
    MODEL_VERSION,
    PUBLISH_TRANSFORM,
    RELEASE_BOOTSTRAP_REPS,
    STRICT_ACCEPTANCE_GATES,
    apply_grid_release_gate,
    canonical_hash,
    evaluate_acceptance,
    fit_release_model,
    train_walk_forward_model,
)
from .probability_ordered import (
    ORDERED_MODEL_VERSION,
    ORDERED_PUBLISH_TRANSFORM,
    RELEASE_REFIT_BOOTSTRAP_REPS,
    VECTOR_CALIBRATION_VERSION,
    VECTOR_PENALTY_GRID,
    fit_ordered_release_model,
    ordered_labels_to_threshold_labels,
    train_ordered_walk_forward_horizon,
)

PROBABILITY_EXPERIMENTS_DIR = DATA / "probability_experiments"
ORDERED_VALIDATION_PATH = (
    PROBABILITY_EXPERIMENTS_DIR / "ordered-vector-v1_validation.json"
)
ORDERED_MODELS_EXPERIMENT_PATH = (
    PROBABILITY_EXPERIMENTS_DIR / "ordered-vector-v1_models.json"
)


def make_synthetic_dataset(
    *,
    learnable: bool,
    issuer_count: int = 210,
    week_count: int = 760,
    seed: int = DEFAULT_SEED,
    model_family: str = INDEPENDENT_MODEL_FAMILY,
) -> pd.DataFrame:
    """Create a balanced temporal panel used only for bounded engine smoke tests."""
    rng = np.random.default_rng(seed + (0 if learnable else 99))
    dates = pd.date_range("2012-01-06", periods=week_count, freq="W-FRI")
    symbols = [f"SYN{index:04d}" for index in range(issuer_count)]
    date_values = np.repeat(dates.to_numpy(), issuer_count)
    issuer_values = np.tile(np.asarray(symbols, dtype=object), week_count)
    row_count = len(date_values)
    matrix = rng.normal(0.0, 1.0, size=(row_count, len(FEATURE_NAMES)))
    common_cycle = np.repeat(
        np.sin(np.arange(week_count) / 17.0), issuer_count
    )
    matrix[:, 0] += common_cycle * 0.25
    matrix[:, FEATURE_NAMES.index("spy_price_sma200")] = np.repeat(
        np.sin(np.arange(week_count) / 40.0), issuer_count
    )
    matrix[:, FEATURE_NAMES.index("spy_vol_60")] = np.repeat(
        0.15 + 0.05 * (1 + np.cos(np.arange(week_count) / 21.0)),
        issuer_count,
    )
    latent = (
        1.0 * matrix[:, 0]
        - 0.7 * matrix[:, 1]
        + 0.5 * matrix[:, 4]
    )
    draws = rng.random(row_count)
    if model_family == INDEPENDENT_MODEL_FAMILY:
        if learnable:
            logits = np.column_stack(
                [
                    -1.45 * latent,
                    np.full(row_count, 0.45),
                    1.45 * latent,
                ]
            )
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            labels = (
                draws[:, None] > np.cumsum(probabilities, axis=1)
            ).sum(axis=1)
        else:
            labels = rng.integers(0, 3, row_count)
        ordered_labels = np.choose(labels, [0, 3, 6])
    elif model_family == ORDERED_MODEL_FAMILY:
        centers = np.arange(-3.0, 4.0)
        if learnable:
            logits = 0.90 * latent[:, None] * centers[None, :] - (
                0.25 * np.square(centers)[None, :]
            )
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            ordered_labels = (
                draws[:, None] > np.cumsum(probabilities, axis=1)
            ).sum(axis=1)
        else:
            ordered_labels = rng.integers(
                0,
                len(ORDERED_CLASS_NAMES),
                row_count,
            )
        labels = ordered_labels_to_threshold_labels(ordered_labels, 0)
    else:
        raise ValueError(f"unsupported synthetic model family: {model_family}")
    dataset = pd.DataFrame(matrix, columns=FEATURE_NAMES)
    dataset.insert(0, "feature_date", pd.to_datetime(date_values))
    dataset.insert(0, "issuer_key", issuer_values)
    dataset.insert(0, "symbol", issuer_values)
    dataset["feature_timestamp"] = dataset["feature_date"]
    dataset["entry_timestamp"] = dataset["feature_date"] + pd.Timedelta(days=3)
    dataset["history_start"] = pd.Timestamp("2011-01-03")
    dataset["history_bars_before"] = 300
    dataset["max_exit_date"] = dataset["feature_date"] + pd.Timedelta(days=358)
    for horizon in HORIZONS:
        dataset[f"exit_timestamp_h{horizon}"] = (
            dataset["feature_date"]
            + pd.to_timedelta(round(horizon * 365.2425 / 252), unit="D")
        )
        if model_family == INDEPENDENT_MODEL_FAMILY:
            synthetic_return = np.where(
                labels == 0,
                -0.20,
                np.where(labels == 2, 0.20, 0.0),
            )
        else:
            a, b, c = (
                threshold / 100.0 + 0.003
                for threshold in THRESHOLD_GRIDS[horizon]
            )
            bin_returns = np.asarray(
                [
                    -c - 0.02,
                    -(c + b) / 2.0,
                    -(b + a) / 2.0,
                    0.0,
                    (a + b) / 2.0,
                    (b + c) / 2.0,
                    c + 0.02,
                ],
                dtype=float,
            )
            synthetic_return = bin_returns[ordered_labels]
        dataset[f"gross_return_h{horizon}"] = synthetic_return
        dataset[f"long_net_return_h{horizon}"] = synthetic_return - 0.003
        dataset[f"material_net_return_h{horizon}"] = np.sign(
            synthetic_return
        ) * np.maximum(np.abs(synthetic_return) - 0.003, 0)
        dataset[ordered_label_column(horizon)] = ordered_labels
        for threshold_index, threshold in enumerate(THRESHOLD_GRIDS[horizon]):
            dataset[label_column(horizon, threshold)] = (
                labels
                if model_family == INDEPENDENT_MODEL_FAMILY
                else ordered_labels_to_threshold_labels(
                    ordered_labels,
                    threshold_index,
                )
            )
    return dataset.sort_values(
        ["feature_date", "issuer_key"]
    ).reset_index(drop=True)


def build_dataset_binding(
    dataset: pd.DataFrame,
    dataset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = copy.deepcopy(dataset_manifest or {})
    summary = dataset_content_summary(dataset)
    requested = int(
        manifest["provider_requested_issuer_count"]
        if "provider_requested_issuer_count" in manifest
        else summary["issuer_count"]
    )
    successful = int(
        manifest["provider_successful_issuer_count"]
        if "provider_successful_issuer_count" in manifest
        else summary["issuer_count"]
    )
    coverage = successful / requested if requested else 0.0
    unavailable = dict(manifest.get("provider_unavailable_symbols") or {})
    binding = {
        "dataset_content_hash": dataset_content_hash(dataset),
        "dataset_manifest_hash": canonical_hash(manifest),
        "panel_manifest_source_hash": manifest.get("panel_manifest_sha256"),
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "publish_transform": dict(PUBLISH_TRANSFORM),
        "code_hash": probability_code_hash(),
        "dependency_versions": dependency_versions(),
        "summary": summary,
        "provider": {
            "requested_issuer_count": requested,
            "successful_issuer_count": successful,
            "success_coverage": coverage,
            "unavailable_count": len(unavailable),
            "unavailable_symbols": unavailable,
            "requested_symbols_sha256": manifest.get(
                "provider_requested_symbols_sha256"
            ),
            "successful_symbols_sha256": manifest.get(
                "provider_successful_symbols_sha256"
            ),
        },
    }
    binding["binding_hash"] = canonical_hash(binding)
    return binding


def checkpoint_key(
    binding: dict[str, Any],
    *,
    horizon: int,
    thresholds: list[int],
    bootstrap_repetitions: int,
    seed: int,
    c_value: float,
    model_family: str = INDEPENDENT_MODEL_FAMILY,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
    refit_bootstrap_repetitions: int = 0,
) -> str:
    return canonical_hash(
        {
            "dataset_binding_hash": binding["binding_hash"],
            "horizon": horizon,
            "thresholds": sorted(thresholds),
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "code_hash": probability_code_hash(),
            "model_version": MODEL_VERSION,
            "model_family": model_family,
            "calibration_version": (
                VECTOR_CALIBRATION_VERSION
                if model_family == ORDERED_MODEL_FAMILY
                else CALIBRATION_VERSION
            ),
            "ordered_model_version": (
                ORDERED_MODEL_VERSION
                if model_family == ORDERED_MODEL_FAMILY
                else None
            ),
            "publish_transform": (
                ORDERED_PUBLISH_TRANSFORM
                if model_family == ORDERED_MODEL_FAMILY
                else PUBLISH_TRANSFORM
            ),
            "vector_penalty_grid": (
                list(float(value) for value in penalty_grid)
                if model_family == ORDERED_MODEL_FAMILY
                else None
            ),
            "c_value": c_value,
            "bootstrap_repetitions": bootstrap_repetitions,
            "refit_bootstrap_repetitions": int(
                refit_bootstrap_repetitions
            ),
            "seed": seed,
            "fold": {
                "minimum_train_years": 5,
                "calibration_months": 12,
                "test_months": 12,
                "embargo_days": 7,
                "longest_horizon_purge_sessions": 252,
            },
            "acceptance_gates": STRICT_ACCEPTANCE_GATES,
            "grid_monotonicity_gates": GRID_MONOTONICITY_GATES,
        }
    )


def _new_validation(now: str, binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "created_at": now,
        "default_history_start": PROBABILITY_HISTORY_START,
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "code_hash": probability_code_hash(),
        "dependency_versions": dependency_versions(),
        "dataset_binding": binding,
        "status": "running",
        "accepted_model_count": 0,
        "tested_model_count": 0,
        "reasons": [],
        "acceptance_gates": STRICT_ACCEPTANCE_GATES,
        "grid_monotonicity_gates": GRID_MONOTONICITY_GATES,
        "publish_transform": dict(PUBLISH_TRANSFORM),
        "models": {},
        "bootstrap_method": (
            "fixed OOS predictions; two-way issuer clusters and calendar "
            "three-month blocks; no full model-refit bootstrap"
        ),
    }


def _load_or_new_validation(
    path: Path,
    *,
    now: str,
    resume: bool,
    binding: dict[str, Any],
) -> dict[str, Any]:
    if resume and path.exists():
        value = load_json(path, required=True, expected_type=dict)
        if (
            value.get("schema") == VALIDATION_SCHEMA
            and value.get("schema_version") == VALIDATION_SCHEMA_VERSION
            and value.get("feature_schema_hash") == feature_schema_hash()
            and value.get("code_hash") == probability_code_hash()
            and value.get("dependency_versions") == dependency_versions()
            and value.get("dataset_binding") == binding
        ):
            return value
    return _new_validation(now, binding)


def _final_training_baseline(
    dataset: pd.DataFrame,
    *,
    horizon: int,
    threshold_pct: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    target = label_column(horizon, threshold_pct)
    dataset = dataset.loc[dataset[target].notna()].copy()
    dates = pd.to_datetime(dataset["feature_date"])
    max_exit = pd.to_datetime(dataset["max_exit_date"])
    calibration_end = dates.max() + pd.Timedelta(days=1)
    calibration_start = calibration_end - pd.DateOffset(years=1)
    training = dataset.loc[
        (dates < calibration_start)
        & (max_exit < calibration_start - pd.Timedelta(days=7))
    ]
    labels = training[target].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=3).astype(float)
    rates = counts / counts.sum()
    acceptance = report.get("acceptance") or {}
    return {
        "model_key": model_key(horizon, threshold_pct),
        "horizon_sessions": horizon,
        "threshold_pct": threshold_pct,
        "rates": {
            name: float(rates[index])
            for index, name in enumerate(("down", "middle", "up"))
        },
        "fit_scope": "final release training segment only",
        "training_count": int(len(training)),
        "training_class_counts": {
            name: int(counts[index])
            for index, name in enumerate(("down", "middle", "up"))
        },
        "sample_size": report.get("aggregate", {}).get("count"),
        "event_counts_calibration_test": {
            name: int(
                (
                    report.get("event_counts_calibration_test_unique") or {}
                ).get(name, 0)
            )
            for name in ("down", "middle", "up")
        },
        "validation": {
            "accepted": bool(acceptance.get("accepted")),
            "reasons": list(acceptance.get("reasons") or []),
            "brier_skill": (report.get("aggregate") or {}).get("brier_skill"),
            "log_loss_improvement": (report.get("aggregate") or {}).get(
                "log_loss_improvement"
            ),
            "classwise_ece": (report.get("aggregate") or {}).get(
                "classwise_ece"
            ),
            "fold_count": report.get("fold_count"),
        },
    }


def _model_pairs(
    horizon: int | None = None,
    threshold: int | None = None,
) -> list[tuple[int, int]]:
    pairs = []
    for selected_horizon in HORIZONS:
        if horizon is not None and selected_horizon != horizon:
            continue
        for selected_threshold in THRESHOLD_GRIDS[selected_horizon]:
            if threshold is not None and selected_threshold != threshold:
                continue
            pairs.append((selected_horizon, selected_threshold))
    if not pairs:
        raise ValueError("requested horizon/threshold is not in the production grid")
    return pairs


def _failed_model_report(
    horizon: int,
    threshold_pct: int,
    reason: str,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "model_key": model_key(horizon, threshold_pct),
        "horizon_sessions": horizon,
        "threshold_pct": threshold_pct,
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "publish_transform": dict(PUBLISH_TRANSFORM),
        "publish_transform_identity_verified": False,
        "history_years": 0.0,
        "fold_count": 0,
        "full_test_fold_count": 0,
        "min_usable_train_years": 0.0,
        "issuer_count": 0,
        "forecast_date_count": 0,
        "min_train_class_count": 0,
        "min_calibration_class_count": 0,
        "min_test_class_count": 0,
        "inference_coverage": 0.0,
        "event_counts_calibration_test_unique": {
            "down": 0,
            "middle": 0,
            "up": 0,
        },
        "aggregate": {},
        "regime": {"groups": []},
        "bootstrap": {
            "repetitions": 0,
            "seed": seed,
            "brier_skill_ci95": [None, None],
        },
        "folds": [],
        "acceptance": {
            "accepted": False,
            "reasons": [reason],
            "gates": STRICT_ACCEPTANCE_GATES,
        },
    }


def train_dataset(
    dataset: pd.DataFrame,
    *,
    validation_path: Path = VALIDATION_PATH,
    models_path: Path = MODELS_PATH,
    bootstrap_repetitions: int = RELEASE_BOOTSTRAP_REPS,
    release: bool,
    resume: bool = True,
    horizon: int | None = None,
    threshold: int | None = None,
    seed: int = DEFAULT_SEED,
    trained_at: str | None = None,
    dataset_manifest: dict[str, Any] | None = None,
    c_value: float = 0.1,
    model_family: str = INDEPENDENT_MODEL_FAMILY,
    refit_bootstrap_repetitions: int = 0,
    dev_refit_override: bool = False,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
    experiment_validation_path: Path | None = None,
    experiment_models_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if model_family == ORDERED_MODEL_FAMILY:
        selected_experiment_validation = (
            experiment_validation_path
            or (
                ORDERED_VALIDATION_PATH
                if validation_path == VALIDATION_PATH
                else validation_path.parent
                / "probability_experiments"
                / "ordered-vector-v1_validation.json"
            )
        )
        selected_experiment_models = (
            experiment_models_path
            or (
                ORDERED_MODELS_EXPERIMENT_PATH
                if models_path == MODELS_PATH
                else models_path.parent
                / "probability_experiments"
                / "ordered-vector-v1_models.json"
            )
        )
        return _train_ordered_dataset(
            dataset,
            validation_path=validation_path,
            models_path=models_path,
            experiment_validation_path=selected_experiment_validation,
            experiment_models_path=selected_experiment_models,
            bootstrap_repetitions=bootstrap_repetitions,
            refit_bootstrap_repetitions=refit_bootstrap_repetitions,
            release=release,
            resume=resume,
            horizon=horizon,
            threshold=threshold,
            seed=seed,
            trained_at=trained_at,
            dataset_manifest=dataset_manifest,
            c_value=c_value,
            dev_refit_override=dev_refit_override,
            penalty_grid=penalty_grid,
        )
    if model_family != INDEPENDENT_MODEL_FAMILY:
        raise ValueError(f"unsupported probability model family: {model_family}")
    if dev_refit_override or refit_bootstrap_repetitions:
        raise ValueError(
            "full-refit bootstrap options apply only to ordered-vector-v1"
        )
    if release and bootstrap_repetitions < RELEASE_BOOTSTRAP_REPS:
        raise ValueError(
            f"release training requires at least {RELEASE_BOOTSTRAP_REPS} "
            "bootstrap repetitions"
        )
    now = trained_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    binding = build_dataset_binding(dataset, dataset_manifest)
    validation = _load_or_new_validation(
        validation_path,
        now=now,
        resume=resume,
        binding=binding,
    )
    selected_pairs = _model_pairs(horizon, threshold)
    pairs_by_horizon: dict[int, list[int]] = {}
    for selected_horizon, selected_threshold in selected_pairs:
        pairs_by_horizon.setdefault(selected_horizon, []).append(selected_threshold)
    for selected_horizon, selected_thresholds in sorted(pairs_by_horizon.items()):
        grid_checkpoint_key = checkpoint_key(
            binding,
            horizon=selected_horizon,
            thresholds=selected_thresholds,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
            c_value=c_value,
        )
        existing_reports = [
            validation["models"].get(model_key(selected_horizon, value))
            for value in selected_thresholds
        ]
        if (
            resume
            and all(isinstance(report, dict) for report in existing_reports)
            and all(
                report.get("grid_checkpoint_key") == grid_checkpoint_key
                for report in existing_reports
            )
        ):
            continue
        reports_by_threshold: dict[int, dict[str, Any]] = {}
        for selected_threshold in sorted(selected_thresholds):
            target = label_column(selected_horizon, selected_threshold)
            model_dataset = dataset.loc[
                dataset[target].notna()
                & dataset[f"exit_timestamp_h{selected_horizon}"].notna()
            ].copy()
            try:
                folds = make_purged_expanding_folds(
                    model_dataset,
                    minimum_folds=5,
                )
                report = train_walk_forward_model(
                    model_dataset,
                    folds,
                    horizon=selected_horizon,
                    threshold_pct=selected_threshold,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=seed,
                    c_value=c_value,
                    include_oos=True,
                )
            except Exception as exc:
                report = _failed_model_report(
                    selected_horizon,
                    selected_threshold,
                    f"validation execution failed closed: {str(exc)[:500]}",
                    seed=seed,
                )
            provider = binding["provider"]
            report.update(
                {
                    "provider_success_coverage": provider["success_coverage"],
                    "provider_requested_issuer_count": provider[
                        "requested_issuer_count"
                    ],
                    "provider_successful_issuer_count": provider[
                        "successful_issuer_count"
                    ],
                    "provider_unavailable_symbols": provider[
                        "unavailable_symbols"
                    ],
                    "fixed_oos_bootstrap_required_repetitions": (
                        bootstrap_repetitions if release else 0
                    ),
                }
            )
            previous_reasons = list(
                (report.get("acceptance") or {}).get("reasons") or []
            )
            report["acceptance"] = evaluate_acceptance(report)
            report["acceptance"]["reasons"] = list(
                dict.fromkeys(
                    [*previous_reasons, *report["acceptance"]["reasons"]]
                )
            )
            report["acceptance"]["accepted"] = not report["acceptance"]["reasons"]
            reports_by_threshold[selected_threshold] = report
        apply_grid_release_gate(reports_by_threshold)
        for selected_threshold, report in reports_by_threshold.items():
            report.pop("_oos", None)
            report["dataset_binding_hash"] = binding["binding_hash"]
            report["grid_checkpoint_key"] = grid_checkpoint_key
            report["hyperparameters"] = {
                "c_value": c_value,
                "bootstrap_repetitions": bootstrap_repetitions,
                "seed": seed,
                "publish_transform": dict(PUBLISH_TRANSFORM),
            }
            validation["models"][
                model_key(selected_horizon, selected_threshold)
            ] = report
        validation["tested_model_count"] = len(validation["models"])
        validation["accepted_model_count"] = sum(
            bool((item.get("acceptance") or {}).get("accepted"))
            for item in validation["models"].values()
        )
        atomic_write_json(validation_path, validation)

    validation["tested_model_count"] = len(validation["models"])
    validation["accepted_model_count"] = sum(
        bool((item.get("acceptance") or {}).get("accepted"))
        for item in validation["models"].values()
    )
    validation["status"] = (
        "accepted_models_available"
        if validation["accepted_model_count"]
        else "no_model_passed"
    )
    validation["reasons"] = (
        []
        if validation["accepted_model_count"]
        else ["No horizon/threshold model passed every strict release gate."]
    )
    atomic_write_json(validation_path, validation)
    if not release:
        return validation, None

    artifact = empty_probability_artifact(
        created_at=now,
        reason="No horizon/threshold model passed every strict release gate.",
    )
    artifact["models"] = {}
    artifact["baselines"] = {}
    artifact["dataset_binding"] = binding
    release_failures = []
    for selected_horizon, selected_threshold in _model_pairs(horizon, threshold):
        key = model_key(selected_horizon, selected_threshold)
        report = validation["models"].get(key)
        if not isinstance(report, dict):
            continue
        try:
            artifact["baselines"][key] = _final_training_baseline(
                dataset,
                horizon=selected_horizon,
                threshold_pct=selected_threshold,
                report=report,
            )
            artifact["baselines"][key]["dataset_binding_hash"] = binding[
                "binding_hash"
            ]
            artifact["baselines"][key]["checkpoint_key"] = report.get(
                "grid_checkpoint_key"
            )
        except Exception as exc:
            release_failures.append(
                f"{key}: baseline fit failed closed: {str(exc)[:300]}"
            )
        if not (report.get("acceptance") or {}).get("accepted"):
            continue
        try:
            artifact["models"][key] = fit_release_model(
                dataset.loc[
                    dataset[label_column(selected_horizon, selected_threshold)].notna()
                ].copy(),
                report,
                horizon=selected_horizon,
                threshold_pct=selected_threshold,
                trained_at=now,
                seed=seed,
                c_value=c_value,
            )
            artifact["models"][key].update(
                {
                    "dataset_binding_hash": binding["binding_hash"],
                    "checkpoint_key": report.get("grid_checkpoint_key"),
                    "provider_coverage": binding["provider"],
                }
            )
            artifact["models"][key].pop("model_hash", None)
            artifact["models"][key]["model_hash"] = canonical_hash(
                artifact["models"][key]
            )
        except Exception as exc:
            reason = f"final release fit failed: {str(exc)[:300]}"
            report["acceptance"]["accepted"] = False
            report["acceptance"]["reasons"].append(reason)
            artifact["baselines"][key]["validation"]["accepted"] = False
            artifact["baselines"][key]["validation"]["reasons"].append(reason)
            release_failures.append(f"{key}: {reason}")
    artifact["accepted_model_keys"] = sorted(artifact["models"])
    latest_feature_date = pd.to_datetime(
        dataset.get("feature_date", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).max()
    artifact["training_cutoff"] = (
        None
        if pd.isna(latest_feature_date)
        else pd.Timestamp(latest_feature_date).date().isoformat()
    )
    if artifact["accepted_model_keys"]:
        artifact["production_status"] = (
            "accepted_full_grid"
            if len(artifact["accepted_model_keys"]) == 12
            else "accepted_partial_grid"
        )
        artifact["production_reasons"] = release_failures
    else:
        artifact["production_status"] = "withheld"
        artifact["production_reasons"] = [
            "No horizon/threshold model passed every strict release gate.",
            *release_failures,
        ]
    artifact = finalize_artifact(artifact)
    validate_probability_artifact(artifact)
    validation["accepted_model_count"] = len(artifact["accepted_model_keys"])
    validation["status"] = (
        artifact["production_status"]
        if artifact["accepted_model_keys"]
        else "no_model_passed"
    )
    if release_failures:
        validation["reasons"].extend(release_failures)
    atomic_write_json(validation_path, validation)
    atomic_write_json(models_path, artifact)
    return validation, artifact


def _new_ordered_validation(
    now: str,
    binding: dict[str, Any],
    *,
    bootstrap_repetitions: int,
    refit_bootstrap_repetitions: int,
    dev_refit_override: bool,
    penalty_grid: Iterable[float],
) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "experiment_version": "ordered-vector-v1",
        "model_family": ORDERED_MODEL_FAMILY,
        "created_at": now,
        "default_history_start": PROBABILITY_HISTORY_START,
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "code_hash": probability_code_hash(),
        "dependency_versions": dependency_versions(),
        "dataset_binding": binding,
        "status": "running",
        "accepted_model_count": 0,
        "tested_model_count": 0,
        "tested_threshold_count": 0,
        "reasons": [],
        "acceptance_gates": STRICT_ACCEPTANCE_GATES,
        "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
        "model_version": ORDERED_MODEL_VERSION,
        "calibration_version": VECTOR_CALIBRATION_VERSION,
        "vector_penalty_grid": list(float(value) for value in penalty_grid),
        "fixed_oos_bootstrap": {
            "requested_repetitions": int(bootstrap_repetitions),
            "release_minimum": RELEASE_BOOTSTRAP_REPS,
            "method": (
                "fixed OOS predictions; two-way issuer clusters and calendar "
                "quarter blocks; requested/attempted/completed/skipped recorded; "
                "deterministic retries capped at 3x requested"
            ),
        },
        "full_refit_bootstrap": {
            "requested_repetitions": int(refit_bootstrap_repetitions),
            "preregistered_release_minimum": (
                RELEASE_REFIT_BOOTSTRAP_REPS
            ),
            "dev_override": bool(dev_refit_override),
            "method": (
                "two-way issuer/calendar-quarter resampling with complete "
                "preprocessor, seven-class model, and vector calibrator refit"
            ),
        },
        "models": {},
        "comparison_experiment": (
            "data/probability_experiments/independent-threshold-v1_summary.json"
        ),
    }


def _load_or_new_ordered_validation(
    path: Path,
    *,
    now: str,
    resume: bool,
    binding: dict[str, Any],
    bootstrap_repetitions: int,
    refit_bootstrap_repetitions: int,
    dev_refit_override: bool,
    penalty_grid: Iterable[float],
) -> dict[str, Any]:
    expected = _new_ordered_validation(
        now,
        binding,
        bootstrap_repetitions=bootstrap_repetitions,
        refit_bootstrap_repetitions=refit_bootstrap_repetitions,
        dev_refit_override=dev_refit_override,
        penalty_grid=penalty_grid,
    )
    if resume and path.exists():
        value = load_json(path, required=True, expected_type=dict)
        fields = (
            "schema",
            "schema_version",
            "experiment_version",
            "model_family",
            "feature_version",
            "feature_schema_hash",
            "code_hash",
            "dependency_versions",
            "dataset_binding",
            "publish_transform",
            "model_version",
            "calibration_version",
            "vector_penalty_grid",
            "fixed_oos_bootstrap",
            "full_refit_bootstrap",
        )
        if all(value.get(field) == expected.get(field) for field in fields):
            return value
    return expected


def _failed_ordered_report(
    horizon: int,
    reason: str,
    *,
    bootstrap_repetitions: int,
    refit_bootstrap_repetitions: int,
    dev_refit_override: bool,
) -> dict[str, Any]:
    return {
        "model_family": ORDERED_MODEL_FAMILY,
        "model_key": ordered_model_key(horizon),
        "horizon_sessions": int(horizon),
        "thresholds_pct": list(THRESHOLD_GRIDS[horizon]),
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "model_version": ORDERED_MODEL_VERSION,
        "calibration_version": VECTOR_CALIBRATION_VERSION,
        "penalty_grid": list(VECTOR_PENALTY_GRID),
        "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
        "exact_monotonicity": {
            "passed": False,
            "reason": reason,
            "tolerance": float(np.finfo(np.float64).eps),
        },
        "threshold_validation": {},
        "fixed_oos_bootstrap": {
            "repetitions": int(bootstrap_repetitions),
        },
        "full_refit_bootstrap": {
            "requested_repetitions": int(refit_bootstrap_repetitions),
            "attempted_repetitions": 0,
            "completed_repetitions": 0,
            "skipped_repetitions": int(refit_bootstrap_repetitions),
            "complete": False,
            "failures": [{"reason": reason}],
        },
        "acceptance": {
            "accepted": False,
            "reasons": [reason],
            "all_three_thresholds_required": True,
            "accepted_threshold_count": 0,
            "tested_threshold_count": 0,
            "production_release_eligible": False,
            "refit": {
                "preregistered_release_minimum": (
                    RELEASE_REFIT_BOOTSTRAP_REPS
                ),
                "dev_override": bool(dev_refit_override),
                "requested_repetitions": int(
                    refit_bootstrap_repetitions
                ),
                "completed_repetitions": 0,
                "acceptance_satisfied": False,
                "production_release_eligible": False,
            },
        },
    }


def _refresh_ordered_acceptance(
    report: dict[str, Any],
    binding: dict[str, Any],
) -> None:
    provider = binding["provider"]
    reasons = []
    accepted_thresholds = 0
    thresholds = report.get("threshold_validation") or {}
    for threshold, threshold_report in thresholds.items():
        threshold_report.update(
            {
                "provider_success_coverage": provider[
                    "success_coverage"
                ],
                "provider_requested_issuer_count": provider[
                    "requested_issuer_count"
                ],
                "provider_successful_issuer_count": provider[
                    "successful_issuer_count"
                ],
                "provider_unavailable_symbols": provider[
                    "unavailable_symbols"
                ],
            }
        )
        previous = list(
            (threshold_report.get("acceptance") or {}).get("reasons")
            or []
        )
        threshold_report["acceptance"] = evaluate_acceptance(
            threshold_report
        )
        threshold_report["acceptance"]["reasons"] = list(
            dict.fromkeys(
                [
                    *previous,
                    *threshold_report["acceptance"]["reasons"],
                ]
            )
        )
        threshold_report["acceptance"]["accepted"] = not (
            threshold_report["acceptance"]["reasons"]
        )
        if threshold_report["acceptance"]["accepted"]:
            accepted_thresholds += 1
        reasons.extend(
            f"x{threshold}: {reason}"
            for reason in threshold_report["acceptance"]["reasons"]
        )
    exact = report.get("exact_monotonicity") or {}
    if exact.get("passed") is not True:
        reasons.append("exact ordered tail-sum monotonicity assertion failed")
    refit = (report.get("acceptance") or {}).get("refit") or {}
    if not refit.get("acceptance_satisfied"):
        reasons.append(
            "full model+calibrator refit bootstrap incomplete: "
            f"{refit.get('completed_repetitions', 0)}/"
            f"{refit.get('acceptance_required_repetitions', RELEASE_REFIT_BOOTSTRAP_REPS)}"
        )
    report["acceptance"].update(
        {
            "accepted": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
            "accepted_threshold_count": accepted_thresholds,
            "tested_threshold_count": len(thresholds),
            "production_release_eligible": bool(
                not reasons
                and refit.get("production_release_eligible")
            ),
        }
    )


def _ordered_final_baselines(
    dataset: pd.DataFrame,
    report: dict[str, Any],
    *,
    horizon: int,
) -> dict[str, dict[str, Any]]:
    target = ordered_label_column(horizon)
    eligible = dataset.loc[dataset[target].notna()].copy()
    dates = pd.to_datetime(eligible["feature_date"])
    max_exit = pd.to_datetime(eligible["max_exit_date"])
    calibration_end = dates.max() + pd.Timedelta(days=1)
    calibration_start = calibration_end - pd.DateOffset(years=1)
    training = eligible.loc[
        (dates < calibration_start)
        & (max_exit < calibration_start - pd.Timedelta(days=7))
    ]
    ordered_labels = training[target].to_numpy(dtype=int)
    output = {}
    for threshold_index, threshold in enumerate(
        THRESHOLD_GRIDS[horizon]
    ):
        labels = ordered_labels_to_threshold_labels(
            ordered_labels,
            threshold_index,
        )
        counts = np.bincount(labels, minlength=3).astype(float)
        rates = counts / counts.sum()
        threshold_report = (
            report.get("threshold_validation") or {}
        ).get(str(threshold), {})
        acceptance = threshold_report.get("acceptance") or {}
        key = model_key(horizon, threshold)
        output[key] = {
            "model_family": ORDERED_MODEL_FAMILY,
            "source_model_key": ordered_model_key(horizon),
            "model_key": key,
            "horizon_sessions": int(horizon),
            "threshold_pct": int(threshold),
            "rates": {
                name: float(rates[index])
                for index, name in enumerate(("down", "middle", "up"))
            },
            "fit_scope": "final ordered release training segment only",
            "training_count": int(len(training)),
            "training_class_counts": {
                name: int(counts[index])
                for index, name in enumerate(("down", "middle", "up"))
            },
            "sample_size": (
                threshold_report.get("aggregate") or {}
            ).get("count"),
            "event_counts_calibration_test": (
                threshold_report.get(
                    "event_counts_calibration_test_unique"
                )
                or {name: 0 for name in ("down", "middle", "up")}
            ),
            "validation": {
                "accepted": bool(
                    acceptance.get("accepted")
                    and (report.get("acceptance") or {}).get("accepted")
                ),
                "reasons": list(
                    (report.get("acceptance") or {}).get("reasons")
                    or acceptance.get("reasons")
                    or []
                ),
                "brier_skill": (
                    threshold_report.get("aggregate") or {}
                ).get("brier_skill"),
                "log_loss_improvement": (
                    threshold_report.get("aggregate") or {}
                ).get("log_loss_improvement"),
                "classwise_ece": (
                    threshold_report.get("aggregate") or {}
                ).get("classwise_ece"),
                "fold_count": threshold_report.get("fold_count"),
            },
        }
    return output


def _publish_ordered_completion(
    validation: dict[str, Any],
    artifact: dict[str, Any],
    *,
    validation_path: Path,
    models_path: Path,
    experiment_validation_path: Path,
    experiment_models_path: Path,
) -> None:
    if (
        validation.get("model_family") != ORDERED_MODEL_FAMILY
        or artifact.get("model_family") != ORDERED_MODEL_FAMILY
        or validation.get("publish_transform")
        != ORDERED_PUBLISH_TRANSFORM
        or artifact.get("publish_transform")
        != ORDERED_PUBLISH_TRANSFORM
    ):
        raise ValueError("ordered canonical publication contract is inconsistent")
    accepted = bool(artifact.get("accepted_model_keys"))
    if accepted:
        if validation.get("status") not in (
            "accepted_full_grid",
            "accepted_partial_grid",
        ) or artifact.get("production_status") not in (
            "accepted_full_grid",
            "accepted_partial_grid",
        ):
            raise ValueError("accepted ordered publication status is inconsistent")
    elif (
        validation.get("status") != "no_model_passed"
        or artifact.get("production_status") != "withheld"
    ):
        raise ValueError("withheld ordered publication status is inconsistent")
    atomic_write_json(experiment_models_path, artifact)
    atomic_write_json(experiment_validation_path, validation)
    # The consumer artifact is replaced before validation advertises its status.
    atomic_write_json(models_path, artifact)
    atomic_write_json(validation_path, validation)


def _train_ordered_dataset(
    dataset: pd.DataFrame,
    *,
    validation_path: Path,
    models_path: Path,
    experiment_validation_path: Path,
    experiment_models_path: Path,
    bootstrap_repetitions: int,
    refit_bootstrap_repetitions: int,
    release: bool,
    resume: bool,
    horizon: int | None,
    threshold: int | None,
    seed: int,
    trained_at: str | None,
    dataset_manifest: dict[str, Any] | None,
    c_value: float,
    dev_refit_override: bool,
    penalty_grid: Iterable[float],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if threshold is not None:
        raise ValueError(
            "ordered-vector-v1 always validates the complete three-threshold horizon"
        )
    if release and dev_refit_override:
        raise ValueError("development refit override cannot create a release artifact")
    if release and bootstrap_repetitions < RELEASE_BOOTSTRAP_REPS:
        raise ValueError(
            f"ordered release requires at least {RELEASE_BOOTSTRAP_REPS} fixed-OOS "
            "bootstrap repetitions"
        )
    if (
        release
        and refit_bootstrap_repetitions < RELEASE_REFIT_BOOTSTRAP_REPS
    ):
        raise ValueError(
            "ordered release requires at least "
            f"{RELEASE_REFIT_BOOTSTRAP_REPS} full model+calibrator refits"
        )
    now = trained_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    selected_horizons = [
        value
        for value in HORIZONS
        if horizon is None or value == horizon
    ]
    if not selected_horizons:
        raise ValueError("requested horizon is not in the production grid")
    binding = build_dataset_binding(dataset, dataset_manifest)
    validation = _load_or_new_ordered_validation(
        experiment_validation_path,
        now=now,
        resume=resume,
        binding=binding,
        bootstrap_repetitions=bootstrap_repetitions,
        refit_bootstrap_repetitions=refit_bootstrap_repetitions,
        dev_refit_override=dev_refit_override,
        penalty_grid=penalty_grid,
    )
    for selected_horizon in selected_horizons:
        key = ordered_model_key(selected_horizon)
        selected_thresholds = list(THRESHOLD_GRIDS[selected_horizon])
        selected_checkpoint = checkpoint_key(
            binding,
            horizon=selected_horizon,
            thresholds=selected_thresholds,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
            c_value=c_value,
            model_family=ORDERED_MODEL_FAMILY,
            penalty_grid=penalty_grid,
            refit_bootstrap_repetitions=(
                refit_bootstrap_repetitions
            ),
        )
        existing = validation["models"].get(key)
        if (
            resume
            and isinstance(existing, dict)
            and existing.get("checkpoint_key") == selected_checkpoint
        ):
            continue
        target = ordered_label_column(selected_horizon)
        try:
            model_dataset = dataset.loc[
                dataset[target].notna()
                & dataset[f"exit_timestamp_h{selected_horizon}"].notna()
            ].copy()
            folds = make_purged_expanding_folds(
                model_dataset,
                minimum_folds=5,
            )
            report = train_ordered_walk_forward_horizon(
                model_dataset,
                folds,
                horizon=selected_horizon,
                bootstrap_repetitions=bootstrap_repetitions,
                refit_bootstrap_repetitions=(
                    refit_bootstrap_repetitions
                ),
                dev_refit_override=dev_refit_override,
                seed=seed,
                c_value=c_value,
                penalty_grid=penalty_grid,
                include_oos=True,
                fixed_oos_release_required=release,
            )
            _refresh_ordered_acceptance(report, binding)
        except Exception as exc:
            report = _failed_ordered_report(
                selected_horizon,
                f"ordered validation execution failed closed: {str(exc)[:500]}",
                bootstrap_repetitions=bootstrap_repetitions,
                refit_bootstrap_repetitions=(
                    refit_bootstrap_repetitions
                ),
                dev_refit_override=dev_refit_override,
            )
        report.pop("_oos", None)
        for threshold_report in (
            report.get("threshold_validation") or {}
        ).values():
            threshold_report.pop("_oos", None)
        report["dataset_binding_hash"] = binding["binding_hash"]
        report["checkpoint_key"] = selected_checkpoint
        report["hyperparameters"] = {
            "model_family": ORDERED_MODEL_FAMILY,
            "c_value": c_value,
            "fixed_oos_bootstrap_repetitions": (
                bootstrap_repetitions
            ),
            "full_refit_bootstrap_repetitions": (
                refit_bootstrap_repetitions
            ),
            "dev_refit_override": bool(dev_refit_override),
            "seed": seed,
            "vector_penalty_grid": list(
                float(value) for value in penalty_grid
            ),
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
        }
        validation["models"][key] = report
        validation["tested_model_count"] = len(validation["models"])
        validation["tested_threshold_count"] = sum(
            len(item.get("threshold_validation") or {})
            for item in validation["models"].values()
        )
        validation["accepted_model_count"] = sum(
            bool((item.get("acceptance") or {}).get("accepted"))
            for item in validation["models"].values()
        )
        atomic_write_json(experiment_validation_path, validation)
    validation["tested_model_count"] = len(validation["models"])
    validation["tested_threshold_count"] = sum(
        len(item.get("threshold_validation") or {})
        for item in validation["models"].values()
    )
    validation["accepted_model_count"] = sum(
        bool((item.get("acceptance") or {}).get("accepted"))
        for item in validation["models"].values()
    )
    validation["status"] = (
        "accepted_ordered_horizons_available"
        if validation["accepted_model_count"]
        else "no_ordered_horizon_passed"
    )
    validation["reasons"] = (
        []
        if validation["accepted_model_count"]
        else [
            "No ordered horizon passed all three unchanged threshold gates "
            "and the full-refit requirement."
        ]
    )
    atomic_write_json(experiment_validation_path, validation)
    if not release:
        return validation, None

    artifact = empty_probability_artifact(
        created_at=now,
        reason="No ordered horizon passed release acceptance.",
        model_family=ORDERED_MODEL_FAMILY,
    )
    artifact.update(
        {
            "engine_version": "probability-ordered-vector-v1",
            "model_family": ORDERED_MODEL_FAMILY,
            "models": {},
            "baselines": {},
            "dataset_binding": binding,
        }
    )
    release_failures = []
    for selected_horizon in selected_horizons:
        key = ordered_model_key(selected_horizon)
        report = validation["models"].get(key)
        if not isinstance(report, dict):
            continue
        try:
            baselines = _ordered_final_baselines(
                dataset,
                report,
                horizon=selected_horizon,
            )
            for baseline in baselines.values():
                baseline["dataset_binding_hash"] = binding[
                    "binding_hash"
                ]
                baseline["checkpoint_key"] = report.get(
                    "checkpoint_key"
                )
            artifact["baselines"].update(baselines)
        except Exception as exc:
            release_failures.append(
                f"{key}: baseline fit failed closed: {str(exc)[:300]}"
            )
        if not (
            (report.get("acceptance") or {}).get(
                "production_release_eligible"
            )
        ):
            continue
        try:
            model = fit_ordered_release_model(
                dataset.loc[
                    dataset[ordered_label_column(selected_horizon)].notna()
                ].copy(),
                report,
                horizon=selected_horizon,
                trained_at=now,
                seed=seed,
                c_value=c_value,
                penalty_grid=penalty_grid,
            )
            model.update(
                {
                    "dataset_binding_hash": binding["binding_hash"],
                    "checkpoint_key": report.get("checkpoint_key"),
                    "provider_coverage": binding["provider"],
                }
            )
            model.pop("model_hash", None)
            model["model_hash"] = canonical_hash(model)
            artifact["models"][key] = model
        except Exception as exc:
            release_failures.append(
                f"{key}: final ordered release fit failed: {str(exc)[:300]}"
            )
    artifact["accepted_model_keys"] = sorted(artifact["models"])
    latest_ordered_feature_date = pd.to_datetime(
        dataset.get("feature_date", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).max()
    artifact["training_cutoff"] = (
        None
        if pd.isna(latest_ordered_feature_date)
        else pd.Timestamp(latest_ordered_feature_date).date().isoformat()
    )
    if not artifact["accepted_model_keys"]:
        artifact["production_status"] = "withheld"
        artifact["production_reasons"] = list(
            dict.fromkeys(
                [
                    "No ordered horizon passed release acceptance.",
                    *release_failures,
                ]
            )
        )
        artifact = finalize_artifact(artifact)
        validate_probability_artifact(artifact)
        validation["accepted_model_count"] = 0
        validation["status"] = "no_model_passed"
        validation["production_status"] = "withheld"
        validation["reasons"] = list(
            dict.fromkeys([*validation["reasons"], *release_failures])
        )
        _publish_ordered_completion(
            validation,
            artifact,
            validation_path=validation_path,
            models_path=models_path,
            experiment_validation_path=experiment_validation_path,
            experiment_models_path=experiment_models_path,
        )
        return validation, artifact
    artifact["production_status"] = (
        "accepted_full_grid"
        if len(artifact["accepted_model_keys"]) == len(HORIZONS)
        else "accepted_partial_grid"
    )
    artifact["production_reasons"] = release_failures
    artifact = finalize_artifact(artifact)
    try:
        validate_probability_artifact(artifact)
    except Exception as exc:
        validation_failure = (
            "ordered production artifact validation failed closed: "
            f"{str(exc)[:300]}"
        )
        release_failures.append(validation_failure)
        withheld = empty_probability_artifact(
            created_at=now,
            reason=validation_failure,
            model_family=ORDERED_MODEL_FAMILY,
        )
        withheld.update(
            {
                "engine_version": "probability-ordered-vector-v1",
                "training_cutoff": artifact.get("training_cutoff"),
                "dataset_binding": binding,
                "baselines": artifact.get("baselines") or {},
                "production_status": "withheld",
                "production_reasons": list(dict.fromkeys(release_failures)),
            }
        )
        artifact = finalize_artifact(withheld)
        validate_probability_artifact(artifact)
        validation["accepted_model_count"] = 0
        validation["status"] = "no_model_passed"
        validation["production_status"] = "withheld"
        validation["reasons"] = list(dict.fromkeys(release_failures))
        _publish_ordered_completion(
            validation,
            artifact,
            validation_path=validation_path,
            models_path=models_path,
            experiment_validation_path=experiment_validation_path,
            experiment_models_path=experiment_models_path,
        )
        return validation, artifact
    validation["accepted_model_count"] = len(
        artifact["accepted_model_keys"]
    )
    validation["status"] = artifact["production_status"]
    validation["production_status"] = artifact["production_status"]
    validation["reasons"] = release_failures
    _publish_ordered_completion(
        validation,
        artifact,
        validation_path=validation_path,
        models_path=models_path,
        experiment_validation_path=experiment_validation_path,
        experiment_models_path=experiment_models_path,
    )
    return validation, artifact


def build_full_dataset(
    *,
    cache_dir: Path = DEFAULT_CACHE,
    start: str = DEFAULT_START,
    end: str | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligibility = load_training_universe()
    symbols = sorted(set(eligibility.symbols) | {"SPY"})
    download_probability_panel(
        symbols,
        start=start,
        end=end,
        cache_dir=cache_dir,
        resume=resume,
    )
    histories, panel_manifest = load_probability_panel(cache_dir)
    dataset, dataset_manifest = build_weekly_dataset(
        histories,
        issuer_keys=eligibility.issuer_keys,
        cache_dir=cache_dir,
        panel_manifest=panel_manifest,
    )
    dataset_manifest["eligibility_exclusions"] = eligibility.excluded
    requested = sorted(eligibility.symbols)
    successful = sorted(set(dataset.get("symbol", pd.Series(dtype=str)).astype(str)))
    provider_failures = dict(panel_manifest.get("failures") or {})
    provider_failures.update(dataset_manifest.get("failures") or {})
    unavailable = {
        symbol: provider_failures.get(
            symbol, "eligible symbol produced no usable weekly dataset rows"
        )
        for symbol in requested
        if symbol not in set(successful)
    }
    dataset_manifest.update(
        {
            "provider_requested_issuer_count": len(requested),
            "provider_successful_issuer_count": len(successful),
            "provider_success_coverage": (
                len(successful) / len(requested) if requested else 0.0
            ),
            "provider_requested_symbols_sha256": hashlib.sha256(
                "\n".join(requested).encode("utf-8")
            ).hexdigest(),
            "provider_successful_symbols_sha256": hashlib.sha256(
                "\n".join(successful).encode("utf-8")
            ).hexdigest(),
            "provider_unavailable_symbols": unavailable,
        }
    )
    dataset_manifest["dataset_content_hash"] = dataset_content_hash(dataset)
    dataset_manifest["dataset_content_summary"] = dataset_content_summary(dataset)
    dataset_manifest["dataset_cache_key"] = canonical_hash(
        {
            "dataset_content_hash": dataset_manifest["dataset_content_hash"],
            "dataset_schema_version": dataset_manifest["schema_version"],
            "storage_format": dataset_manifest["storage_format"],
            "panel_manifest_sha256": dataset_manifest.get(
                "panel_manifest_sha256"
            ),
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "code_hash": probability_code_hash(),
            "provider_requested_symbols_sha256": dataset_manifest[
                "provider_requested_symbols_sha256"
            ],
            "provider_successful_symbols_sha256": dataset_manifest[
                "provider_successful_symbols_sha256"
            ],
            "summary": dataset_manifest["dataset_content_summary"],
        }
    )
    atomic_write_json(Path(cache_dir) / "dataset_manifest.json", dataset_manifest)
    return dataset, dataset_manifest


def run_smoke(
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    results = {}
    for name, learnable in (("learnable", True), ("random", False)):
        dataset = make_synthetic_dataset(
            learnable=learnable,
            seed=seed,
        )
        folds = make_purged_expanding_folds(dataset, minimum_folds=5)
        report = train_walk_forward_model(
            dataset,
            folds,
            horizon=21,
            threshold_pct=3,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed,
        )
        results[name] = {
            "accepted": report["acceptance"]["accepted"],
            "reasons": report["acceptance"]["reasons"],
            "brier_skill": report["aggregate"]["brier_skill"],
            "log_loss_improvement": report["aggregate"][
                "log_loss_improvement"
            ],
            "fold_count": report["fold_count"],
        }
    if not results["learnable"]["accepted"]:
        raise RuntimeError(
            "learnable synthetic signal did not pass strict acceptance: "
            + "; ".join(results["learnable"]["reasons"])
        )
    if results["random"]["accepted"]:
        raise RuntimeError("random synthetic target incorrectly passed acceptance")
    return results


def run_ordered_smoke(
    *,
    bootstrap_repetitions: int = 50,
    refit_bootstrap_repetitions: int,
    dev_refit_override: bool,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if refit_bootstrap_repetitions <= 0 or not dev_refit_override:
        raise ValueError(
            "ordered smoke requires an explicit positive development refit "
            "count and --dev-refit-override"
        )
    results = {}
    for name, learnable in (("learnable", True), ("random", False)):
        dataset = make_synthetic_dataset(
            learnable=learnable,
            seed=seed,
            model_family=ORDERED_MODEL_FAMILY,
        )
        folds = make_purged_expanding_folds(
            dataset,
            minimum_folds=5,
        )
        report = train_ordered_walk_forward_horizon(
            dataset,
            folds,
            horizon=21,
            bootstrap_repetitions=bootstrap_repetitions,
            refit_bootstrap_repetitions=(
                refit_bootstrap_repetitions
            ),
            dev_refit_override=True,
            seed=seed,
        )
        first = report["threshold_validation"][
            str(THRESHOLD_GRIDS[21][0])
        ]
        results[name] = {
            "accepted": report["acceptance"]["accepted"],
            "production_release_eligible": report["acceptance"][
                "production_release_eligible"
            ],
            "reasons": report["acceptance"]["reasons"],
            "accepted_threshold_count": report["acceptance"][
                "accepted_threshold_count"
            ],
            "brier_skill": first["aggregate"]["brier_skill"],
            "log_loss_improvement": first["aggregate"][
                "log_loss_improvement"
            ],
            "fold_count": first["fold_count"],
            "exact_monotonicity": report["exact_monotonicity"][
                "passed"
            ],
            "refit_acceptance": report["acceptance"]["refit"],
        }
    if not results["learnable"]["accepted"]:
        raise RuntimeError(
            "learnable ordered synthetic signal did not pass development "
            "acceptance: "
            + "; ".join(results["learnable"]["reasons"])
        )
    if results["learnable"]["production_release_eligible"]:
        raise RuntimeError(
            "development refit override incorrectly became release eligible"
        )
    if results["random"]["accepted"]:
        raise RuntimeError(
            "random ordered synthetic target incorrectly passed acceptance"
        )
    return {
        "model_family": ORDERED_MODEL_FAMILY,
        "development_only": True,
        "fixed_oos_bootstrap_repetitions": int(
            bootstrap_repetitions
        ),
        "full_refit_bootstrap_repetitions": int(
            refit_bootstrap_repetitions
        ),
        "preregistered_release_refit_minimum": (
            RELEASE_REFIT_BOOTSTRAP_REPS
        ),
        **results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrated Stock Radar probability engine"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    smoke = subparsers.add_parser(
        "smoke", help="bounded learnable/random synthetic acceptance smoke"
    )
    smoke.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    smoke.add_argument("--seed", type=int, default=DEFAULT_SEED)
    smoke.add_argument(
        "--model-family",
        choices=(INDEPENDENT_MODEL_FAMILY, ORDERED_MODEL_FAMILY),
        default=INDEPENDENT_MODEL_FAMILY,
    )
    smoke.add_argument("--refit-bootstrap", type=int, default=2)
    smoke.add_argument("--dev-refit-override", action="store_true")

    build = subparsers.add_parser("build", help="download/restart panel and dataset")
    build.add_argument("--start", default=DEFAULT_START)
    build.add_argument("--end")
    build.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    build.add_argument("--no-resume", action="store_true")

    for mode in ("train", "validation-only", "full"):
        command = subparsers.add_parser(mode)
        command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
        command.add_argument(
            "--bootstrap",
            type=int,
            default=(
                DEFAULT_BOOTSTRAP_REPS
                if mode == "validation-only"
                else RELEASE_BOOTSTRAP_REPS
            ),
        )
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument(
            "--model-family",
            choices=(
                INDEPENDENT_MODEL_FAMILY,
                ORDERED_MODEL_FAMILY,
            ),
            default=INDEPENDENT_MODEL_FAMILY,
        )
        command.add_argument(
            "--refit-bootstrap",
            type=int,
            default=(
                0
                if mode == "validation-only"
                else RELEASE_REFIT_BOOTSTRAP_REPS
            ),
        )
        command.add_argument("--horizon", type=int, choices=HORIZONS)
        command.add_argument("--threshold", type=int)
        command.add_argument("--no-resume", action="store_true")
        if mode == "full":
            command.add_argument("--start", default=DEFAULT_START)
            command.add_argument("--end")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.mode == "smoke":
        result = (
            run_ordered_smoke(
                bootstrap_repetitions=args.bootstrap,
                refit_bootstrap_repetitions=args.refit_bootstrap,
                dev_refit_override=args.dev_refit_override,
                seed=args.seed,
            )
            if args.model_family == ORDERED_MODEL_FAMILY
            else run_smoke(
                bootstrap_repetitions=args.bootstrap,
                seed=args.seed,
            )
        )
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.mode == "build":
        dataset, manifest = build_full_dataset(
            cache_dir=args.cache_dir,
            start=args.start,
            end=args.end,
            resume=not args.no_resume,
        )
        print(
            f"Built {len(dataset):,} weekly rows for "
            f"{manifest.get('symbol_count', 0):,} symbols."
        )
        return
    if args.mode == "full":
        dataset, _manifest = build_full_dataset(
            cache_dir=args.cache_dir,
            start=args.start,
            end=args.end,
            resume=not args.no_resume,
        )
    else:
        dataset, _manifest = load_weekly_dataset(args.cache_dir)
    validation, artifact = train_dataset(
        dataset,
        bootstrap_repetitions=args.bootstrap,
        release=args.mode != "validation-only",
        resume=not args.no_resume,
        horizon=args.horizon,
        threshold=args.threshold,
        seed=args.seed,
        dataset_manifest=_manifest,
        model_family=args.model_family,
        refit_bootstrap_repetitions=(
            args.refit_bootstrap
            if args.model_family == ORDERED_MODEL_FAMILY
            else 0
        ),
    )
    print(
        f"Validated {validation['tested_model_count']} models; "
        f"accepted {validation['accepted_model_count']}."
    )
    if artifact is not None:
        print(
            f"Production artifact status: {artifact['production_status']} -> "
            f"{MODELS_PATH}"
        )


if __name__ == "__main__":
    main()
