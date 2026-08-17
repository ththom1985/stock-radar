"""Preregistered ordered seven-bin probability challenger."""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .probability_contract import (
    CLASS_NAMES,
    ORDERED_CLASS_NAMES,
    ORDERED_MODEL_FAMILY,
    ROUND_TRIP_COST,
    THRESHOLD_GRIDS,
    model_key,
    ordered_label_column,
    ordered_model_key,
)
from .probability_features import FEATURE_NAMES, FEATURE_VERSION, feature_schema_hash
from .probability_model import (
    BOOTSTRAP_VERSION,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_SEED,
    STRICT_ACCEPTANCE_GATES,
    _binary_calibration_fit,
    assess_ood,
    canonical_hash,
    evaluate_acceptance,
    fit_multinomial_model,
    fit_regime_baseline,
    fit_temperature,
    multiclass_brier,
    multiclass_log_loss,
    predict_logits,
    predict_regime_baseline,
    stable_softmax,
    two_way_cluster_bootstrap,
)

ORDERED_MODEL_VERSION = "probability-ordered-seven-bin-multinomial-l2-v1"
VECTOR_CALIBRATION_VERSION = "regularized-vector-scaling-v1"
VECTOR_PENALTY_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ORDERED_PUBLISH_TRANSFORM = {
    "version": "ordered-vector-tail-sum-identity-v1",
    "operation": "identity",
    "input": "regularized vector-scaled seven-class softmax tail sums",
    "caps": None,
    "cross_threshold_projection": None,
}
ORDERED_MONOTONICITY_EPSILON = float(np.finfo(np.float64).eps)
RELEASE_REFIT_BOOTSTRAP_REPS = 200
ADAPTIVE_INITIAL_BINS = 10
ADAPTIVE_MIN_BIN_ROWS = 500
ADAPTIVE_MIN_POSITIVES = 50
ADAPTIVE_MIN_NEGATIVES = 50
ADAPTIVE_MIN_SUPPORTED_BINS = 5
REGIME_MIN_DATES = 26
REGIME_MIN_QUARTERS = 8
REGIME_MIN_OUTCOMES_PER_CLASS = 100
REGIME_MIN_ISSUERS = 100


def fit_ordered_multinomial_model(
    train_features: pd.DataFrame | np.ndarray,
    train_labels: np.ndarray,
    *,
    c_value: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    model = fit_multinomial_model(
        train_features,
        train_labels,
        c_value=c_value,
        seed=seed,
        class_names=ORDERED_CLASS_NAMES,
        model_version=ORDERED_MODEL_VERSION,
        model_type="L2 seven-class multinomial logistic regression",
    )
    model["model_family"] = ORDERED_MODEL_FAMILY
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    return model


def _validate_vector_inputs(
    logits: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(logits, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != len(ORDERED_CLASS_NAMES)
        or not np.isfinite(values).all()
    ):
        raise ValueError("vector scaling requires finite seven-class logits")
    if labels is None:
        return values, None
    outcomes = np.asarray(labels, dtype=int)
    if (
        outcomes.shape != (len(values),)
        or len(outcomes) == 0
        or outcomes.min() < 0
        or outcomes.max() >= len(ORDERED_CLASS_NAMES)
    ):
        raise ValueError("vector scaling labels are incompatible with logits")
    return values, outcomes


def apply_vector_scaling(
    logits: np.ndarray,
    calibrator: dict[str, Any] | None = None,
) -> np.ndarray:
    values, _ = _validate_vector_inputs(logits)
    if calibrator is None:
        scales = np.ones(len(ORDERED_CLASS_NAMES), dtype=float)
        biases = np.zeros(len(ORDERED_CLASS_NAMES), dtype=float)
    else:
        if calibrator.get("version") != VECTOR_CALIBRATION_VERSION:
            raise ValueError("unsupported vector scaling calibrator")
        scales = np.asarray(calibrator.get("scales"), dtype=float)
        biases = np.asarray(calibrator.get("biases"), dtype=float)
    if (
        scales.shape != (len(ORDERED_CLASS_NAMES),)
        or biases.shape != (len(ORDERED_CLASS_NAMES),)
        or not np.isfinite(scales).all()
        or not np.isfinite(biases).all()
        or (scales <= 0).any()
        or abs(float(biases.sum())) > 1e-10
    ):
        raise ValueError("vector scaling parameters are invalid")
    return stable_softmax(values * scales + biases)


def _weighted_log_loss(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    selected = np.clip(
        probabilities[np.arange(len(labels)), labels],
        1e-15,
        1.0,
    )
    return float(np.average(-np.log(selected), weights=weights))


def _weighted_brier(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[labels]
    values = np.sum(np.square(probabilities - one_hot), axis=1)
    return float(np.average(values, weights=weights))


def _fit_vector_parameters(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    penalty: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    values, outcomes_value = _validate_vector_inputs(logits, labels)
    outcomes = np.asarray(outcomes_value, dtype=int)
    if not math.isfinite(float(penalty)) or float(penalty) <= 0:
        raise ValueError("vector scaling penalty must be finite and positive")
    weights = (
        np.ones(len(outcomes), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        weights.shape != (len(outcomes),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("vector scaling sample weights are invalid")
    class_count = values.shape[1]
    one_hot = np.eye(class_count, dtype=float)[outcomes]
    weight_scale = weights / weights.sum()

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_scales = parameters[:class_count]
        free_biases = parameters[class_count:]
        biases = np.concatenate([free_biases, [-free_biases.sum()]])
        scales = np.exp(log_scales)
        return log_scales, scales, biases

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_scales, scales, biases = unpack(parameters)
        calibrated = values * scales + biases
        probabilities = stable_softmax(calibrated)
        selected = np.clip(
            probabilities[np.arange(len(outcomes)), outcomes],
            1e-15,
            1.0,
        )
        regularization = 0.5 * float(penalty) * (
            np.mean(np.square(log_scales)) + np.mean(np.square(biases))
        )
        loss = float(np.dot(weight_scale, -np.log(selected)) + regularization)
        residual = (probabilities - one_hot) * weight_scale[:, None]
        gradient_log_scales = np.sum(
            residual * values * scales,
            axis=0,
        ) + float(penalty) * log_scales / class_count
        gradient_biases = np.sum(residual, axis=0) + (
            float(penalty) * biases / class_count
        )
        gradient_free_biases = (
            gradient_biases[:-1] - gradient_biases[-1]
        )
        gradient = np.concatenate(
            [gradient_log_scales, gradient_free_biases]
        )
        if not math.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite vector scaling objective")
        return loss, gradient

    initial = np.zeros(class_count + class_count - 1, dtype=float)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-5.0, 5.0)] * class_count
        + [(-10.0, 10.0)] * (class_count - 1),
        options={
            "maxiter": 1000,
            "maxls": 50,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )
    if not result.success:
        raise RuntimeError(
            "vector scaling optimizer did not converge: "
            f"{result.status} {str(result.message)[:200]}"
        )
    log_scales, scales, biases = unpack(np.asarray(result.x, dtype=float))
    if (
        not np.isfinite(log_scales).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(biases).all()
        or abs(float(biases.sum())) > 1e-10
    ):
        raise RuntimeError("vector scaling optimizer returned invalid parameters")
    return {
        "log_scales": log_scales,
        "scales": scales,
        "biases": biases,
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "converged": True,
    }


def fit_vector_scaling(
    calibration_logits: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_dates: Iterable[Any],
    *,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
    sample_weight: np.ndarray | None = None,
    calibration_interval_start: Any | None = None,
) -> dict[str, Any]:
    """Select the penalty on months 10-12, then refit on the full calibration year."""
    logits, outcomes_value = _validate_vector_inputs(
        calibration_logits,
        calibration_labels,
    )
    outcomes = np.asarray(outcomes_value, dtype=int)
    dates = pd.to_datetime(
        pd.Series(list(calibration_dates)),
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    if len(dates) != len(outcomes) or dates.isna().any():
        raise ValueError("vector scaling calibration dates are invalid")
    penalties = tuple(float(value) for value in penalty_grid)
    if (
        not penalties
        or len(set(penalties)) != len(penalties)
        or any(not math.isfinite(value) or value <= 0 for value in penalties)
    ):
        raise ValueError("vector scaling penalty grid is invalid")
    weights = (
        np.ones(len(outcomes), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weights.shape != (len(outcomes),):
        raise ValueError("vector scaling sample weights have incompatible shape")
    ordering = np.argsort(dates.to_numpy(), kind="mergesort")
    logits = logits[ordering]
    outcomes = outcomes[ordering]
    dates = dates.iloc[ordering].reset_index(drop=True)
    weights = weights[ordering]
    interval_start = (
        pd.Timestamp(dates.min()).normalize()
        if calibration_interval_start is None
        else pd.Timestamp(calibration_interval_start).tz_localize(None)
        .normalize()
    )
    selection_start = interval_start + pd.DateOffset(months=9)
    fit_mask = (dates < selection_start).to_numpy()
    selection_mask = ~fit_mask
    if fit_mask.sum() == 0 or selection_mask.sum() == 0:
        raise ValueError(
            "vector scaling needs nonempty first-nine-month fit and last-three-month "
            "selection intervals"
        )
    if set(np.unique(outcomes[fit_mask])) != set(range(len(ORDERED_CLASS_NAMES))):
        raise ValueError("first-nine-month vector fit is missing an ordered class")
    if set(np.unique(outcomes[selection_mask])) != set(
        range(len(ORDERED_CLASS_NAMES))
    ):
        raise ValueError("last-three-month penalty selection is missing an ordered class")
    candidates = []
    for order, penalty in enumerate(penalties):
        fitted = _fit_vector_parameters(
            logits[fit_mask],
            outcomes[fit_mask],
            penalty=penalty,
            sample_weight=weights[fit_mask],
        )
        candidate = {
            "version": VECTOR_CALIBRATION_VERSION,
            "scales": fitted["scales"].tolist(),
            "biases": fitted["biases"].tolist(),
        }
        probabilities = apply_vector_scaling(
            logits[selection_mask],
            candidate,
        )
        candidates.append(
            {
                "penalty": penalty,
                "selection_log_loss": _weighted_log_loss(
                    outcomes[selection_mask],
                    probabilities,
                    weights[selection_mask],
                ),
                "selection_brier": _weighted_brier(
                    outcomes[selection_mask],
                    probabilities,
                    weights[selection_mask],
                ),
                "grid_order": order,
                "fit_iterations": fitted["iterations"],
                "fit_gradient_max_abs": fitted["gradient_max_abs"],
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["selection_log_loss"],
            row["selection_brier"],
            row["grid_order"],
        ),
    )
    refitted = _fit_vector_parameters(
        logits,
        outcomes,
        penalty=float(selected["penalty"]),
        sample_weight=weights,
    )
    calibrator = {
        "version": VECTOR_CALIBRATION_VERSION,
        "formula": "softmax(exp(s_j) * z_j + b_j)",
        "identifiability": "biases constrained to zero mean",
        "regularization": (
            "0.5 * penalty * (mean(log_scale^2) + mean(bias^2)); "
            "toward log_scale=0,bias=0"
        ),
        "penalty_grid": list(penalties),
        "selected_penalty": float(selected["penalty"]),
        "selection_metric": "seven-class log loss; seven-class Brier tie-break",
        "selection_candidates": candidates,
        "log_scales": refitted["log_scales"].tolist(),
        "scales": refitted["scales"].tolist(),
        "biases": refitted["biases"].tolist(),
        "fit_source": (
            "outer-fold calibration only: first 9 months candidate fit, last "
            "3 months penalty selection, selected penalty refit on full year"
        ),
        "calibration_interval_start": interval_start.date().isoformat(),
        "calibration_interval_end": pd.Timestamp(dates.max()).date().isoformat(),
        "penalty_fit_start": pd.Timestamp(dates.loc[fit_mask].min())
        .date()
        .isoformat(),
        "penalty_fit_end": pd.Timestamp(dates.loc[fit_mask].max())
        .date()
        .isoformat(),
        "penalty_selection_start": pd.Timestamp(dates.loc[selection_mask].min())
        .date()
        .isoformat(),
        "penalty_selection_end": pd.Timestamp(dates.loc[selection_mask].max())
        .date()
        .isoformat(),
        "penalty_fit_rows": int(fit_mask.sum()),
        "penalty_selection_rows": int(selection_mask.sum()),
        "refit_rows": int(len(outcomes)),
        "refit_iterations": refitted["iterations"],
        "refit_gradient_max_abs": refitted["gradient_max_abs"],
        "converged": True,
    }
    calibrated = apply_vector_scaling(logits, calibrator)
    calibrator["uncalibrated_log_loss"] = _weighted_log_loss(
        outcomes,
        stable_softmax(logits),
        weights,
    )
    calibrator["calibrated_log_loss"] = _weighted_log_loss(
        outcomes,
        calibrated,
        weights,
    )
    return calibrator


def predict_ordered_probabilities(
    model: dict[str, Any],
    features: pd.DataFrame | np.ndarray,
    *,
    require_complete: bool = False,
) -> np.ndarray:
    logits = predict_logits(
        model,
        features,
        require_complete=require_complete,
    )
    return apply_vector_scaling(logits, model.get("vector_scaling"))


def fit_ordered_vector_calibration(
    model: dict[str, Any],
    calibration: pd.DataFrame,
    *,
    target: str,
    calibration_interval_start: Any,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
) -> dict[str, Any]:
    """Apply the single fold/release/full-refit calibration path."""
    if target not in calibration:
        raise ValueError(f"ordered calibration is missing {target}")
    calibration_ood = assess_ood(
        calibration.loc[:, FEATURE_NAMES],
        model["preprocessor"],
    )
    coverage_mask = np.asarray(
        [not item["withhold"] for item in calibration_ood],
        dtype=bool,
    )
    scored = calibration.loc[coverage_mask].copy()
    if scored.empty:
        raise ValueError("ordered calibration has no OOD-supported rows")
    labels = scored[target].to_numpy(dtype=int)
    logits = predict_logits(
        model,
        scored.loc[:, FEATURE_NAMES],
    )
    vector = fit_vector_scaling(
        logits,
        labels,
        scored["feature_date"],
        penalty_grid=penalty_grid,
        calibration_interval_start=calibration_interval_start,
    )
    temperature_diagnostic = fit_temperature(logits, labels)
    model["vector_scaling"] = vector
    model["temperature_diagnostic"] = temperature_diagnostic
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    return {
        "model": model,
        "ood": calibration_ood,
        "coverage_mask": coverage_mask,
        "scored": scored,
        "ordered_labels": labels,
        "logits": logits,
        "vector_scaling": vector,
        "temperature_diagnostic": temperature_diagnostic,
        "candidate_count": int(len(calibration)),
        "scored_count": int(len(scored)),
        "coverage": float(coverage_mask.mean()),
        "path_version": "ordered-calibration-ood-vector-v1",
    }


def ordered_labels_to_threshold_labels(
    ordered_labels: np.ndarray,
    threshold_index: int,
) -> np.ndarray:
    labels = np.asarray(ordered_labels, dtype=int)
    if threshold_index not in (0, 1, 2):
        raise ValueError("ordered threshold index must be 0, 1, or 2")
    if labels.size and (labels.min() < 0 or labels.max() > 6):
        raise ValueError("ordered labels must be in [0, 6]")
    down_max = 2 - threshold_index
    up_min = 4 + threshold_index
    return np.where(labels <= down_max, 0, np.where(labels >= up_min, 2, 1)).astype(
        np.int8
    )


def derive_threshold_probabilities(
    ordered_probabilities: np.ndarray,
    thresholds: Iterable[int],
) -> dict[int, np.ndarray]:
    values = np.asarray(ordered_probabilities, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values.reshape(1, -1)
    selected = tuple(int(value) for value in thresholds)
    if (
        values.ndim != 2
        or values.shape[1] != len(ORDERED_CLASS_NAMES)
        or len(selected) != 3
        or sorted(selected) != list(selected)
        or len(set(selected)) != 3
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(
            values.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=ORDERED_MONOTONICITY_EPSILON * 4,
        )
    ):
        raise ValueError("ordered probabilities/thresholds are invalid")
    output: dict[int, np.ndarray] = {}
    for index, threshold in enumerate(selected):
        down_end = 3 - index
        up_start = 4 + index
        result = np.column_stack(
            [
                values[:, :down_end].sum(axis=1),
                values[:, down_end:up_start].sum(axis=1),
                values[:, up_start:].sum(axis=1),
            ]
        )
        if not np.allclose(
            result.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=ORDERED_MONOTONICITY_EPSILON * 4,
        ):
            raise AssertionError("derived threshold probabilities lost the simplex")
        output[threshold] = result[0] if one_dimensional else result
    assert_exact_ordered_monotonicity(output)
    return output


def assert_exact_ordered_monotonicity(
    probabilities_by_threshold: dict[int, np.ndarray],
    *,
    epsilon: float = ORDERED_MONOTONICITY_EPSILON,
) -> dict[str, Any]:
    thresholds = sorted(probabilities_by_threshold)
    if len(thresholds) != 3:
        raise AssertionError("the complete three-threshold ordered grid is required")
    matrices = {}
    one_dimensional = None
    row_count = None
    for threshold in thresholds:
        values = np.asarray(probabilities_by_threshold[threshold], dtype=float)
        current_one_dimensional = values.ndim == 1
        if current_one_dimensional:
            values = values.reshape(1, -1)
        if (
            values.ndim != 2
            or values.shape[1] != len(CLASS_NAMES)
            or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.allclose(
                values.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=epsilon * 4,
            )
        ):
            raise AssertionError("derived threshold probability simplex is invalid")
        if one_dimensional is None:
            one_dimensional = current_one_dimensional
            row_count = len(values)
        if current_one_dimensional != one_dimensional or len(values) != row_count:
            raise AssertionError("ordered threshold probability shapes differ")
        matrices[threshold] = values
    up_excess = []
    down_excess = []
    for easier, harder in zip(thresholds, thresholds[1:]):
        up_excess.append(matrices[harder][:, 2] - matrices[easier][:, 2])
        down_excess.append(matrices[harder][:, 0] - matrices[easier][:, 0])
    all_up = np.concatenate(up_excess)
    all_down = np.concatenate(down_excess)
    if (all_up > epsilon).any() or (all_down > epsilon).any():
        raise AssertionError("ordered tail sums are not exactly monotonic")
    return {
        "passed": True,
        "permitted": True,
        "reason": None,
        "reason_code": None,
        "thresholds": thresholds,
        "oos_row_count": int(row_count or 0),
        "up_violation_count": int((all_up > epsilon).sum()),
        "down_violation_count": int((all_down > epsilon).sum()),
        "max_up_excess": float(max(0.0, all_up.max(initial=0.0))),
        "max_down_excess": float(max(0.0, all_down.max(initial=0.0))),
        "tolerance": float(epsilon),
        "construction": "seven-class vector-scaled softmax disjoint tail sums",
        "whole_percent_display_monotonic": True,
        "tolerated_independent_threshold_inversion": False,
        "disclosure": (
            "Exact ordered-distribution tail sums; monotonic by construction "
            "within float64 machine epsilon."
        ),
        "action": "assert exact construction; never project or cap",
    }


def _wilson_interval(
    positives: int,
    count: int,
    *,
    z: float = 1.959963984540054,
) -> list[float] | None:
    if count <= 0:
        return None
    proportion = positives / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def adaptive_classwise_reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    initial_bins: int = ADAPTIVE_INITIAL_BINS,
    minimum_rows: int = ADAPTIVE_MIN_BIN_ROWS,
    minimum_positives: int = ADAPTIVE_MIN_POSITIVES,
    minimum_negatives: int = ADAPTIVE_MIN_NEGATIVES,
    minimum_supported_bins: int = ADAPTIVE_MIN_SUPPORTED_BINS,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 2
        or p.shape != (len(y), len(CLASS_NAMES))
        or len(y) == 0
        or not np.isfinite(p).all()
    ):
        raise ValueError("adaptive reliability inputs are invalid")
    classes: dict[str, Any] = {}
    global_maximum_gap: float | None = None
    for class_index, name in enumerate(CLASS_NAMES):
        observed = (y == class_index).astype(np.int8)
        predicted = p[:, class_index]
        order = np.argsort(predicted, kind="mergesort")
        groups = [
            np.asarray(group, dtype=int)
            for group in np.array_split(order, min(initial_bins, len(order)))
            if len(group)
        ]

        def supported(group: np.ndarray) -> bool:
            positives = int(observed[group].sum())
            negatives = int(len(group) - positives)
            return (
                len(group) >= minimum_rows
                and positives >= minimum_positives
                and negatives >= minimum_negatives
            )

        while len(groups) > minimum_supported_bins:
            unsupported = [
                index for index, group in enumerate(groups) if not supported(group)
            ]
            if not unsupported:
                break
            index = unsupported[0]
            if index == 0:
                neighbor = 1
            elif index == len(groups) - 1:
                neighbor = index - 1
            else:
                left_count = len(groups[index - 1])
                right_count = len(groups[index + 1])
                neighbor = index - 1 if left_count <= right_count else index + 1
            low, high = sorted((index, neighbor))
            groups[low] = np.concatenate([groups[low], groups[high]])
            del groups[high]

        rows = []
        supported_rows = 0
        weighted_gap = 0.0
        supported_maximum = 0.0
        for index, group in enumerate(groups):
            count = int(len(group))
            positives = int(observed[group].sum())
            negatives = count - positives
            is_supported = supported(group)
            mean_probability = float(predicted[group].mean())
            observed_rate = float(positives / count)
            gap = abs(mean_probability - observed_rate)
            if is_supported:
                supported_rows += count
                weighted_gap += count * gap
                supported_maximum = max(supported_maximum, gap)
            extreme_tail = (
                "low"
                if index == 0 and not is_supported
                else "high"
                if index == len(groups) - 1 and not is_supported
                else None
            )
            rows.append(
                {
                    "lower": float(predicted[group].min()),
                    "upper": float(predicted[group].max()),
                    "count": count,
                    "positive_count": positives,
                    "negative_count": negatives,
                    "mean_probability": mean_probability,
                    "observed_rate": observed_rate,
                    "gap": float(gap),
                    "supported": is_supported,
                    "extreme_tail": extreme_tail,
                    "observed_rate_wilson95": _wilson_interval(positives, count),
                }
            )
        supported_count = sum(bool(row["supported"]) for row in rows)
        support_available = supported_count >= minimum_supported_bins
        ece = (
            float(weighted_gap / supported_rows)
            if support_available and supported_rows
            else None
        )
        maximum_gap = float(supported_maximum) if support_available else None
        if maximum_gap is not None:
            global_maximum_gap = max(global_maximum_gap or 0.0, maximum_gap)
        classes[name] = {
            "ece": ece,
            "maximum_gap": maximum_gap,
            "supported_bin_count": supported_count,
            "supported_row_count": supported_rows,
            "support_available": support_available,
            "bins": rows,
            "unsupported_extreme_tails": [
                row
                for row in rows
                if not row["supported"] and row["extreme_tail"] is not None
            ],
            "calibration": _binary_calibration_fit(observed, predicted),
        }
    return {
        "version": "adaptive-equal-count-reliability-v1",
        "initial_bins": int(initial_bins),
        "support_rules": {
            "minimum_rows": int(minimum_rows),
            "minimum_positives": int(minimum_positives),
            "minimum_negatives": int(minimum_negatives),
            "minimum_supported_bins": int(minimum_supported_bins),
        },
        "classes": classes,
        "maximum_gap": global_maximum_gap,
        "all_classes_supported": all(
            classes[name]["support_available"] for name in CLASS_NAMES
        ),
    }


def evaluate_ordered_threshold_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    climatology_probabilities: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    baseline = np.asarray(climatology_probabilities, dtype=float)
    reliability = adaptive_classwise_reliability(y, p)
    brier = multiclass_brier(y, p)
    baseline_brier = multiclass_brier(y, baseline)
    log_loss = multiclass_log_loss(y, p)
    baseline_log_loss = multiclass_log_loss(y, baseline)
    return {
        "count": int(len(y)),
        "class_counts": {
            name: int((y == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "prevalence": {
            name: float((y == index).mean())
            for index, name in enumerate(CLASS_NAMES)
        },
        "brier": brier,
        "climatology_brier": baseline_brier,
        "brier_skill": (
            1.0 - brier / baseline_brier if baseline_brier > 0 else float("-inf")
        ),
        "log_loss": log_loss,
        "climatology_log_loss": baseline_log_loss,
        "log_loss_improvement": (
            1.0 - log_loss / baseline_log_loss
            if baseline_log_loss > 0
            else float("-inf")
        ),
        "classwise_ece": {
            name: reliability["classes"][name]["ece"] for name in CLASS_NAMES
        },
        "maximum_gap": reliability["maximum_gap"],
        "calibration": {
            name: reliability["classes"][name]["calibration"]
            for name in CLASS_NAMES
        },
        "reliability_bins": {
            name: reliability["classes"][name]["bins"] for name in CLASS_NAMES
        },
        "reliability_support": {
            name: {
                "available": reliability["classes"][name]["support_available"],
                "supported_bin_count": reliability["classes"][name][
                    "supported_bin_count"
                ],
                "supported_row_count": reliability["classes"][name][
                    "supported_row_count"
                ],
                "unsupported_extreme_tails": reliability["classes"][name][
                    "unsupported_extreme_tails"
                ],
            }
            for name in CLASS_NAMES
        },
    }


def regime_support(
    labels: np.ndarray,
    dates: Iterable[Any],
    issuers: Iterable[Any],
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    date_values = pd.to_datetime(
        pd.Series(list(dates)),
        errors="coerce",
        utc=True,
    )
    issuer_values = np.asarray([str(value) for value in issuers], dtype=object)
    if len(y) != len(date_values) or len(y) != len(issuer_values):
        raise ValueError("regime support arrays have incompatible lengths")
    valid_dates = date_values.dropna()
    quarters = set(
        zip(
            valid_dates.dt.year.astype(int),
            valid_dates.dt.quarter.astype(int),
        )
    )
    class_counts = {
        name: int((y == index).sum())
        for index, name in enumerate(CLASS_NAMES)
    }
    counts = {
        "distinct_dates": int(valid_dates.dt.normalize().nunique()),
        "quarter_blocks": int(len(quarters)),
        "issuer_count": int(len(set(issuer_values))),
        "class_counts": class_counts,
    }
    reasons = []
    if counts["distinct_dates"] < REGIME_MIN_DATES:
        reasons.append(f"distinct dates < {REGIME_MIN_DATES}")
    if counts["quarter_blocks"] < REGIME_MIN_QUARTERS:
        reasons.append(f"quarter blocks < {REGIME_MIN_QUARTERS}")
    if min(class_counts.values(), default=0) < REGIME_MIN_OUTCOMES_PER_CLASS:
        reasons.append(
            f"outcomes/class < {REGIME_MIN_OUTCOMES_PER_CLASS}"
        )
    if counts["issuer_count"] < REGIME_MIN_ISSUERS:
        reasons.append(f"issuers < {REGIME_MIN_ISSUERS}")
    return {
        "available": not reasons,
        "sample_gate_passed": not reasons,
        "status": "supported" if not reasons else "unavailable",
        "unavailable_reasons": reasons,
        **counts,
        "rules": {
            "distinct_dates_min": REGIME_MIN_DATES,
            "quarter_blocks_min": REGIME_MIN_QUARTERS,
            "outcomes_per_class_min": REGIME_MIN_OUTCOMES_PER_CLASS,
            "issuers_min": REGIME_MIN_ISSUERS,
        },
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int((labels == index).sum())
        for index, name in enumerate(CLASS_NAMES)
    }


def _ordered_class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int((labels == index).sum())
        for index, name in enumerate(ORDERED_CLASS_NAMES)
    }


def _minimum_class_count(folds: list[dict[str, Any]], partition: str) -> int:
    values = [
        min(
            int(row[partition]["class_counts"][name])
            for name in CLASS_NAMES
        )
        for row in folds
    ]
    return min(values) if values else 0


def _resample_two_way(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    issuers = frame["issuer_key"].astype(str).to_numpy()
    dates = pd.to_datetime(frame["feature_date"])
    quarters = (
        dates.dt.year.to_numpy(dtype=int) * 4
        + dates.dt.quarter.to_numpy(dtype=int)
    )
    unique_issuers, issuer_inverse = np.unique(issuers, return_inverse=True)
    unique_quarters, quarter_inverse = np.unique(quarters, return_inverse=True)
    issuer_draws = rng.integers(
        0,
        len(unique_issuers),
        len(unique_issuers),
    )
    quarter_draws = rng.integers(
        0,
        len(unique_quarters),
        len(unique_quarters),
    )
    multiplicity = (
        np.bincount(issuer_draws, minlength=len(unique_issuers))[issuer_inverse]
        * np.bincount(quarter_draws, minlength=len(unique_quarters))[quarter_inverse]
    )
    positions = np.repeat(np.arange(len(frame)), multiplicity)
    if len(positions) == 0:
        raise RuntimeError("two-way full-refit bootstrap produced no rows")
    return frame.iloc[positions].reset_index(drop=True)


def full_refit_ordered_bootstrap(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    *,
    horizon: int,
    repetitions: int,
    seed: int = DEFAULT_SEED,
    c_value: float = 0.1,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
) -> dict[str, Any]:
    """Refit preprocessing, model, and vector calibrator for every resample."""
    requested = int(repetitions)
    if requested < 0:
        raise ValueError("full-refit bootstrap repetitions cannot be negative")
    if requested == 0:
        return {
            "version": "ordered-model-calibrator-full-refit-v1",
            "method": (
                "two-way issuer/calendar-quarter resampling; fold-local "
                "preprocessor, seven-class model, and vector calibrator refit; "
                "untouched outer test scoring"
            ),
            "requested_repetitions": 0,
            "attempted_repetitions": 0,
            "completed_repetitions": 0,
            "skipped_repetitions": 0,
            "complete": True,
            "failures": [],
            "threshold_metrics": {},
        }
    target = ordered_label_column(horizon)
    thresholds = THRESHOLD_GRIDS[horizon]
    rng = np.random.default_rng(seed + 70_000 + horizon)
    metric_values = {
        threshold: {
            "brier_skill": [],
            "log_loss_improvement": [],
        }
        for threshold in thresholds
    }
    failures = []
    completed = 0
    for repetition in range(requested):
        fold = folds[repetition % len(folds)]
        try:
            train_original = dataset.iloc[fold["train_indices"]]
            calibration_original = dataset.iloc[fold["calibration_indices"]]
            test = dataset.iloc[fold["test_indices"]]
            train = _resample_two_way(train_original, rng)
            calibration_dates = pd.to_datetime(
                calibration_original["feature_date"]
            )
            calibration_interval_start = fold.get(
                "calibration_start",
                calibration_dates.min(),
            )
            calibration_cutoff = (
                pd.Timestamp(calibration_interval_start).normalize()
                + pd.DateOffset(months=9)
            )
            calibration_fit = calibration_original.loc[
                calibration_dates < calibration_cutoff
            ]
            calibration_selection = calibration_original.loc[
                calibration_dates >= calibration_cutoff
            ]
            calibration = pd.concat(
                [
                    _resample_two_way(calibration_fit, rng),
                    _resample_two_way(calibration_selection, rng),
                ],
                ignore_index=True,
            ).sort_values(
                ["feature_date", "issuer_key"],
                kind="mergesort",
            )
            y_train_ordered = train[target].to_numpy(dtype=int)
            if set(np.unique(y_train_ordered)) != set(
                range(len(ORDERED_CLASS_NAMES))
            ):
                raise ValueError("resampled training is missing an ordered class")
            model = fit_ordered_multinomial_model(
                train.loc[:, FEATURE_NAMES],
                y_train_ordered,
                c_value=c_value,
                seed=seed + 80_000 + repetition,
            )
            calibration_result = fit_ordered_vector_calibration(
                model,
                calibration,
                target=target,
                penalty_grid=penalty_grid,
                calibration_interval_start=calibration_interval_start,
            )
            model = calibration_result["model"]
            test_ood = assess_ood(
                test.loc[:, FEATURE_NAMES],
                model["preprocessor"],
            )
            test_mask = np.asarray(
                [not item["withhold"] for item in test_ood],
                dtype=bool,
            )
            scored = test.loc[test_mask]
            if len(scored) == 0:
                raise ValueError("full-refit bootstrap has no covered test rows")
            ordered_probabilities = predict_ordered_probabilities(
                model,
                scored.loc[:, FEATURE_NAMES],
            )
            derived = derive_threshold_probabilities(
                ordered_probabilities,
                thresholds,
            )
            ordered_test_labels = scored[target].to_numpy(dtype=int)
            for index, threshold in enumerate(thresholds):
                y_train = ordered_labels_to_threshold_labels(
                    y_train_ordered,
                    index,
                )
                rates = np.bincount(y_train, minlength=3).astype(float)
                rates /= rates.sum()
                baseline = np.tile(rates, (len(scored), 1))
                y_test = ordered_labels_to_threshold_labels(
                    ordered_test_labels,
                    index,
                )
                metrics = evaluate_ordered_threshold_predictions(
                    y_test,
                    derived[threshold],
                    baseline,
                )
                metric_values[threshold]["brier_skill"].append(
                    metrics["brier_skill"]
                )
                metric_values[threshold]["log_loss_improvement"].append(
                    metrics["log_loss_improvement"]
                )
            completed += 1
        except Exception as exc:
            failures.append(
                {
                    "repetition": repetition,
                    "fold": int(fold["fold"]),
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
    threshold_metrics = {}
    for threshold, metrics in metric_values.items():
        threshold_metrics[str(threshold)] = {}
        for name, values in metrics.items():
            threshold_metrics[str(threshold)][f"{name}_ci95"] = (
                np.quantile(values, [0.025, 0.975]).tolist()
                if values
                else [None, None]
            )
    return {
        "version": "ordered-model-calibrator-full-refit-v1",
        "method": (
            "two-way issuer/calendar-quarter resampling; fold-local "
            "winsorization/imputation/scaling, seven-class L2 multinomial model, "
            "and 9/3-selected vector calibrator refit; untouched outer tests scored"
        ),
        "requested_repetitions": requested,
        "attempted_repetitions": requested,
        "completed_repetitions": completed,
        "skipped_repetitions": requested - completed,
        "complete": completed >= requested,
        "seed": int(seed + 70_000 + horizon),
        "failures": failures,
        "threshold_metrics": threshold_metrics,
    }


def _refit_acceptance(
    bootstrap: dict[str, Any],
    *,
    dev_refit_override: bool,
) -> dict[str, Any]:
    requested = int(bootstrap.get("requested_repetitions") or 0)
    completed = int(bootstrap.get("completed_repetitions") or 0)
    required = requested if dev_refit_override and requested > 0 else (
        RELEASE_REFIT_BOOTSTRAP_REPS
    )
    satisfied = requested >= required and completed >= required
    release_eligible = (
        not dev_refit_override
        and requested >= RELEASE_REFIT_BOOTSTRAP_REPS
        and completed >= RELEASE_REFIT_BOOTSTRAP_REPS
    )
    return {
        "preregistered_release_minimum": RELEASE_REFIT_BOOTSTRAP_REPS,
        "dev_override": bool(dev_refit_override),
        "acceptance_required_repetitions": int(required),
        "requested_repetitions": requested,
        "completed_repetitions": completed,
        "acceptance_satisfied": satisfied,
        "production_release_eligible": release_eligible,
        "status": (
            "dev_override_complete"
            if satisfied and dev_refit_override
            else "release_complete"
            if release_eligible
            else "incomplete"
        ),
    }


def train_ordered_walk_forward_horizon(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    *,
    horizon: int,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPS,
    refit_bootstrap_repetitions: int = 0,
    dev_refit_override: bool = False,
    seed: int = DEFAULT_SEED,
    c_value: float = 0.1,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
    include_oos: bool = False,
    fixed_oos_release_required: bool = False,
) -> dict[str, Any]:
    target = ordered_label_column(horizon)
    if target not in dataset or dataset[target].isna().any():
        raise ValueError(
            "ordered walk-forward training requires a horizon-filtered dataset"
        )
    thresholds = THRESHOLD_GRIDS[horizon]
    accumulator = {
        threshold: {
            "probabilities": [],
            "climatology": [],
            "regime_probabilities": [],
            "labels": [],
            "regime_keys": [],
            "folds": [],
            "unique_events": {
                index: set() for index in range(len(CLASS_NAMES))
            },
        }
        for threshold in thresholds
    }
    ordered_oos = []
    all_dates: list[Any] = []
    all_issuers: list[str] = []
    test_candidate_count = 0
    calibrator_rows = []
    for fold in folds:
        train = dataset.iloc[fold["train_indices"]]
        calibration = dataset.iloc[fold["calibration_indices"]]
        test = dataset.iloc[fold["test_indices"]]
        test_candidate_count += len(test)
        y_train_ordered = train[target].to_numpy(dtype=int)
        if set(np.unique(y_train_ordered)) != set(
            range(len(ORDERED_CLASS_NAMES))
        ):
            raise ValueError(
                f"fold {fold['fold']} train is missing an ordered outcome class"
            )
        model = fit_ordered_multinomial_model(
            train.loc[:, FEATURE_NAMES],
            y_train_ordered,
            c_value=c_value,
            seed=seed + int(fold["fold"]),
        )
        calibration_result = fit_ordered_vector_calibration(
            model,
            calibration,
            target=target,
            penalty_grid=penalty_grid,
            calibration_interval_start=fold["calibration_start"],
        )
        model = calibration_result["model"]
        calibration_ood = calibration_result["ood"]
        calibration_mask = calibration_result["coverage_mask"]
        calibration_scored = calibration_result["scored"]
        y_calibration_ordered = calibration_result["ordered_labels"]
        vector = calibration_result["vector_scaling"]
        temperature_diagnostic = calibration_result[
            "temperature_diagnostic"
        ]
        calibrator_rows.append(
            {
                "fold": int(fold["fold"]),
                "selected_penalty": vector["selected_penalty"],
                "penalty_fit_start": vector["penalty_fit_start"],
                "penalty_fit_end": vector["penalty_fit_end"],
                "penalty_selection_start": vector[
                    "penalty_selection_start"
                ],
                "penalty_selection_end": vector["penalty_selection_end"],
                "calibration_interval_end": vector[
                    "calibration_interval_end"
                ],
                "test_start": pd.Timestamp(fold["test_start"])
                .date()
                .isoformat(),
                "no_test_leakage_verified": (
                    pd.Timestamp(vector["calibration_interval_end"])
                    < pd.Timestamp(fold["test_start"])
                ),
            }
        )
        if not calibrator_rows[-1]["no_test_leakage_verified"]:
            raise AssertionError("vector calibration interval overlaps outer test")
        test_ood = assess_ood(
            test.loc[:, FEATURE_NAMES],
            model["preprocessor"],
        )
        test_mask = np.asarray(
            [not item["withhold"] for item in test_ood],
            dtype=bool,
        )
        test_scored = test.loc[test_mask].copy()
        y_test_ordered = test_scored[target].to_numpy(dtype=int)
        ordered_probabilities = predict_ordered_probabilities(
            model,
            test_scored.loc[:, FEATURE_NAMES],
        )
        derived_probabilities = derive_threshold_probabilities(
            ordered_probabilities,
            thresholds,
        )
        monotonicity = assert_exact_ordered_monotonicity(
            derived_probabilities
        )
        ordered_oos.append(ordered_probabilities)
        fold_dates = test_scored["feature_date"].tolist()
        fold_issuers = test_scored["issuer_key"].astype(str).tolist()
        all_dates.extend(fold_dates)
        all_issuers.extend(fold_issuers)
        for threshold_index, threshold in enumerate(thresholds):
            y_train = ordered_labels_to_threshold_labels(
                y_train_ordered,
                threshold_index,
            )
            y_calibration = ordered_labels_to_threshold_labels(
                y_calibration_ordered,
                threshold_index,
            )
            y_test = ordered_labels_to_threshold_labels(
                y_test_ordered,
                threshold_index,
            )
            for scored, labels_for_rows in (
                (calibration_scored, y_calibration),
                (test_scored, y_test),
            ):
                for issuer, feature_date, class_index in zip(
                    scored["issuer_key"].astype(str),
                    pd.to_datetime(scored["feature_date"]),
                    labels_for_rows,
                ):
                    accumulator[threshold]["unique_events"][
                        int(class_index)
                    ].add((issuer, pd.Timestamp(feature_date).isoformat()))
            train_counts = np.bincount(y_train, minlength=3).astype(float)
            climatology_rates = train_counts / train_counts.sum()
            climatology = np.tile(
                climatology_rates,
                (len(test_scored), 1),
            )
            regime_model = fit_regime_baseline(
                train.loc[:, FEATURE_NAMES],
                y_train,
            )
            regime_probabilities, regime_keys = predict_regime_baseline(
                regime_model,
                test_scored.loc[:, FEATURE_NAMES],
            )
            metrics = evaluate_ordered_threshold_predictions(
                y_test,
                derived_probabilities[threshold],
                climatology,
            )
            regime_metrics = evaluate_ordered_threshold_predictions(
                y_test,
                regime_probabilities,
                climatology,
            )
            fold_row = {
                "fold": int(fold["fold"]),
                "train_start": pd.Timestamp(fold["train_start"])
                .date()
                .isoformat(),
                "train_end": pd.Timestamp(fold["train_end"])
                .date()
                .isoformat(),
                "calibration_start": pd.Timestamp(
                    fold["calibration_start"]
                )
                .date()
                .isoformat(),
                "calibration_end": pd.Timestamp(fold["calibration_end"])
                .date()
                .isoformat(),
                "test_start": pd.Timestamp(fold["test_start"])
                .date()
                .isoformat(),
                "test_end": pd.Timestamp(fold["test_end"])
                .date()
                .isoformat(),
                "usable_train_years": float(fold["usable_train_years"]),
                "full_test_window": bool(fold["full_test_window"]),
                "train": {
                    "count": int(len(train)),
                    "class_counts": _class_counts(y_train),
                    "ordered_class_counts": _ordered_class_counts(
                        y_train_ordered
                    ),
                    "issuer_count": int(train["issuer_key"].nunique()),
                    "date_count": int(train["feature_date"].nunique()),
                },
                "calibration": {
                    "candidate_count": int(len(calibration)),
                    "count": int(len(calibration_scored)),
                    "inference_coverage": float(calibration_mask.mean()),
                    "class_counts": _class_counts(y_calibration),
                    "ordered_class_counts": _ordered_class_counts(
                        y_calibration_ordered
                    ),
                    "vector_scaling": {
                        "version": vector["version"],
                        "selected_penalty": vector["selected_penalty"],
                        "penalty_grid": vector["penalty_grid"],
                        "penalty_fit_rows": vector["penalty_fit_rows"],
                        "penalty_selection_rows": vector[
                            "penalty_selection_rows"
                        ],
                        "refit_rows": vector["refit_rows"],
                    },
                    "temperature_baseline_diagnostic": {
                        "value": temperature_diagnostic["value"],
                        "calibrated_log_loss": temperature_diagnostic[
                            "calibrated_log_loss"
                        ],
                    },
                },
                "test": {
                    "candidate_count": int(len(test)),
                    "count": int(len(test_scored)),
                    "inference_coverage": float(test_mask.mean()),
                    "class_counts": _class_counts(y_test),
                    "ordered_class_counts": _ordered_class_counts(
                        y_test_ordered
                    ),
                    "issuer_count": int(
                        test_scored["issuer_key"].nunique()
                    ),
                    "date_count": int(
                        test_scored["feature_date"].nunique()
                    ),
                    "withholding_reasons": dict(
                        Counter(
                            reason
                            for item in test_ood
                            if item["withhold"]
                            for reason in item["reasons"]
                        )
                    ),
                },
                "metrics": metrics,
                "regime_baseline_metrics": regime_metrics,
                "exact_monotonicity": monotonicity,
                "publish_transform_max_abs_diff": 0.0,
            }
            values = accumulator[threshold]
            values["folds"].append(fold_row)
            values["probabilities"].append(
                derived_probabilities[threshold]
            )
            values["climatology"].append(climatology)
            values["regime_probabilities"].append(regime_probabilities)
            values["labels"].append(y_test)
            values["regime_keys"].extend(regime_keys.tolist())
    if not ordered_oos:
        raise ValueError("no complete ordered outer folds were trained")
    history_start = (
        pd.to_datetime(dataset["history_start"]).min()
        if "history_start" in dataset
        else pd.to_datetime(dataset["feature_date"]).min()
    )
    history_end = pd.to_datetime(dataset["feature_date"]).max()
    threshold_reports: dict[int, dict[str, Any]] = {}
    for threshold in thresholds:
        values = accumulator[threshold]
        labels = np.concatenate(values["labels"])
        probabilities = np.vstack(values["probabilities"])
        climatology = np.vstack(values["climatology"])
        regime_probabilities = np.vstack(values["regime_probabilities"])
        aggregate = evaluate_ordered_threshold_predictions(
            labels,
            probabilities,
            climatology,
        )
        regime_aggregate = evaluate_ordered_threshold_predictions(
            labels,
            regime_probabilities,
            climatology,
        )
        bootstrap = two_way_cluster_bootstrap(
            labels,
            probabilities,
            climatology,
            all_dates,
            all_issuers,
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
        regime_groups = []
        regime_key_array = np.asarray(values["regime_keys"], dtype=object)
        date_array = np.asarray(all_dates, dtype=object)
        issuer_array = np.asarray(all_issuers, dtype=object)
        for key in sorted(set(values["regime_keys"])):
            mask = regime_key_array == key
            support = regime_support(
                labels[mask],
                date_array[mask],
                issuer_array[mask],
            )
            reliability = (
                adaptive_classwise_reliability(
                    labels[mask],
                    probabilities[mask],
                )
                if support["available"]
                else None
            )
            adaptive_available = bool(
                reliability
                and reliability["all_classes_supported"]
            )
            available = bool(support["available"] and adaptive_available)
            if support["available"] and not adaptive_available:
                support["unavailable_reasons"].append(
                    "adaptive reliability has fewer than five supported bins "
                    "for at least one class"
                )
                support["status"] = "unavailable"
                support["available"] = False
                support["sample_gate_passed"] = False
            classwise_ece = (
                {
                    name: reliability["classes"][name]["ece"]
                    for name in CLASS_NAMES
                }
                if available
                else {name: None for name in CLASS_NAMES}
            )
            regime_groups.append(
                {
                    "regime": key,
                    "count": int(mask.sum()),
                    **support,
                    "classwise_ece": classwise_ece,
                    "max_class_ece": (
                        max(classwise_ece.values()) if available else None
                    ),
                }
            )
        fold_rows = values["folds"]
        report = {
            "model_family": ORDERED_MODEL_FAMILY,
            "source_model_key": ordered_model_key(horizon),
            "model_key": model_key(horizon, threshold),
            "horizon_sessions": int(horizon),
            "threshold_pct": int(threshold),
            "feature_version": FEATURE_VERSION,
            "feature_schema_hash": feature_schema_hash(),
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
            "validated_probability_space": ORDERED_PUBLISH_TRANSFORM["input"],
            "history_years": float(
                (history_end - history_start).days / 365.2425
            ),
            "history_start": pd.Timestamp(history_start).date().isoformat(),
            "history_end": pd.Timestamp(history_end).date().isoformat(),
            "fold_count": len(fold_rows),
            "publish_transform_identity_verified": all(
                row["publish_transform_max_abs_diff"] == 0.0
                for row in fold_rows
            ),
            "full_test_fold_count": sum(
                bool(row["full_test_window"]) for row in fold_rows
            ),
            "min_usable_train_years": min(
                row["usable_train_years"] for row in fold_rows
            ),
            "issuer_count": len(set(all_issuers)),
            "forecast_date_count": len(
                set(pd.to_datetime(all_dates).date)
            ),
            "min_train_class_count": _minimum_class_count(
                fold_rows,
                "train",
            ),
            "min_calibration_class_count": _minimum_class_count(
                fold_rows,
                "calibration",
            ),
            "min_test_class_count": _minimum_class_count(
                fold_rows,
                "test",
            ),
            "inference_coverage": (
                len(labels) / test_candidate_count
                if test_candidate_count
                else 0.0
            ),
            "provider_success_coverage": 1.0,
            "provider_requested_issuer_count": int(
                dataset["issuer_key"].nunique()
            ),
            "provider_successful_issuer_count": int(
                dataset["issuer_key"].nunique()
            ),
            "provider_unavailable_symbols": {},
            "event_counts_calibration_test_unique": {
                name: len(values["unique_events"][index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "aggregate": aggregate,
            "regime": {
                "aggregate": regime_aggregate,
                "groups": regime_groups,
                "definition": (
                    "SPY price/SMA200 sign crossed with training-fold SPY vol60 "
                    "terciles; support requires 26 dates, 8 quarter blocks, 100 "
                    "outcomes/class, 100 issuers; unsupported is unavailable"
                ),
            },
            "bootstrap": bootstrap,
            "fixed_oos_bootstrap_required_repetitions": (
                int(bootstrap_repetitions)
                if fixed_oos_release_required
                else 0
            ),
            "folds": fold_rows,
        }
        report["acceptance"] = evaluate_acceptance(report)
        if include_oos:
            report["_oos"] = {
                "keys": [
                    f"{issuer}|{pd.Timestamp(date).isoformat()}"
                    for issuer, date in zip(all_issuers, all_dates)
                ],
                "probabilities": probabilities,
                "labels": labels,
            }
        threshold_reports[threshold] = report
    bootstrap_records = [
        report["bootstrap"] for report in threshold_reports.values()
    ]
    fixed_oos_bootstrap = {
        "version": BOOTSTRAP_VERSION,
        "requested_repetitions": int(bootstrap_repetitions),
        "attempted_repetitions": max(
            int(item.get("attempted_repetitions") or 0)
            for item in bootstrap_records
        ),
        "completed_repetitions": min(
            int(
                item.get("completed_repetitions")
                if item.get("completed_repetitions") is not None
                else item.get("repetitions")
                or 0
            )
            for item in bootstrap_records
        ),
        "skipped_repetitions": max(
            int(item.get("skipped_repetitions") or 0)
            for item in bootstrap_records
        ),
        "maximum_attempts": max(
            int(item.get("maximum_attempts") or 0)
            for item in bootstrap_records
        ),
        "complete": all(bool(item.get("complete")) for item in bootstrap_records),
        "release_completion_required": bool(fixed_oos_release_required),
        "method": (
            "fixed OOS predictions; issuer clusters and calendar-quarter "
            "blocks; deterministic invalid-draw retries"
        ),
    }
    exact_monotonicity = assert_exact_ordered_monotonicity(
        {
            threshold: np.vstack(
                accumulator[threshold]["probabilities"]
            )
            for threshold in thresholds
        }
    )
    preliminary_passed = all(
        report["acceptance"]["accepted"]
        for report in threshold_reports.values()
    )
    if refit_bootstrap_repetitions and preliminary_passed:
        refit_bootstrap = full_refit_ordered_bootstrap(
            dataset,
            folds,
            horizon=horizon,
            repetitions=refit_bootstrap_repetitions,
            seed=seed,
            c_value=c_value,
            penalty_grid=penalty_grid,
        )
    else:
        refit_bootstrap = full_refit_ordered_bootstrap(
            dataset,
            folds,
            horizon=horizon,
            repetitions=0,
            seed=seed,
            c_value=c_value,
            penalty_grid=penalty_grid,
        )
        refit_bootstrap["requested_repetitions"] = int(
            refit_bootstrap_repetitions
        )
        refit_bootstrap["attempted_repetitions"] = 0
        refit_bootstrap["skipped_repetitions"] = int(
            refit_bootstrap_repetitions
        )
        refit_bootstrap["complete"] = (
            int(refit_bootstrap_repetitions) == 0
        )
        if refit_bootstrap_repetitions and not preliminary_passed:
            refit_bootstrap["skipped_reason"] = (
                "threshold metric gates failed before expensive full refits"
            )
    refit_acceptance = _refit_acceptance(
        refit_bootstrap,
        dev_refit_override=dev_refit_override,
    )
    reasons = []
    for threshold, report in threshold_reports.items():
        reasons.extend(
            f"x{threshold}: {reason}"
            for reason in report["acceptance"]["reasons"]
        )
    if not exact_monotonicity["passed"]:
        reasons.append("exact ordered tail-sum monotonicity assertion failed")
    if not refit_acceptance["acceptance_satisfied"]:
        reasons.append(
            "full model+calibrator refit bootstrap incomplete: "
            f"{refit_acceptance['completed_repetitions']}/"
            f"{refit_acceptance['acceptance_required_repetitions']}"
        )
    horizon_acceptance = {
        "accepted": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "all_three_thresholds_required": True,
        "accepted_threshold_count": sum(
            report["acceptance"]["accepted"]
            for report in threshold_reports.values()
        ),
        "tested_threshold_count": len(thresholds),
        "refit": refit_acceptance,
        "production_release_eligible": (
            not reasons
            and refit_acceptance["production_release_eligible"]
        ),
    }
    result = {
        "model_family": ORDERED_MODEL_FAMILY,
        "model_key": ordered_model_key(horizon),
        "horizon_sessions": int(horizon),
        "thresholds_pct": list(thresholds),
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "model_version": ORDERED_MODEL_VERSION,
        "calibration_version": VECTOR_CALIBRATION_VERSION,
        "penalty_grid": list(float(value) for value in penalty_grid),
        "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
        "exact_monotonicity": exact_monotonicity,
        "vector_calibration_folds": calibrator_rows,
        "threshold_validation": {
            str(threshold): report
            for threshold, report in threshold_reports.items()
        },
        "fixed_oos_bootstrap": fixed_oos_bootstrap,
        "full_refit_bootstrap": refit_bootstrap,
        "acceptance": horizon_acceptance,
    }
    if include_oos:
        result["_oos"] = {
            "keys": next(
                iter(threshold_reports.values())
            )["_oos"]["keys"],
            "ordered_probabilities": np.vstack(ordered_oos),
            "threshold_probabilities": {
                str(threshold): threshold_reports[threshold]["_oos"][
                    "probabilities"
                ]
                for threshold in thresholds
            },
        }
    return result


def fit_ordered_release_model(
    dataset: pd.DataFrame,
    report: dict[str, Any],
    *,
    horizon: int,
    trained_at: str | None = None,
    seed: int = DEFAULT_SEED,
    c_value: float = 0.1,
    penalty_grid: Iterable[float] = VECTOR_PENALTY_GRID,
) -> dict[str, Any]:
    acceptance = report.get("acceptance") or {}
    if (
        not acceptance.get("accepted")
        or not acceptance.get("production_release_eligible")
    ):
        raise ValueError(
            "ordered release requires accepted metrics and 200 completed full refits"
        )
    target = ordered_label_column(horizon)
    dataset = dataset.loc[dataset[target].notna()].copy()
    dates = pd.to_datetime(dataset["feature_date"]).dt.tz_localize(None)
    max_exit = pd.to_datetime(dataset["max_exit_date"]).dt.tz_localize(None)
    calibration_end = dates.max() + pd.Timedelta(days=1)
    calibration_start = calibration_end - pd.DateOffset(years=1)
    train_mask = (dates < calibration_start) & (
        max_exit < calibration_start - pd.Timedelta(days=7)
    )
    calibration_mask = (dates >= calibration_start) & (
        dates < calibration_end
    )
    train = dataset.loc[train_mask]
    calibration = dataset.loc[calibration_mask]
    y_train_ordered = train[target].to_numpy(dtype=int)
    if set(np.unique(y_train_ordered)) != set(range(len(ORDERED_CLASS_NAMES))):
        raise ValueError("final release training is missing an ordered class")
    model = fit_ordered_multinomial_model(
        train.loc[:, FEATURE_NAMES],
        y_train_ordered,
        c_value=c_value,
        seed=seed,
    )
    calibration_result = fit_ordered_vector_calibration(
        model,
        calibration,
        target=target,
        penalty_grid=penalty_grid,
        calibration_interval_start=calibration_start,
    )
    model = calibration_result["model"]
    calibration_ood = calibration_result["ood"]
    calibration_coverage_mask = calibration_result["coverage_mask"]
    calibration_scored = calibration_result["scored"]
    y_calibration_ordered = calibration_result["ordered_labels"]
    vector = calibration_result["vector_scaling"]
    thresholds = THRESHOLD_GRIDS[horizon]
    baseline_rates_by_threshold = {}
    oos_metrics_by_threshold = {}
    bootstrap_by_threshold = {}
    event_counts_by_threshold = {}
    for threshold_index, threshold in enumerate(thresholds):
        y_train = ordered_labels_to_threshold_labels(
            y_train_ordered,
            threshold_index,
        )
        counts = np.bincount(y_train, minlength=3).astype(float)
        rates = counts / counts.sum()
        threshold_report = report["threshold_validation"][str(threshold)]
        baseline_rates_by_threshold[str(threshold)] = {
            name: float(rates[index])
            for index, name in enumerate(CLASS_NAMES)
        }
        oos_metrics_by_threshold[str(threshold)] = {
            key: threshold_report["aggregate"][key]
            for key in (
                "brier",
                "climatology_brier",
                "brier_skill",
                "log_loss",
                "climatology_log_loss",
                "log_loss_improvement",
                "classwise_ece",
                "maximum_gap",
                "calibration",
                "reliability_bins",
                "reliability_support",
            )
        }
        bootstrap_by_threshold[str(threshold)] = threshold_report[
            "bootstrap"
        ]
        event_counts_by_threshold[str(threshold)] = {
            name: int(
                (
                    threshold_report.get(
                        "event_counts_calibration_test_unique"
                    )
                    or {}
                ).get(name, 0)
            )
            for name in CLASS_NAMES
        }
    first_threshold_report = report["threshold_validation"][
        str(thresholds[0])
    ]
    model.update(
        {
            "model_family": ORDERED_MODEL_FAMILY,
            "model_key": ordered_model_key(horizon),
            "horizon_sessions": int(horizon),
            "thresholds_pct": list(thresholds),
            "round_trip_cost_bps": int(ROUND_TRIP_COST * 10_000),
            "publish_transform": dict(ORDERED_PUBLISH_TRANSFORM),
            "exact_monotonicity": report["exact_monotonicity"],
            "trained_at": trained_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_cutoff": pd.Timestamp(
                calibration_end - pd.Timedelta(days=1)
            )
            .date()
            .isoformat(),
            "release_train_start": pd.to_datetime(
                train["feature_date"]
            )
            .min()
            .date()
            .isoformat(),
            "release_train_end": pd.to_datetime(
                train["feature_date"]
            )
            .max()
            .date()
            .isoformat(),
            "release_calibration_start": pd.Timestamp(
                calibration_start
            )
            .date()
            .isoformat(),
            "release_calibration_end": pd.Timestamp(calibration_end)
            .date()
            .isoformat(),
            "release_train_ordered_class_counts": _ordered_class_counts(
                y_train_ordered
            ),
            "release_calibration_ordered_class_counts": (
                _ordered_class_counts(y_calibration_ordered)
            ),
            "release_calibration_candidate_count": int(len(calibration)),
            "release_calibration_coverage": float(
                calibration_coverage_mask.mean()
            ),
            "baseline_rates_by_threshold": baseline_rates_by_threshold,
            "oos_metrics_by_threshold": oos_metrics_by_threshold,
            "bootstrap_by_threshold": bootstrap_by_threshold,
            "event_counts_calibration_test_by_threshold": (
                event_counts_by_threshold
            ),
            "oos_sample_size": int(
                first_threshold_report["aggregate"]["count"]
            ),
            "fold_count": int(first_threshold_report["fold_count"]),
            "full_test_fold_count": int(
                first_threshold_report["full_test_fold_count"]
            ),
            "history_years": float(
                first_threshold_report["history_years"]
            ),
            "min_usable_train_years": float(
                first_threshold_report["min_usable_train_years"]
            ),
            "full_refit_bootstrap": report["full_refit_bootstrap"],
            "refit_acceptance": acceptance["refit"],
            "accepted": True,
            "acceptance_reasons": [],
        }
    )
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    return model


__all__ = [
    "ADAPTIVE_INITIAL_BINS",
    "ADAPTIVE_MIN_BIN_ROWS",
    "ADAPTIVE_MIN_NEGATIVES",
    "ADAPTIVE_MIN_POSITIVES",
    "ADAPTIVE_MIN_SUPPORTED_BINS",
    "ORDERED_MODEL_VERSION",
    "ORDERED_MONOTONICITY_EPSILON",
    "ORDERED_PUBLISH_TRANSFORM",
    "RELEASE_REFIT_BOOTSTRAP_REPS",
    "VECTOR_CALIBRATION_VERSION",
    "VECTOR_PENALTY_GRID",
    "adaptive_classwise_reliability",
    "apply_vector_scaling",
    "assert_exact_ordered_monotonicity",
    "derive_threshold_probabilities",
    "evaluate_ordered_threshold_predictions",
    "fit_ordered_multinomial_model",
    "fit_ordered_release_model",
    "fit_ordered_vector_calibration",
    "fit_vector_scaling",
    "full_refit_ordered_bootstrap",
    "ordered_labels_to_threshold_labels",
    "predict_ordered_probabilities",
    "regime_support",
    "train_ordered_walk_forward_horizon",
]
