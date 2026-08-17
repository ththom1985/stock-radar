"""Fold-local multinomial models, calibration, validation, and release artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .probability_contract import CLASS_NAMES, HORIZONS, label_column, model_key
from .probability_features import FEATURE_NAMES, FEATURE_VERSION, feature_schema_hash

MODEL_VERSION = "probability-multinomial-l2-v1"
CALIBRATION_VERSION = "temporal-temperature-v1"
BOOTSTRAP_VERSION = "oos-issuer-quarter-block-v1"
PUBLISH_TRANSFORM_VERSION = "raw-temperature-scaled-identity-v1"
PUBLISH_TRANSFORM = {
    "version": PUBLISH_TRANSFORM_VERSION,
    "operation": "identity",
    "input": "temperature-scaled multinomial softmax",
    "caps": None,
    "cross_threshold_projection": None,
}
GRID_MONOTONICITY_GATES = {
    "violation_rate_max": 0.01,
    "violation_p95_magnitude_max": 0.01,
    "violation_max_magnitude_max": 0.03,
}
DEFAULT_SEED = 1729
DEFAULT_BOOTSTRAP_REPS = 200
RELEASE_BOOTSTRAP_REPS = 1000
RELIABILITY_BINS = 10
MATERIAL_NEGATIVE_FOLD_SKILL = -0.02

STRICT_ACCEPTANCE_GATES = {
    "history_years_min": 8.0,
    "fold_count_min": 5,
    "full_test_fold_count_min": 5,
    "usable_train_years_min": 5.0,
    "issuer_count_min": 200,
    "forecast_date_count_min": 100,
    "train_per_class_min": 1000,
    "calibration_per_class_min": 300,
    "test_per_class_min": 200,
    "inference_coverage_min": 0.80,
    "provider_success_coverage_min": 0.80,
    "provider_successful_issuer_count_min": 200,
    "brier_skill_min": 0.02,
    "brier_skill_ci_low_strict_min": 0.0,
    "log_loss_improvement_min": 0.01,
    "classwise_ece_max": 0.03,
    "maximum_gap_max": 0.08,
    "calibration_slope_min": 0.8,
    "calibration_slope_max": 1.2,
    "calibration_intercept_min": -0.1,
    "calibration_intercept_max": 0.1,
    "regime_ece_max": 0.05,
}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -745.0, 50.0))
    return exponent / exponent.sum(axis=1, keepdims=True)


def apply_publish_transform(
    probabilities: np.ndarray,
    transform: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply the serialized production transform (identity in this MVP)."""
    selected = transform or PUBLISH_TRANSFORM
    if selected != PUBLISH_TRANSFORM:
        raise ValueError("unsupported probability publish transform")
    values = np.asarray(probabilities, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values.reshape(1, -1)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise ValueError("publish-transform input is not a valid three-class simplex")
    result = values.copy()
    return result[0] if one_dimensional else result


def _as_matrix(
    values: pd.DataFrame | np.ndarray,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> np.ndarray:
    names = tuple(feature_names)
    if isinstance(values, pd.DataFrame):
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        matrix = values.loc[:, names].to_numpy(dtype=float)
    else:
        matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(
            f"feature matrix must have {len(names)} columns, got {matrix.shape}"
        )
    return matrix


def _safe_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        result = np.nanquantile(values, quantile, axis=0)
    if not np.isfinite(result).all():
        bad = np.flatnonzero(~np.isfinite(result)).tolist()
        raise ValueError(f"training features are entirely missing/nonfinite: {bad}")
    return result


def _psi_reference(values: np.ndarray) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    quantiles = np.linspace(0.0, 1.0, 11)
    for column in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column]), column]
        edges = np.unique(np.quantile(finite, quantiles))
        if len(edges) < 3:
            center = float(np.median(finite))
            scale = max(abs(center) * 1e-6, 1e-9)
            edges = np.array([center - scale, center, center + scale])
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(finite, bins=edges)
        frequencies = (counts / max(counts.sum(), 1)).tolist()
        references.append(
            {
                "edges": [
                    None if not np.isfinite(value) else float(value)
                    for value in edges
                ],
                "frequencies": [float(value) for value in frequencies],
            }
        )
    return references


def fit_preprocessor(
    values: pd.DataFrame | np.ndarray,
    *,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    """Fit winsorization, imputation, scaling, and OOD references on train only."""
    names = tuple(feature_names)
    matrix = _as_matrix(values, names)
    finite_matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    q005 = _safe_quantile(finite_matrix, 0.005)
    q995 = _safe_quantile(finite_matrix, 0.995)
    clipped = np.clip(finite_matrix, q005, q995)
    median = _safe_quantile(clipped, 0.5)
    missing_indicator_indices = np.flatnonzero(np.isnan(clipped).any(axis=0))
    filled = np.where(np.isnan(clipped), median, clipped)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    robust_center = _safe_quantile(finite_matrix, 0.5)
    q25 = _safe_quantile(finite_matrix, 0.25)
    q75 = _safe_quantile(finite_matrix, 0.75)
    robust_scale = q75 - q25
    robust_scale = np.where(robust_scale > 1e-12, robust_scale, scale)
    robust_filled = np.where(np.isnan(finite_matrix), robust_center, finite_matrix)
    standardized = np.clip(
        (robust_filled - robust_center) / robust_scale,
        -50.0,
        50.0,
    )
    distances = np.sqrt(np.mean(np.square(standardized), axis=1))
    distance_p995 = float(np.quantile(distances, 0.995))
    return {
        "feature_names": list(names),
        "winsor_lower_005": q005.tolist(),
        "winsor_upper_995": q995.tolist(),
        "median": median.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "missing_indicator_indices": missing_indicator_indices.tolist(),
        "output_dimension": int(
            len(names) + len(missing_indicator_indices)
        ),
        "fit_row_count": int(len(matrix)),
        "fit_scope": "outer_fold_training_only",
        "ood": {
            "raw_lower_005": q005.tolist(),
            "raw_upper_995": q995.tolist(),
            "robust_center": robust_center.tolist(),
            "robust_scale": robust_scale.tolist(),
            "distance_p995": distance_p995,
            "psi_reference": _psi_reference(finite_matrix),
        },
    }


def transform_preprocessor(
    values: pd.DataFrame | np.ndarray,
    preprocessor: dict[str, Any],
    *,
    require_complete: bool = False,
) -> np.ndarray:
    names = tuple(preprocessor["feature_names"])
    matrix = _as_matrix(values, names)
    nonfinite = ~np.isfinite(matrix)
    if require_complete and nonfinite.any():
        locations = np.argwhere(nonfinite)
        detail = [
            f"row {int(row)}:{names[int(column)]}"
            for row, column in locations[:8]
        ]
        raise ValueError("nonfinite inference features: " + ", ".join(detail))
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    lower = np.asarray(preprocessor["winsor_lower_005"], dtype=float)
    upper = np.asarray(preprocessor["winsor_upper_995"], dtype=float)
    medians = np.asarray(preprocessor["median"], dtype=float)
    clipped = np.clip(matrix, lower, upper)
    indicators = np.isnan(clipped)[
        :, np.asarray(preprocessor["missing_indicator_indices"], dtype=int)
    ].astype(float)
    filled = np.where(np.isnan(clipped), medians, clipped)
    transformed = (
        filled - np.asarray(preprocessor["mean"], dtype=float)
    ) / np.asarray(preprocessor["scale"], dtype=float)
    if indicators.shape[1]:
        transformed = np.concatenate([transformed, indicators], axis=1)
    return transformed


def assess_ood(
    values: pd.DataFrame | np.ndarray,
    preprocessor: dict[str, Any],
) -> list[dict[str, Any]]:
    names = tuple(preprocessor["feature_names"])
    matrix = _as_matrix(values, names)
    ood = preprocessor["ood"]
    lower = np.asarray(ood["raw_lower_005"], dtype=float)
    upper = np.asarray(ood["raw_upper_995"], dtype=float)
    center = np.asarray(ood["robust_center"], dtype=float)
    scale = np.asarray(ood["robust_scale"], dtype=float)
    threshold = float(ood["distance_p995"])
    results = []
    for row in matrix:
        missing = [names[index] for index in np.flatnonzero(~np.isfinite(row))]
        outside_mask = np.isfinite(row) & ((row < lower) | (row > upper))
        outside = [names[index] for index in np.flatnonzero(outside_mask)]
        filled = np.where(np.isfinite(row), row, center)
        distance = float(
            np.sqrt(np.mean(np.square(np.clip((filled - center) / scale, -50, 50))))
        )
        reasons = []
        if missing:
            reasons.append(f"missing/nonfinite features: {missing}")
        if len(outside) > 2:
            reasons.append(
                f"{len(outside)} features outside training 0.5/99.5 percentiles"
            )
        if distance > threshold:
            reasons.append(
                f"robust distance {distance:.4f} exceeds training p99.5 {threshold:.4f}"
            )
        results.append(
            {
                "withhold": bool(reasons),
                "reasons": reasons,
                "missing_features": missing,
                "outside_features": outside,
                "outside_count": len(outside),
                "robust_distance": distance,
                "distance_threshold": threshold,
            }
        )
    return results


def fit_multinomial_model(
    train_features: pd.DataFrame | np.ndarray,
    train_labels: np.ndarray,
    *,
    feature_names: Iterable[str] = FEATURE_NAMES,
    c_value: float = 0.1,
    seed: int = DEFAULT_SEED,
    class_names: Iterable[str] = CLASS_NAMES,
    model_version: str = MODEL_VERSION,
    model_type: str = "L2 multinomial logistic regression",
) -> dict[str, Any]:
    """Fit deterministic L2 multinomial logistic regression on one outer train."""
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(train_labels, dtype=int)
    names = tuple(str(name) for name in class_names)
    expected_classes = list(range(len(names)))
    if len(names) < 2 or set(np.unique(labels)) != set(expected_classes):
        raise ValueError(
            f"all configured classes {expected_classes} are required in training"
        )
    preprocessor = fit_preprocessor(train_features, feature_names=feature_names)
    transformed = transform_preprocessor(train_features, preprocessor)
    estimator = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
        tol=1e-8,
        random_state=seed,
    )
    estimator.fit(transformed, labels)
    if estimator.n_iter_.max() >= estimator.max_iter:
        raise RuntimeError("multinomial logistic regression did not converge")
    if estimator.classes_.tolist() != expected_classes:
        raise ValueError(f"unexpected class order: {estimator.classes_.tolist()}")
    model = {
        "model_version": model_version,
        "model_type": model_type,
        "class_names": list(names),
        "class_order": expected_classes,
        "c_value": float(c_value),
        "coefficient": estimator.coef_.astype(float).tolist(),
        "intercept": estimator.intercept_.astype(float).tolist(),
        "preprocessor": preprocessor,
        "training_rows": int(len(labels)),
        "seed": int(seed),
    }
    model["model_hash"] = canonical_hash(model)
    return model


def predict_logits(
    model: dict[str, Any],
    features: pd.DataFrame | np.ndarray,
    *,
    require_complete: bool = False,
) -> np.ndarray:
    expected_hash = model.get("model_hash")
    if expected_hash:
        unhashed = {key: value for key, value in model.items() if key != "model_hash"}
        if canonical_hash(unhashed) != expected_hash:
            raise ValueError("model hash mismatch")
    transformed = transform_preprocessor(
        features,
        model["preprocessor"],
        require_complete=require_complete,
    )
    coefficient = np.asarray(model["coefficient"], dtype=float)
    intercept = np.asarray(model["intercept"], dtype=float)
    class_count = len(model.get("class_order") or [])
    if (
        class_count < 2
        or coefficient.shape != (class_count, transformed.shape[1])
        or intercept.shape != (class_count,)
    ):
        raise ValueError("model coefficient shape is incompatible")
    return transformed @ coefficient.T + intercept


def predict_probabilities(
    model: dict[str, Any],
    features: pd.DataFrame | np.ndarray,
    *,
    temperature: float | None = None,
    require_complete: bool = False,
) -> np.ndarray:
    logits = predict_logits(model, features, require_complete=require_complete)
    selected_temperature = (
        float(temperature)
        if temperature is not None
        else float((model.get("temperature") or {}).get("value", 1.0))
    )
    if not math.isfinite(selected_temperature) or selected_temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return stable_softmax(logits / selected_temperature)


def multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    selected = np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(selected)))


def fit_temperature(
    calibration_logits: np.ndarray,
    calibration_labels: np.ndarray,
) -> dict[str, Any]:
    """Fit one scalar T using calibration predictions only."""
    logits = np.asarray(calibration_logits, dtype=float)
    labels = np.asarray(calibration_labels, dtype=int)
    if (
        logits.ndim != 2
        or logits.shape[1] < 2
        or len(logits) != len(labels)
        or len(labels) == 0
        or labels.min(initial=0) < 0
        or labels.max(initial=0) >= logits.shape[1]
    ):
        raise ValueError("calibration logits/labels have incompatible shape")

    def objective(log_temperature: float) -> float:
        return multiclass_log_loss(
            labels, stable_softmax(logits / math.exp(log_temperature))
        )

    grid = np.linspace(math.log(0.20), math.log(5.0), 161)
    losses = np.array([objective(value) for value in grid])
    best = int(np.argmin(losses))
    left = grid[max(0, best - 1)]
    right = grid[min(len(grid) - 1, best + 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(64):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = objective(x2)
    log_temperature = (left + right) / 2.0
    temperature = float(math.exp(log_temperature))
    return {
        "version": CALIBRATION_VERSION,
        "value": temperature,
        "fit_source": "outer_fold_calibration_only",
        "fit_rows": int(len(labels)),
        "uncalibrated_log_loss": multiclass_log_loss(labels, stable_softmax(logits)),
        "calibrated_log_loss": objective(log_temperature),
    }


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or len(p) != len(y) or p.shape[1] < 2:
        raise ValueError("Brier labels/probabilities have incompatible shape")
    one_hot = np.eye(p.shape[1], dtype=float)[y]
    return float(np.mean(np.sum(np.square(p - one_hot), axis=1)))


def _binary_calibration_fit(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    y = np.asarray(outcomes, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) < 30 or y.min() == y.max():
        return {"intercept": None, "slope": None}
    logit = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(y)), logit])
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        linear = np.clip(design @ beta, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-8)
        gradient = design.T @ (y - fitted)
        hessian = design.T @ (weights[:, None] * design)
        hessian += np.eye(2) * 1e-8
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return {"intercept": None, "slope": None}
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def classwise_reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = RELIABILITY_BINS,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    classes: dict[str, Any] = {}
    maximum_gap = 0.0
    for class_index, name in enumerate(CLASS_NAMES):
        observed = (y == class_index).astype(float)
        predicted = p[:, class_index]
        assignments = np.clip(
            np.searchsorted(edges, predicted, side="right") - 1,
            0,
            bins - 1,
        )
        rows = []
        ece = 0.0
        for bin_index in range(bins):
            mask = assignments == bin_index
            if not mask.any():
                rows.append(
                    {
                        "lower": float(edges[bin_index]),
                        "upper": float(edges[bin_index + 1]),
                        "count": 0,
                        "mean_probability": None,
                        "observed_rate": None,
                        "gap": None,
                    }
                )
                continue
            mean_probability = float(predicted[mask].mean())
            observed_rate = float(observed[mask].mean())
            gap = abs(mean_probability - observed_rate)
            maximum_gap = max(maximum_gap, gap)
            ece += mask.mean() * gap
            rows.append(
                {
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "count": int(mask.sum()),
                    "mean_probability": mean_probability,
                    "observed_rate": observed_rate,
                    "gap": float(gap),
                }
            )
        classes[name] = {
            "ece": float(ece),
            "bins": rows,
            "calibration": _binary_calibration_fit(observed, predicted),
        }
    return {"classes": classes, "maximum_gap": float(maximum_gap)}


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    climatology_probabilities: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    baseline = np.asarray(climatology_probabilities, dtype=float)
    brier = multiclass_brier(y, p)
    baseline_brier = multiclass_brier(y, baseline)
    log_loss = multiclass_log_loss(y, p)
    baseline_log_loss = multiclass_log_loss(y, baseline)
    reliability = classwise_reliability(y, p)
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
    }


def fit_regime_baseline(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    shrinkage: float = 100.0,
) -> dict[str, Any]:
    required = ("spy_price_sma200", "spy_vol_60")
    if any(name not in features for name in required):
        raise ValueError("SPY trend/volatility features are required for regime baseline")
    y = np.asarray(labels, dtype=int)
    global_counts = np.bincount(y, minlength=3).astype(float)
    global_rates = global_counts / global_counts.sum()
    volatility = features["spy_vol_60"].to_numpy(dtype=float)
    finite = volatility[np.isfinite(volatility)]
    cutoffs = np.quantile(finite, [1 / 3, 2 / 3]).astype(float)
    keys = _regime_keys(features, cutoffs)
    rates: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for key in sorted(set(keys)):
        mask = keys == key
        local = np.bincount(y[mask], minlength=3).astype(float)
        smoothed = (local + shrinkage * global_rates) / (
            local.sum() + shrinkage
        )
        rates[key] = smoothed.tolist()
        counts[key] = int(mask.sum())
    return {
        "global_rates": global_rates.tolist(),
        "volatility_terciles": cutoffs.tolist(),
        "rates": rates,
        "counts": counts,
        "shrinkage": float(shrinkage),
        "fit_scope": "outer_fold_training_only",
    }


def _regime_keys(features: pd.DataFrame, cutoffs: np.ndarray) -> np.ndarray:
    trend = np.where(
        features["spy_price_sma200"].to_numpy(dtype=float) >= 0,
        "trend_up",
        "trend_down",
    )
    volatility = features["spy_vol_60"].to_numpy(dtype=float)
    buckets = np.where(
        volatility <= cutoffs[0],
        "vol_low",
        np.where(volatility <= cutoffs[1], "vol_mid", "vol_high"),
    )
    return np.char.add(np.char.add(trend, "|"), buckets)


def predict_regime_baseline(
    baseline: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cutoffs = np.asarray(baseline["volatility_terciles"], dtype=float)
    keys = _regime_keys(features, cutoffs)
    global_rates = np.asarray(baseline["global_rates"], dtype=float)
    probabilities = np.vstack(
        [
            np.asarray(baseline["rates"].get(str(key), global_rates), dtype=float)
            for key in keys
        ]
    )
    return probabilities, keys


def _weighted_brier(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    one_hot = np.eye(3)[labels]
    values = np.sum(np.square(probabilities - one_hot), axis=1)
    return float(np.average(values, weights=weights))


def _weighted_log_loss(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    values = -np.log(
        np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0)
    )
    return float(np.average(values, weights=weights))


def two_way_cluster_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    climatology_probabilities: np.ndarray,
    dates: Iterable[Any],
    issuers: Iterable[Any],
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Bootstrap fixed OOS predictions by issuer and calendar-quarter blocks.

    Models are not refit.  Per-class mean OOS calibration residuals provide a
    documented approximation for published model-probability intervals.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    baseline = np.asarray(climatology_probabilities, dtype=float)
    date_index = pd.to_datetime(pd.Series(list(dates)))
    quarter = (
        date_index.dt.year.to_numpy(dtype=int) * 4
        + date_index.dt.quarter.to_numpy(dtype=int)
    )
    issuer = np.asarray([str(value) for value in issuers], dtype=object)
    unique_issuers, issuer_inverse = np.unique(issuer, return_inverse=True)
    unique_blocks, block_inverse = np.unique(quarter, return_inverse=True)
    if len(y) == 0 or len(unique_issuers) < 2 or len(unique_blocks) < 2:
        raise ValueError("two-way bootstrap needs observations, issuers, and blocks")
    requested = int(repetitions)
    if requested <= 0:
        raise ValueError("two-way bootstrap requires a positive requested count")
    attempt_limit = (
        requested * 3 if max_attempts is None else int(max_attempts)
    )
    if attempt_limit <= 0:
        raise ValueError("two-way bootstrap max attempts must be positive")
    rng = np.random.default_rng(seed)
    brier_skills = []
    log_skills = []
    residuals = []
    skipped = Counter()
    attempted = 0
    while len(brier_skills) < requested and attempted < attempt_limit:
        attempted += 1
        sampled_issuers = rng.integers(0, len(unique_issuers), len(unique_issuers))
        sampled_blocks = rng.integers(0, len(unique_blocks), len(unique_blocks))
        issuer_multiplicity = np.bincount(
            sampled_issuers, minlength=len(unique_issuers)
        )
        block_multiplicity = np.bincount(
            sampled_blocks, minlength=len(unique_blocks)
        )
        weights = (
            issuer_multiplicity[issuer_inverse]
            * block_multiplicity[block_inverse]
        ).astype(float)
        if weights.sum() == 0:
            skipped["zero_intersection_weight"] += 1
            continue
        model_brier = _weighted_brier(y, p, weights)
        baseline_brier = _weighted_brier(y, baseline, weights)
        model_log = _weighted_log_loss(y, p, weights)
        baseline_log = _weighted_log_loss(y, baseline, weights)
        if (
            not all(
                math.isfinite(value)
                for value in (
                    model_brier,
                    baseline_brier,
                    model_log,
                    baseline_log,
                )
            )
            or baseline_brier <= 0
            or baseline_log <= 0
        ):
            skipped["nonfinite_or_degenerate_baseline"] += 1
            continue
        brier_skill = 1.0 - model_brier / baseline_brier
        log_skill = 1.0 - model_log / baseline_log
        one_hot = np.eye(3)[y]
        residual = [
            float(np.average(one_hot[:, index] - p[:, index], weights=weights))
            for index in range(3)
        ]
        if not all(
            math.isfinite(value)
            for value in [brier_skill, log_skill, *residual]
        ):
            skipped["nonfinite_metric"] += 1
            continue
        brier_skills.append(brier_skill)
        log_skills.append(log_skill)
        residuals.append(residual)

    def interval(values: list[float]) -> list[float | None]:
        return (
            np.quantile(values, [0.025, 0.975]).tolist()
            if values
            else [None, None]
        )

    residual_array = np.asarray(residuals, dtype=float)
    completed = len(brier_skills)
    return {
        "version": BOOTSTRAP_VERSION,
        "method": (
            "fixed out-of-sample predictions; independent resampling multiplicities "
            "for issuer clusters and calendar three-month blocks; invalid draws "
            "retry deterministically up to three times the requested count; no "
            "model refit"
        ),
        "repetitions": completed,
        "requested_repetitions": requested,
        "attempted_repetitions": attempted,
        "completed_repetitions": completed,
        "skipped_repetitions": attempted - completed,
        "maximum_attempts": attempt_limit,
        "skip_reasons": dict(sorted(skipped.items())),
        "complete": completed >= requested,
        "seed": int(seed),
        "issuer_clusters": int(len(unique_issuers)),
        "calendar_quarter_blocks": int(len(unique_blocks)),
        "brier_skill_ci95": interval(brier_skills),
        "log_loss_improvement_ci95": interval(log_skills),
        "probability_error_offsets_ci95": {
            name: (
                np.quantile(
                    residual_array[:, index],
                    [0.025, 0.975],
                ).tolist()
                if completed
                else [None, None]
            )
            for index, name in enumerate(CLASS_NAMES)
        },
        "probability_interval_limitation": (
            "interval shifts approximate aggregate calibration uncertainty from "
            "fixed OOS predictions; they are not individual-return intervals"
        ),
    }


def _minimum_class_count(fold_rows: list[dict[str, Any]], key: str) -> int:
    counts = [
        min(int(row[key]["class_counts"][name]) for name in CLASS_NAMES)
        for row in fold_rows
    ]
    return min(counts) if counts else 0


def evaluate_acceptance(
    report: dict[str, Any],
    *,
    gates: dict[str, float] = STRICT_ACCEPTANCE_GATES,
) -> dict[str, Any]:
    """Apply immutable release gates and return every failed reason."""
    reasons: list[str] = []
    if report.get("publish_transform_identity_verified") is not True:
        reasons.append(
            "validated OOS probabilities do not match the serialized identity "
            "publish transform"
        )

    def minimum(field: str, gate: str, label: str) -> None:
        value = report.get(field)
        if not isinstance(value, (int, float)) or value < gates[gate]:
            reasons.append(f"{label} {value!r} is below {gates[gate]}")

    minimum("history_years", "history_years_min", "usable history years")
    minimum("fold_count", "fold_count_min", "outer test folds")
    minimum(
        "full_test_fold_count",
        "full_test_fold_count_min",
        "full untouched 12-month test folds",
    )
    minimum(
        "min_usable_train_years",
        "usable_train_years_min",
        "minimum usable train feature-date span",
    )
    minimum("issuer_count", "issuer_count_min", "issuer count")
    minimum("forecast_date_count", "forecast_date_count_min", "forecast dates")
    minimum("min_train_class_count", "train_per_class_min", "minimum train/class")
    minimum(
        "min_calibration_class_count",
        "calibration_per_class_min",
        "minimum calibration/class",
    )
    minimum("min_test_class_count", "test_per_class_min", "minimum test/class")
    minimum(
        "inference_coverage",
        "inference_coverage_min",
        "out-of-sample inference coverage",
    )
    minimum(
        "provider_success_coverage",
        "provider_success_coverage_min",
        "provider successful-issuer coverage",
    )
    minimum(
        "provider_successful_issuer_count",
        "provider_successful_issuer_count_min",
        "provider successful issuer count",
    )
    aggregate = report.get("aggregate") or {}
    if aggregate.get("brier_skill", float("-inf")) < gates["brier_skill_min"]:
        reasons.append(
            f"aggregate Brier skill {aggregate.get('brier_skill')!r} is below "
            f"{gates['brier_skill_min']}"
        )
    bootstrap = report.get("bootstrap") or {}
    required_bootstrap = int(
        report.get("fixed_oos_bootstrap_required_repetitions") or 0
    )
    if required_bootstrap:
        requested_bootstrap = int(
            bootstrap.get("requested_repetitions") or 0
        )
        completed_bootstrap = int(
            bootstrap.get("completed_repetitions")
            if bootstrap.get("completed_repetitions") is not None
            else bootstrap.get("repetitions")
            or 0
        )
        if (
            requested_bootstrap < required_bootstrap
            or completed_bootstrap < requested_bootstrap
            or completed_bootstrap < required_bootstrap
        ):
            reasons.append(
                "fixed OOS bootstrap incomplete: "
                f"requested={requested_bootstrap}, "
                f"completed={completed_bootstrap}, "
                f"release minimum={required_bootstrap}"
            )
    ci = bootstrap.get("brier_skill_ci95") or [None, None]
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not isinstance(ci[0], (int, float))
        or ci[0] <= gates["brier_skill_ci_low_strict_min"]
    ):
        reasons.append(
            "Brier skill bootstrap lower bound "
            f"{(ci[0] if ci else None)!r} is not > 0"
        )
    if (
        aggregate.get("log_loss_improvement", float("-inf"))
        < gates["log_loss_improvement_min"]
    ):
        reasons.append(
            "aggregate log-loss improvement "
            f"{aggregate.get('log_loss_improvement')!r} is below "
            f"{gates['log_loss_improvement_min']}"
        )
    for name in CLASS_NAMES:
        ece = (aggregate.get("classwise_ece") or {}).get(name)
        if not isinstance(ece, (int, float)) or ece > gates["classwise_ece_max"]:
            reasons.append(f"{name} ECE {ece!r} exceeds {gates['classwise_ece_max']}")
        calibration = (aggregate.get("calibration") or {}).get(name) or {}
        slope = calibration.get("slope")
        intercept = calibration.get("intercept")
        if (
            not isinstance(slope, (int, float))
            or not gates["calibration_slope_min"]
            <= slope
            <= gates["calibration_slope_max"]
        ):
            reasons.append(
                f"{name} calibration slope {slope!r} is outside "
                f"[{gates['calibration_slope_min']}, {gates['calibration_slope_max']}]"
            )
        if (
            not isinstance(intercept, (int, float))
            or not gates["calibration_intercept_min"]
            <= intercept
            <= gates["calibration_intercept_max"]
        ):
            reasons.append(
                f"{name} calibration intercept {intercept!r} is outside "
                f"[{gates['calibration_intercept_min']}, "
                f"{gates['calibration_intercept_max']}]"
            )
    maximum_gap = aggregate.get("maximum_gap")
    if (
        not isinstance(maximum_gap, (int, float))
        or maximum_gap > gates["maximum_gap_max"]
    ):
        reasons.append(
            f"maximum reliability gap {maximum_gap!r} exceeds "
            f"{gates['maximum_gap_max']}"
        )
    fold_rows = sorted(report.get("folds") or [], key=lambda row: row.get("fold", -1))
    materially_negative = [
        (row.get("metrics") or {}).get("brier_skill", float("-inf"))
        < MATERIAL_NEGATIVE_FOLD_SKILL
        for row in fold_rows
    ]
    if any(
        materially_negative[index] and materially_negative[index + 1]
        for index in range(max(0, len(materially_negative) - 1))
    ):
        reasons.append(
            "two consecutive outer folds have materially negative Brier skill "
            f"(< {MATERIAL_NEGATIVE_FOLD_SKILL})"
        )
    for row in (report.get("regime") or {}).get("groups", []):
        if row.get("sample_gate_passed") and (
            not isinstance(row.get("max_class_ece"), (int, float))
            or row["max_class_ece"] > gates["regime_ece_max"]
        ):
            reasons.append(
                f"regime {row.get('regime')} ECE {row.get('max_class_ece')!r} "
                f"exceeds {gates['regime_ece_max']}"
            )
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "gates": dict(gates),
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int((labels == index).sum())
        for index, name in enumerate(CLASS_NAMES)
    }


def train_walk_forward_model(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    *,
    horizon: int,
    threshold_pct: int,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    c_value: float = 0.1,
    include_oos: bool = False,
) -> dict[str, Any]:
    """Train/calibrate/evaluate one horizon-threshold model across outer folds."""
    target = label_column(horizon, threshold_pct)
    if target not in dataset:
        raise ValueError(f"dataset does not contain {target}")
    valid_target = dataset[target].notna()
    if not valid_target.all():
        raise ValueError(
            "train_walk_forward_model requires a horizon-filtered dataset"
        )
    all_probabilities = []
    all_climatology = []
    all_regime = []
    all_labels = []
    all_dates = []
    all_issuers = []
    all_regime_keys = []
    fold_rows: list[dict[str, Any]] = []
    test_candidate_count = 0
    unique_calibration_test_events = {
        index: set() for index in range(len(CLASS_NAMES))
    }
    for fold in folds:
        train = dataset.iloc[fold["train_indices"]]
        calibration = dataset.iloc[fold["calibration_indices"]]
        test = dataset.iloc[fold["test_indices"]]
        test_candidate_count += len(test)
        y_train = train[target].to_numpy(dtype=int)
        if set(np.unique(y_train)) != {0, 1, 2}:
            raise ValueError(
                f"fold {fold['fold']} train is missing an outcome class"
            )
        model = fit_multinomial_model(
            train.loc[:, FEATURE_NAMES],
            y_train,
            c_value=c_value,
            seed=seed + int(fold["fold"]),
        )
        calibration_ood = assess_ood(
            calibration.loc[:, FEATURE_NAMES],
            model["preprocessor"],
        )
        calibration_mask = np.array(
            [not item["withhold"] for item in calibration_ood],
            dtype=bool,
        )
        calibration_scored = calibration.loc[calibration_mask].copy()
        y_calibration = calibration_scored[target].to_numpy(dtype=int)
        if set(np.unique(y_calibration)) != {0, 1, 2}:
            raise ValueError(
                f"fold {fold['fold']} calibration coverage is missing an outcome class"
            )
        calibration_logits = predict_logits(
            model, calibration_scored.loc[:, FEATURE_NAMES]
        )
        temperature = fit_temperature(calibration_logits, y_calibration)
        model["temperature"] = temperature
        model.pop("model_hash", None)
        model["model_hash"] = canonical_hash(model)
        test_ood = assess_ood(
            test.loc[:, FEATURE_NAMES],
            model["preprocessor"],
        )
        test_mask = np.array(
            [not item["withhold"] for item in test_ood],
            dtype=bool,
        )
        test_scored = test.loc[test_mask].copy()
        y_test = test_scored[target].to_numpy(dtype=int)
        if set(np.unique(y_test)) != {0, 1, 2}:
            raise ValueError(
                f"fold {fold['fold']} test coverage is missing an outcome class"
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
                unique_calibration_test_events[int(class_index)].add(
                    (issuer, pd.Timestamp(feature_date).isoformat())
                )
        raw_test_probabilities = predict_probabilities(
            model, test_scored.loc[:, FEATURE_NAMES]
        )
        test_probabilities = apply_publish_transform(
            raw_test_probabilities,
            PUBLISH_TRANSFORM,
        )
        train_counts = np.bincount(y_train, minlength=3).astype(float)
        climatology_rates = train_counts / train_counts.sum()
        climatology = np.tile(climatology_rates, (len(test_scored), 1))
        regime_model = fit_regime_baseline(
            train.loc[:, FEATURE_NAMES], y_train
        )
        regime_probabilities, regime_keys = predict_regime_baseline(
            regime_model, test_scored.loc[:, FEATURE_NAMES]
        )
        metrics = evaluate_predictions(y_test, test_probabilities, climatology)
        regime_metrics = evaluate_predictions(y_test, regime_probabilities, climatology)
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "train_start": pd.Timestamp(fold["train_start"]).date().isoformat(),
                "train_end": pd.Timestamp(fold["train_end"]).date().isoformat(),
                "calibration_start": pd.Timestamp(
                    fold["calibration_start"]
                ).date().isoformat(),
                "calibration_end": pd.Timestamp(
                    fold["calibration_end"]
                ).date().isoformat(),
                "test_start": pd.Timestamp(fold["test_start"]).date().isoformat(),
                "test_end": pd.Timestamp(fold["test_end"]).date().isoformat(),
                "usable_train_years": float(fold["usable_train_years"]),
                "full_test_window": bool(fold["full_test_window"]),
                "train": {
                    "count": int(len(train)),
                    "class_counts": _class_counts(y_train),
                    "issuer_count": int(train["issuer_key"].nunique()),
                    "date_count": int(train["feature_date"].nunique()),
                },
                "calibration": {
                    "candidate_count": int(len(calibration)),
                    "count": int(len(calibration_scored)),
                    "inference_coverage": float(calibration_mask.mean()),
                    "class_counts": _class_counts(y_calibration),
                    "temperature": temperature["value"],
                    "temperature_fit_source": temperature["fit_source"],
                },
                "test": {
                    "candidate_count": int(len(test)),
                    "count": int(len(test_scored)),
                    "inference_coverage": float(test_mask.mean()),
                    "class_counts": _class_counts(y_test),
                    "issuer_count": int(test_scored["issuer_key"].nunique()),
                    "date_count": int(test_scored["feature_date"].nunique()),
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
                "publish_transform_max_abs_diff": float(
                    np.max(
                        np.abs(
                            test_probabilities - raw_test_probabilities
                        )
                    )
                ),
            }
        )
        all_probabilities.append(test_probabilities)
        all_climatology.append(climatology)
        all_regime.append(regime_probabilities)
        all_labels.append(y_test)
        all_dates.extend(test_scored["feature_date"].tolist())
        all_issuers.extend(test_scored["issuer_key"].astype(str).tolist())
        all_regime_keys.extend(regime_keys.tolist())

    if not all_labels:
        raise ValueError("no complete outer folds were trained")
    labels = np.concatenate(all_labels)
    probabilities = np.vstack(all_probabilities)
    climatology = np.vstack(all_climatology)
    regime_probabilities = np.vstack(all_regime)
    aggregate = evaluate_predictions(labels, probabilities, climatology)
    regime_aggregate = evaluate_predictions(labels, regime_probabilities, climatology)
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
    regime_key_array = np.asarray(all_regime_keys)
    for key in sorted(set(all_regime_keys)):
        mask = regime_key_array == key
        class_counts = _class_counts(labels[mask])
        sample_gate = int(mask.sum()) >= 200 and min(class_counts.values()) >= 30
        reliability = classwise_reliability(labels[mask], probabilities[mask])
        regime_groups.append(
            {
                "regime": key,
                "count": int(mask.sum()),
                "class_counts": class_counts,
                "sample_gate_passed": sample_gate,
                "classwise_ece": {
                    name: reliability["classes"][name]["ece"]
                    for name in CLASS_NAMES
                },
                "max_class_ece": max(
                    reliability["classes"][name]["ece"]
                    for name in CLASS_NAMES
                ),
            }
        )
    history_start = (
        pd.to_datetime(dataset["history_start"]).min()
        if "history_start" in dataset
        else pd.to_datetime(dataset["feature_date"]).min()
    )
    history_end = pd.to_datetime(dataset["feature_date"]).max()
    report = {
        "model_key": model_key(horizon, threshold_pct),
        "horizon_sessions": int(horizon),
        "threshold_pct": int(threshold_pct),
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "publish_transform": dict(PUBLISH_TRANSFORM),
        "validated_probability_space": PUBLISH_TRANSFORM["input"],
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
        "forecast_date_count": len(set(pd.to_datetime(all_dates).date)),
        "min_train_class_count": _minimum_class_count(fold_rows, "train"),
        "min_calibration_class_count": _minimum_class_count(
            fold_rows, "calibration"
        ),
        "min_test_class_count": _minimum_class_count(fold_rows, "test"),
        "inference_coverage": (
            len(labels) / test_candidate_count if test_candidate_count else 0.0
        ),
        "provider_success_coverage": 1.0,
        "provider_requested_issuer_count": int(dataset["issuer_key"].nunique()),
        "provider_successful_issuer_count": int(dataset["issuer_key"].nunique()),
        "provider_unavailable_symbols": {},
        "event_counts_calibration_test_unique": {
            name: len(unique_calibration_test_events[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "aggregate": aggregate,
        "regime": {
            "aggregate": regime_aggregate,
            "groups": regime_groups,
            "definition": (
                "SPY price/SMA200 sign crossed with training-fold SPY vol60 terciles; "
                "100-observation shrinkage toward training climatology"
            ),
        },
        "bootstrap": bootstrap,
        "folds": fold_rows,
    }
    if include_oos:
        report["_oos"] = {
            "keys": [
                f"{issuer}|{pd.Timestamp(date).isoformat()}"
                for issuer, date in zip(all_issuers, all_dates)
            ],
            "probabilities": probabilities,
            "labels": labels,
        }
    report["acceptance"] = evaluate_acceptance(report)
    return report


def evaluate_raw_grid_monotonicity(
    reports_by_threshold: dict[int, dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Gate an entire horizon grid; raw probabilities are never projected."""
    thresholds = sorted(reports_by_threshold)
    if len(thresholds) < 2:
        return {
            "passed": False,
            "reason": "complete horizon threshold grid is required",
            "thresholds": thresholds,
            "up_violation_count": 0,
            "down_violation_count": 0,
            "max_up_excess": None,
            "max_down_excess": None,
            "publish_transform": dict(PUBLISH_TRANSFORM),
        }
    first = reports_by_threshold[thresholds[0]].get("_oos") or {}
    keys = first.get("keys")
    if not isinstance(keys, list):
        return {
            "passed": False,
            "reason": "joint OOS predictions are unavailable",
            "thresholds": thresholds,
            "up_violation_count": 0,
            "down_violation_count": 0,
            "max_up_excess": None,
            "max_down_excess": None,
            "publish_transform": dict(PUBLISH_TRANSFORM),
        }
    matrices = {}
    for threshold in thresholds:
        oos = reports_by_threshold[threshold].get("_oos") or {}
        if oos.get("keys") != keys:
            return {
                "passed": False,
                "reason": "joint OOS row identities differ across thresholds",
                "thresholds": thresholds,
                "up_violation_count": 0,
                "down_violation_count": 0,
                "max_up_excess": None,
                "max_down_excess": None,
                "publish_transform": dict(PUBLISH_TRANSFORM),
            }
        matrix = np.asarray(oos.get("probabilities"), dtype=float)
        if matrix.shape != (len(keys), 3):
            return {
                "passed": False,
                "reason": "joint OOS probability shape is invalid",
                "thresholds": thresholds,
                "up_violation_count": 0,
                "down_violation_count": 0,
                "max_up_excess": None,
                "max_down_excess": None,
                "publish_transform": dict(PUBLISH_TRANSFORM),
            }
        matrices[threshold] = matrix
    up_excesses = []
    down_excesses = []
    pair_rows = []

    def magnitude_stats(values: np.ndarray) -> dict[str, Any]:
        violations = values[values > tolerance]
        return {
            "violation_count": int(len(violations)),
            "violation_rate": float(len(violations) / len(values)) if len(values) else 0.0,
            "mean_magnitude": float(violations.mean()) if len(violations) else 0.0,
            "p50_magnitude": float(np.quantile(violations, 0.50))
            if len(violations)
            else 0.0,
            "p95_magnitude": float(np.quantile(violations, 0.95))
            if len(violations)
            else 0.0,
            "max_magnitude": float(violations.max()) if len(violations) else 0.0,
        }

    for easier, harder in zip(thresholds, thresholds[1:]):
        up_excess = matrices[harder][:, 2] - matrices[easier][:, 2]
        down_excess = matrices[harder][:, 0] - matrices[easier][:, 0]
        up_excesses.append(up_excess)
        down_excesses.append(down_excess)
        pair_rows.append(
            {
                "easier_threshold_pct": easier,
                "harder_threshold_pct": harder,
                "up": magnitude_stats(up_excess),
                "down": magnitude_stats(down_excess),
            }
        )
    up = magnitude_stats(np.concatenate(up_excesses))
    down = magnitude_stats(np.concatenate(down_excesses))

    def direction_passed(stats: dict[str, Any]) -> bool:
        return (
            stats["violation_rate"]
            <= GRID_MONOTONICITY_GATES["violation_rate_max"]
            and stats["p95_magnitude"]
            <= GRID_MONOTONICITY_GATES["violation_p95_magnitude_max"]
            and stats["max_magnitude"]
            <= GRID_MONOTONICITY_GATES["violation_max_magnitude_max"]
        )

    passed = direction_passed(up) and direction_passed(down)
    return {
        "passed": passed,
        "reason": (
            None
            if passed
            else "raw calibrated grid monotonicity diagnostics exceed acceptance tolerances"
        ),
        "thresholds": thresholds,
        "oos_row_count": len(keys),
        "comparison_count_per_direction": len(keys) * (len(thresholds) - 1),
        "up": up,
        "down": down,
        "up_violation_count": up["violation_count"],
        "down_violation_count": down["violation_count"],
        "up_violation_rate": up["violation_rate"],
        "down_violation_rate": down["violation_rate"],
        "p95_up_excess": up["p95_magnitude"],
        "p95_down_excess": down["p95_magnitude"],
        "max_up_excess": up["max_magnitude"],
        "max_down_excess": down["max_magnitude"],
        "pairs": pair_rows,
        "tolerance": tolerance,
        "gates": dict(GRID_MONOTONICITY_GATES),
        "publish_transform": dict(PUBLISH_TRANSFORM),
        "action": (
            "withhold horizon grid only when rate/magnitude tolerances fail; "
            "raw probabilities are never projected"
        ),
    }


def apply_grid_release_gate(
    reports_by_threshold: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    result = evaluate_raw_grid_monotonicity(reports_by_threshold)
    if not result["passed"]:
        reason = result["reason"] or "horizon grid release gate failed"
        for report in reports_by_threshold.values():
            acceptance = report.setdefault(
                "acceptance",
                {"accepted": False, "reasons": [], "gates": {}},
            )
            acceptance["accepted"] = False
            if reason not in acceptance["reasons"]:
                acceptance["reasons"].append(reason)
    for report in reports_by_threshold.values():
        report["grid_monotonicity"] = result
    return result


def fit_release_model(
    dataset: pd.DataFrame,
    report: dict[str, Any],
    *,
    horizon: int,
    threshold_pct: int,
    trained_at: str | None = None,
    seed: int = DEFAULT_SEED,
    c_value: float = 0.1,
) -> dict[str, Any]:
    """Refit an accepted model with a final temporal calibration segment."""
    if not (report.get("acceptance") or {}).get("accepted"):
        raise ValueError("release model cannot be fit from a failed validation report")
    target = label_column(horizon, threshold_pct)
    dataset = dataset.loc[dataset[target].notna()].copy()
    dates = pd.to_datetime(dataset["feature_date"]).dt.tz_localize(None)
    max_exit = pd.to_datetime(dataset["max_exit_date"]).dt.tz_localize(None)
    calibration_end = dates.max() + pd.Timedelta(days=1)
    calibration_start = calibration_end - pd.DateOffset(years=1)
    train_mask = (dates < calibration_start) & (
        max_exit < calibration_start - pd.Timedelta(days=7)
    )
    calibration_mask = (dates >= calibration_start) & (dates < calibration_end)
    train = dataset.loc[train_mask]
    calibration = dataset.loc[calibration_mask]
    y_train = train[target].to_numpy(dtype=int)
    train_counts = _class_counts(y_train)
    if min(train_counts.values(), default=0) < 1000:
        raise ValueError("final release training has fewer than 1000 observations/class")
    model = fit_multinomial_model(
        train.loc[:, FEATURE_NAMES],
        y_train,
        c_value=c_value,
        seed=seed,
    )
    calibration_ood = assess_ood(
        calibration.loc[:, FEATURE_NAMES],
        model["preprocessor"],
    )
    calibration_coverage_mask = np.array(
        [not item["withhold"] for item in calibration_ood],
        dtype=bool,
    )
    calibration_scored = calibration.loc[calibration_coverage_mask].copy()
    y_calibration = calibration_scored[target].to_numpy(dtype=int)
    calibration_counts = _class_counts(y_calibration)
    if min(calibration_counts.values(), default=0) < 300:
        raise ValueError(
            "final release calibration coverage has fewer than 300 "
            "observations/class"
        )
    calibration_logits = predict_logits(
        model, calibration_scored.loc[:, FEATURE_NAMES]
    )
    temperature = fit_temperature(calibration_logits, y_calibration)
    model["temperature"] = temperature
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    climatology = np.bincount(y_train, minlength=3).astype(float)
    climatology /= climatology.sum()
    regime = fit_regime_baseline(train.loc[:, FEATURE_NAMES], y_train)
    event_counts = {
        name: int(
            (report.get("event_counts_calibration_test_unique") or {}).get(
                name, 0
            )
        )
        for name in CLASS_NAMES
    }
    model.update(
        {
            "model_key": model_key(horizon, threshold_pct),
            "horizon_sessions": int(horizon),
            "threshold_pct": int(threshold_pct),
            "round_trip_cost_bps": 30,
            "publish_transform": dict(PUBLISH_TRANSFORM),
            "grid_monotonicity": report.get("grid_monotonicity"),
            "trained_at": trained_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_cutoff": pd.Timestamp(calibration_end - pd.Timedelta(days=1))
            .date()
            .isoformat(),
            "release_train_start": pd.to_datetime(train["feature_date"]).min().date().isoformat(),
            "release_train_end": pd.to_datetime(train["feature_date"]).max().date().isoformat(),
            "release_calibration_start": pd.Timestamp(calibration_start).date().isoformat(),
            "release_calibration_end": pd.Timestamp(calibration_end).date().isoformat(),
            "release_train_class_counts": train_counts,
            "release_calibration_class_counts": calibration_counts,
            "release_calibration_candidate_count": int(len(calibration)),
            "release_calibration_coverage": float(
                calibration_coverage_mask.mean()
            ),
            "baseline_rates": {
                name: float(climatology[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "regime_baseline": regime,
            "oos_sample_size": int(report["aggregate"]["count"]),
            "oos_metrics": {
                key: report["aggregate"][key]
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
                )
            },
            "fold_count": int(report["fold_count"]),
            "full_test_fold_count": int(report["full_test_fold_count"]),
            "history_years": float(report["history_years"]),
            "min_usable_train_years": float(
                report["min_usable_train_years"]
            ),
            "bootstrap": report["bootstrap"],
            "event_counts_calibration_test": event_counts,
            "accepted": True,
            "acceptance_reasons": [],
        }
    )
    model.pop("model_hash", None)
    model["model_hash"] = canonical_hash(model)
    return model


def whole_identity_percentages(probabilities: Iterable[float]) -> list[int]:
    """Whole-number display for identity probabilities; event rounding is monotone."""
    values = np.asarray(list(probabilities), dtype=float)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or not np.isclose(values.sum(), 1.0, atol=1e-12)
    ):
        raise ValueError("display probabilities must be a three-class simplex")
    down = int(round(values[0] * 100))
    up = int(round(values[2] * 100))
    middle = 100 - down - up
    if not all(0 <= value <= 100 for value in (down, middle, up)):
        raise AssertionError("whole-number probability display is infeasible")
    return [down, middle, up]


__all__ = [
    "BOOTSTRAP_VERSION",
    "CALIBRATION_VERSION",
    "DEFAULT_BOOTSTRAP_REPS",
    "DEFAULT_SEED",
    "GRID_MONOTONICITY_GATES",
    "MODEL_VERSION",
    "PUBLISH_TRANSFORM",
    "PUBLISH_TRANSFORM_VERSION",
    "RELEASE_BOOTSTRAP_REPS",
    "STRICT_ACCEPTANCE_GATES",
    "assess_ood",
    "apply_grid_release_gate",
    "apply_publish_transform",
    "canonical_hash",
    "classwise_reliability",
    "evaluate_acceptance",
    "evaluate_predictions",
    "evaluate_raw_grid_monotonicity",
    "fit_multinomial_model",
    "fit_preprocessor",
    "fit_regime_baseline",
    "fit_release_model",
    "fit_temperature",
    "multiclass_brier",
    "multiclass_log_loss",
    "predict_logits",
    "predict_probabilities",
    "predict_regime_baseline",
    "stable_softmax",
    "train_walk_forward_model",
    "transform_preprocessor",
    "two_way_cluster_bootstrap",
    "whole_identity_percentages",
]
