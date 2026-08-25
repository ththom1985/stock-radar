"""Export a compact, login-free GitHub Pages insight payload."""
from __future__ import annotations

import base64
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data_quality import validate_insight_contract, validate_output_contract
from .insights import INSIGHT_CONTRACT_VERSION
from .persistence import atomic_write_bytes, load_json, schema_meta
from .probability_inference import (
    FORECAST_SCHEMA_VERSION,
    WITHHELD_MESSAGE,
    load_probability_validation_summary,
    validate_probability_forecast,
)
from .probability_contract import HORIZONS
from .probability_forward_public import (
    load_forward_validation_status,
    validate_forward_validation_status,
)
from .probability_model import PUBLISH_TRANSFORM
from .sweet_spot import FORMULA as SWEET_SPOT_FORMULA
from .sweet_spot import MODEL_STATUS as SWEET_SPOT_MODEL_STATUS
from .sweet_spot import MODEL_VERSION as SWEET_SPOT_MODEL_VERSION
from .sweet_spot import THRESHOLDS as SWEET_SPOT_THRESHOLDS
from .sweet_spot import confirmed_status_violations, green_invariant_blockers

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "output" / "latest.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data.json"
STATIC_SCHEMA_VERSION = 3
MAX_STATIC_BYTES = 10 * 1024 * 1024
TARGET_STATIC_BYTES = int(8.5 * 1024 * 1024)
_SWEET_REASON_FIELDS = (
    "why_zone_here",
    "why_green_or_not",
    "confirmation_needed",
    "invalidation_signals",
    "investor_overlay_reasons",
)
_SWEET_ROW_DEFAULTS = {
    "label": "Sweet-Spot-Beobachtungszone",
    "technical_label": "technische Einstiegsbeobachtung",
    "currency": "USD",
    "currency_status": "upstream_absolute_levels_usd",
    "reliability_label": "heuristic evidence quality, not likelihood",
    "note": "Beobachtungszone, keine Ordermarke; keine Empfehlung oder Garantie.",
}
_SWEET_GATE_EVIDENCE_FIELDS = (
    "weinstein_stage",
    "trend_phase_tone",
    "sma200",
    "macd_hist",
    "macd_hist_prev",
    "atr",
    "completed_bars_only",
    "bar_age_days",
    "technical_complete",
    "fundamental_complete_current",
    "fundamental_source_current",
    "altman_z",
    "major_counterargument",
)
_SWEET_COMPONENT_LABELS = (
    "20T-Tief",
    "EMA21",
    "Pivot S1",
    "Prior Pivot",
    "SMA20",
    "SMA50",
    "SMA150",
    "SMA200",
)
_SWEET_SOURCE_FAMILIES = (
    "moving_average_fast",
    "moving_average_long",
    "moving_average_medium",
    "pivot",
    "price_structure",
)
_SWEET_ZONE_TIERS = (
    "confirmed_confluence",
    "reference_only",
    "single_anchor",
    "unavailable",
)
_SWEET_ANCHOR_SCOPES = (
    "cluster",
    "strategic_reference",
    "tactical_reference",
)
_SWEET_ANCHOR_DISTANCE_CLASSES = (
    "far_below",
    "tactical",
)
_PROBABILITY_STATUS_CODES = ("withheld", "partial", "accepted")
_PROBABILITY_CLASS_ORDER = ("down", "middle", "up")
_EXPERT_SIGNALS = (
    "positive_setup",
    "wait_for_pullback",
    "risk_too_high",
    "insufficient_data",
)
_EXPERT_EVIDENCE_QUALITY = ("low", "medium", "high")
_EXPERT_VALUATION_VERDICTS = (
    "unavailable",
    "clearly_undervalued",
    "fair",
    "expensive",
    "overpriced",
)
_PROBABILITY_ROW_DEFAULTS = {
    "schema_version": FORECAST_SCHEMA_VERSION,
    "actionable": False,
    "separate_from_radar_score": True,
    "separate_from_insight_ranking": True,
    "separate_from_sweet_spot": True,
    "supported_partition": "USD_company_equity",
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
    "artifact_created_at": None,
    "training_cutoff": None,
    "survivorship_warning": (
        "Validation uses the currently observable eligible company universe and is "
        "subject to current-universe survivorship bias."
    ),
}

ROW_FIELDS = (
    "symbol",
    "short_name",
    "display_name_full",
    "headquarters_country",
    "legal_domicile",
    "legal_domicile_verified",
    "legal_domicile_source",
    "asset_type",
    "currency",
    "economic_exposure_country",
    "economic_exposure_region",
    "listing_market",
    "listing_country",
    "sector_display",
    "industry_display",
    "jurisdiction_risk",
    "price",
    "bar_date",
    "radar_score",
    "longterm_score",
    "daily_signal_direction",
    "entry_timing_score",
    "entry_timing_label",
    "entry_timing_reason",
    "falling_knife",
    "bottoming",
    "downside_structure",
    "risk_warnings",
    "bull_thesis",
    "priced_in_note",
    "trend_phase",
    "research_summary",
    "analyst_context",
    "valuation_context",
    "valuation_thesis",
    "entry_thesis",
    "sweet_spot",
    "scenario_long",
    "next_earnings",
    "earnings_in_days",
    "news",
    "rsi",
    "macd",
    "macd_signal",
    "ret_20d",
    "ret_60d",
    "pct_from_high52",
    "atr_pct",
    "vol_annual_pct",
    "rvol",
    "minervini_score",
    "weinstein_label",
    "ea",
)
STATIC_INSTRUMENT_CONTRACT = {
    "model_status": "heuristic_unvalidated",
    "actionable": False,
    "group_provenance": "insight_metadata.provenance_catalog",
    "omitted_redundant_fields": [
        "per-row model_status/actionable/inputs_used/missing_inputs",
        "identity compatibility aliases and ISO/source metadata",
        "non-rendered context groups and feature internals",
    ],
}


def _contains_actionable_true(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("actionable") is True or any(
            _contains_actionable_true(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_actionable_true(item) for item in value)
    return False


def _compact_group(
    value: Any,
    keys: tuple[str, ...],
) -> Any:
    if not isinstance(value, dict):
        return value
    return {key: value.get(key) for key in keys}


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {key: row.get(key) for key in ROW_FIELDS}
    compact["probability_forecast"] = row.get("probability_forecast")
    expert = row.get("expert_analysis")
    if isinstance(expert, dict):
        long_term = expert.get("long_term") or {}
        short_term = expert.get("short_term") or {}
        valuation = expert.get("valuation") or {}
        fair_range = valuation.get("fair_value_range") or {}
        compact["ea"] = [
            long_term.get("score"),
            long_term.get("coverage_pct"),
            short_term.get("score"),
            short_term.get("coverage_pct"),
            (
                _EXPERT_SIGNALS.index(expert.get("signal"))
                if expert.get("signal") in _EXPERT_SIGNALS
                else len(_EXPERT_SIGNALS) - 1
            ),
            (
                _EXPERT_EVIDENCE_QUALITY.index(expert.get("evidence_quality"))
                if expert.get("evidence_quality") in _EXPERT_EVIDENCE_QUALITY
                else 0
            ),
            (
                _EXPERT_VALUATION_VERDICTS.index(valuation.get("verdict"))
                if valuation.get("verdict") in _EXPERT_VALUATION_VERDICTS
                else 0
            ),
            fair_range.get("lower"),
            fair_range.get("upper"),
        ]
    compact["news"] = (compact.get("news") or [])[:3]
    compact["scenario_long"] = [
        {
            key: scenario.get(key)
            for key in (
                "label",
                "reference_change_pct",
                "range_low_price",
                "range_high_price",
            )
        }
        for scenario in (compact.get("scenario_long") or [])[:4]
    ]
    compact["risk_warnings"] = (compact.get("risk_warnings") or [])[:6]
    compact["analyst_context"] = _compact_group(
        compact.get("analyst_context"),
        ("available", "analyst_count", "consensus", "target_price", "upside_pct"),
    )
    compact["valuation_context"] = _compact_group(
        compact.get("valuation_context"),
        (
            "available",
            "unavailable_reason",
            "value_score",
            "quality_score",
            "growth_score",
            "fundamental_score",
            "reasons",
            "comparison_note",
        ),
    )
    compact["valuation_thesis"] = _compact_group(
        compact.get("valuation_thesis"),
        (
            "available",
            "why_it_looks_cheap",
            "why_discount_may_be_justified",
            "strongest_positive_evidence",
            "strongest_counterarguments",
            "raw_score",
            "risk_penalty",
            "risk_adjusted_score",
            "value_trap_risk",
            "penalty_components",
            "penalty_evidence_ids",
            "penalty_reasons",
            "formula",
        ),
    )
    compact["entry_thesis"] = _compact_group(
        compact.get("entry_thesis"),
        (
            "available",
            "why_timing_may_be_good",
            "what_confirms",
            "what_invalidates",
            "strongest_supporting_evidence",
            "strongest_counterarguments",
            "timing_score",
            "trend",
            "regime",
            "falling_knife_bottoming_status",
        ),
    )
    compact["jurisdiction_risk"] = _compact_group(
        compact.get("jurisdiction_risk"),
        (
            "level",
            "penalty_points",
            "reasons",
            "heuristic_note",
        ),
    )
    compact["falling_knife"] = _compact_group(
        compact.get("falling_knife"),
        ("warning", "severity"),
    )
    compact["bottoming"] = _compact_group(
        compact.get("bottoming"),
        ("strength", "n", "signals", "speculative", "note"),
    )
    compact["downside_structure"] = _compact_group(
        compact.get("downside_structure"),
        ("support1", "support1_pct", "risk", "verdict"),
    )
    compact["trend_phase"] = _compact_group(
        compact.get("trend_phase"),
        ("phase",),
    )
    compact["sweet_spot"] = copy.deepcopy(
        _compact_group(
            compact.get("sweet_spot"),
            (
            "available",
            "label",
            "technical_label",
            "currency",
            "currency_status",
            "lower",
            "ideal",
            "upper",
            "current_price",
            "current_distance_pct",
            "distance_to_zone_pct",
            "current_position",
            "zone_tier",
            "anchor_scope",
            "anchor_distance_class",
            "confluence_count",
            "independent_family_count",
            "components",
            "zone_width_pct",
            "zone_width_atr",
            "nearest_support",
            "invalidation_reference",
            "technical_status",
            "combined_status",
            "tone",
            "reliability_score",
            "reliability_label",
            "why_zone_here",
            "why_green_or_not",
            "confirmation_needed",
            "invalidation_signals",
            "investor_overlay_status",
            "investor_overlay_reasons",
            "valuation_alignment",
                "note",
            ),
        )
    )
    coverage = row.get("feature_coverage") or {}
    source_status = row.get("fundamental_source_status") or {}
    compact["sweet_spot"]["gate_evidence"] = [
        row.get("weinstein_stage"),
        (row.get("trend_phase") or {}).get("tone"),
        row.get("sma200"),
        row.get("macd_hist"),
        row.get("macd_hist_prev"),
        row.get("atr"),
        row.get("completed_bars_only"),
        row.get("bar_age_days"),
        coverage.get("technical_complete") is True,
        (
            coverage.get("fundamental_complete") is True
            and coverage.get("fundamental_current") is True
        ),
        source_status.get("status") in {None, "current"},
        row.get("altman_z"),
        row.get("major_counterargument"),
    ]
    return compact


def _default_probability_forecast(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FORECAST_SCHEMA_VERSION,
        "status": "withheld",
        "message": WITHHELD_MESSAGE,
        "actionable": False,
        "separate_from_radar_score": True,
        "separate_from_insight_ranking": True,
        "separate_from_sweet_spot": True,
        "listing_currency": row.get("currency"),
        "supported_partition": "USD_company_equity",
        "signal_timestamp": row.get("bar_date"),
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
        "reasons": ["Probability engine not present in the source snapshot."],
        "ood": [],
        "baselines": [],
        "forecasts": [],
        "artifact_created_at": None,
        "training_cutoff": None,
        "survivorship_warning": (
            "Validation uses the currently observable eligible company universe and "
            "is subject to current-universe survivorship bias."
        ),
    }


def _pack_probability_octet(values: list[int]) -> bytes:
    if len(values) != 8 or any(
        not isinstance(value, int) or not 0 <= value <= 100 for value in values
    ):
        raise ValueError("Static probability values must be eight integers in 0..100")
    output = bytearray()
    buffer = 0
    bit_count = 0
    for value in values:
        buffer |= value << bit_count
        bit_count += 7
        while bit_count >= 8:
            output.append(buffer & 0xFF)
            buffer >>= 8
            bit_count -= 8
    if bit_count:
        output.append(buffer & 0xFF)
    if len(output) != 7:
        raise AssertionError("Static probability bit packing must produce seven bytes")
    return bytes(output)


def _unpack_probability_octet(value: bytes) -> list[int]:
    if len(value) != 7:
        raise ValueError("Static probability bit payload must contain seven bytes")
    output = []
    buffer = 0
    bit_count = 0
    cursor = 0
    while len(output) < 8:
        while bit_count < 7:
            buffer |= value[cursor] << bit_count
            cursor += 1
            bit_count += 8
        output.append(buffer & 0x7F)
        buffer >>= 7
        bit_count -= 7
    if any(item > 100 for item in output):
        raise ValueError("Static probability bit payload contains out-of-range values")
    return output


def _compact_probability_data(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Intern model/baseline metadata; retain only row-varying integer forecasts."""
    reason_values: set[str] = set()
    models: dict[str, dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        forecast = row.get("probability_forecast")
        if not isinstance(forecast, dict):
            forecast = _default_probability_forecast(row)
        validate_probability_forecast(forecast)
        normalized.append(forecast)
        reason_values.update(
            reason for reason in forecast.get("reasons") or [] if isinstance(reason, str)
        )
        for item in forecast.get("forecasts") or []:
            key = item["model_key"]
            metadata = {
                field: item.get(field)
                for field in (
                    "model_key",
                    "artifact_model_key",
                    "model_family",
                    "publish_transform",
                    "horizon_sessions",
                    "horizon_label",
                    "threshold_pct",
                    "sample_size",
                    "baseline_rates_pct",
                    "brier_skill",
                    "log_loss_improvement",
                    "classwise_ece",
                    "maximum_gap",
                    "fixed_oos_bootstrap",
                    "fold_count",
                    "full_test_fold_count",
                    "history_years",
                    "min_usable_train_years",
                    "artifact_trained_at",
                    "training_cutoff",
                )
            }
            if key in models and models[key] != metadata:
                raise ValueError(f"Inconsistent static probability model metadata: {key}")
            models[key] = metadata
        for item in forecast.get("baselines") or []:
            key = item["model_key"]
            metadata = {
                field: item.get(field)
                for field in (
                    "model_key",
                    "model_family",
                    "source_model_key",
                    "horizon_sessions",
                    "horizon_label",
                    "threshold_pct",
                    "rates_pct",
                    "sample_size",
                    "accepted_stock_specific_model",
                    "brier_skill",
                    "classwise_ece",
                    "fold_count",
                    "full_test_fold_count",
                    "history_years",
                    "min_usable_train_years",
                )
            }
            if key in baselines and baselines[key] != metadata:
                raise ValueError(f"Inconsistent static probability baseline: {key}")
            baselines[key] = metadata
    reason_catalog = sorted(reason_values)
    model_catalog = [models[key] for key in sorted(models)]
    baseline_catalog = [baselines[key] for key in sorted(baselines)]
    reason_refs = {value: index for index, value in enumerate(reason_catalog)}
    model_refs = {
        value["model_key"]: index for index, value in enumerate(model_catalog)
    }
    for row, forecast in zip(rows, normalized):
        forecast_items = sorted(
            forecast.get("forecasts") or [],
            key=lambda item: model_refs[item["model_key"]],
        )
        compact_forecasts = bytearray()
        if forecast_items:
            if len(model_catalog) > 16:
                raise ValueError("Static probability model catalog exceeds 16 models")
            mask = 0
            outside_count = 0
            distance_ratio_pct = 0
            tolerance_horizon_mask = 0
            compact_forecasts.extend((0, 0, 0, 0, 0))
        for item in forecast_items:
            probability = item["probabilities_pct"]
            intervals = item["model_interval_95_pct"]
            ood = item.get("ood") or {}
            distance = ood.get("robust_distance")
            threshold = ood.get("distance_threshold")
            item_distance_ratio_pct = (
                int(round(float(distance) / float(threshold) * 100))
                if isinstance(distance, (int, float))
                and isinstance(threshold, (int, float))
                and threshold > 0
                else 0
            )
            model_ref = model_refs[item["model_key"]]
            mask |= 1 << model_ref
            if (item.get("threshold_monotonicity") or {}).get(
                "tolerated_independent_threshold_inversion"
            ):
                tolerance_horizon_mask |= 1 << HORIZONS.index(
                    int(item["horizon_sessions"])
                )
            outside_count = max(
                outside_count, min(255, int(ood.get("outside_count") or 0))
            )
            distance_ratio_pct = max(
                distance_ratio_pct,
                min(255, max(0, item_distance_ratio_pct)),
            )
            compact_forecasts.extend(
                _pack_probability_octet(
                    [
                    probability["down"],
                    probability["up"],
                    intervals["down"][0],
                    intervals["down"][1],
                    intervals["middle"][0],
                    intervals["middle"][1],
                    intervals["up"][0],
                    intervals["up"][1],
                    ]
                )
            )
        if forecast_items:
            compact_forecasts[0] = mask & 0xFF
            compact_forecasts[1] = (mask >> 8) & 0xFF
            compact_forecasts[2] = outside_count
            compact_forecasts[3] = distance_ratio_pct
            compact_forecasts[4] = tolerance_horizon_mask
        row["pf"] = [
            _PROBABILITY_STATUS_CODES.index(forecast["status"]),
            [reason_refs[reason] for reason in forecast.get("reasons") or []],
            base64.b64encode(bytes(compact_forecasts)).decode("ascii"),
        ]
        row.pop("probability_forecast", None)
    return reason_catalog, model_catalog, baseline_catalog


def _hydrate_compact_probability(
    value: Any,
    *,
    reason_catalog: list[str],
    model_catalog: list[dict[str, Any]],
    baseline_catalog: list[dict[str, Any]],
    contract: dict[str, Any],
    listing_currency: Any,
    signal_timestamp: Any,
) -> dict[str, Any]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not isinstance(value[0], int)
        or not 0 <= value[0] < len(_PROBABILITY_STATUS_CODES)
        or not isinstance(value[1], list)
        or not isinstance(value[2], str)
    ):
        raise ValueError("Static compact probability forecast is invalid")
    reasons = [reason_catalog[index] for index in value[1]]
    try:
        packed = base64.b64decode(value[2], validate=True)
    except ValueError as exc:
        raise ValueError("Static probability byte payload is invalid") from exc
    if packed and len(packed) < 5:
        raise ValueError("Static probability byte payload is missing its header")
    mask = (packed[1] << 8 | packed[0]) if packed else 0
    selected_refs = [
        index for index in range(len(model_catalog)) if mask & (1 << index)
    ]
    if packed and len(packed) != 5 + 7 * len(selected_refs):
        raise ValueError("Static probability byte payload has invalid stride")
    forecasts = []
    for position, model_ref in enumerate(selected_refs):
        offset = 5 + position * 7
        (
            down,
            up,
            down_low,
            down_high,
            middle_low,
            middle_high,
            up_low,
            up_high,
        ) = _unpack_probability_octet(packed[offset : offset + 7])
        metadata = copy.deepcopy(model_catalog[model_ref])
        tolerated = bool(
            packed[4]
            & (1 << HORIZONS.index(int(metadata["horizon_sessions"])))
        )
        probabilities = {
            "down": down,
            "middle": 100 - down - up,
            "up": up,
        }
        intervals = {
            "down": [down_low, down_high],
            "middle": [middle_low, middle_high],
            "up": [up_low, up_high],
        }
        metadata.update(
            {
                "definition": (
                    f"gross UP >= +{metadata['threshold_pct'] + 0.30:.2f}%; "
                    f"gross DOWN <= -{metadata['threshold_pct'] + 0.30:.2f}%; "
                    "otherwise MIDDLE"
                ),
                "publish_transform": metadata.get("publish_transform")
                or contract["publish_transform"],
                "model_interval_method": contract["model_interval_scope"],
                "threshold_monotonicity": {
                    "permitted": True,
                    "reason_code": None,
                    "tolerated_independent_threshold_inversion": (
                        tolerated
                    ),
                    "disclosure": (
                        "Independent-threshold tolerance applied: raw inversion is "
                        "at most 0.5 percentage point, whole-percent display is "
                        "monotonic, and raw probabilities are unchanged."
                        if tolerated
                        else (
                            "Exact ordered-distribution tail sums; monotonic by "
                            "construction within float64 machine epsilon."
                        )
                        if metadata.get("model_family")
                        == "ordered-vector-v1"
                        else "Raw independent-threshold probabilities are monotonic."
                    ),
                    "action": "never project; withhold horizon when not permitted",
                },
                "probabilities": {
                    name: probabilities[name] / 100.0
                    for name in _PROBABILITY_CLASS_ORDER
                },
                "probabilities_pct": probabilities,
                "sum_pct": sum(probabilities.values()),
                "model_interval_95_pct": intervals,
                "listing_currency": listing_currency,
                "signal_timestamp": signal_timestamp,
                "entry_assumption": _PROBABILITY_ROW_DEFAULTS[
                    "entry_assumption"
                ],
                "exit_assumption": (
                    f"adjusted close {metadata.get('horizon_sessions')} sessions after t"
                ),
                "cost_assumption_bps_round_trip": 30,
                "artifact_created_at": None,
                "survivorship_warning": _PROBABILITY_ROW_DEFAULTS[
                    "survivorship_warning"
                ],
                "ood": {
                    "withhold": False,
                    "reasons": [],
                    "missing_features": [],
                    "outside_features": [],
                    "outside_count": packed[2],
                    "robust_distance_ratio": packed[3] / 100.0,
                },
                "acceptance_passed": True,
                "withholding_reasons": [],
            }
        )
        forecasts.append(metadata)
    status = _PROBABILITY_STATUS_CODES[value[0]]
    hydrated = {
        **contract["row_defaults"],
        "status": status,
        "message": (
            WITHHELD_MESSAGE
            if status == "withheld"
            else "Strictly validated calibrated material-move probabilities"
            if status == "accepted"
            else "Only the listed horizon/threshold models passed all current gates"
        ),
        "listing_currency": listing_currency,
        "signal_timestamp": signal_timestamp,
        "reasons": reasons,
        "ood": [],
        "baselines": copy.deepcopy(baseline_catalog),
        "forecasts": forecasts,
    }
    validate_probability_forecast(hydrated)
    return hydrated


def _intern_sweet_spot_text(rows: list[dict[str, Any]]) -> list[str]:
    """Intern repeated visible sweet-spot prose without dropping UI reasons."""
    counts: Counter[str] = Counter()
    for row in rows:
        sweet = row.get("sweet_spot") or {}
        for field in _SWEET_REASON_FIELDS:
            counts.update(
                item for item in (sweet.get(field) or []) if isinstance(item, str)
            )
        alignment_note = (sweet.get("valuation_alignment") or {}).get("note")
        if isinstance(alignment_note, str):
            counts[alignment_note] += 1
    catalog = sorted(text for text, count in counts.items() if count >= 2)
    references = {text: index for index, text in enumerate(catalog)}
    label_references = {
        label: index for index, label in enumerate(_SWEET_COMPONENT_LABELS)
    }
    family_references = {
        family: index for index, family in enumerate(_SWEET_SOURCE_FAMILIES)
    }
    for row in rows:
        sweet = row.get("sweet_spot") or {}
        for field in _SWEET_REASON_FIELDS:
            sweet[field] = [
                references.get(item, item) for item in (sweet.get(field) or [])
            ]
        alignment = sweet.get("valuation_alignment") or {}
        note = alignment.pop("note", None)
        if note is not None:
            alignment["note_ref"] = references.get(note, note)
        compact_components = []
        for component in sweet.get("components") or []:
            label = component["label"]
            family = component["source_family"]
            if label not in label_references:
                raise ValueError("Unknown compact sweet-spot component label")
            if family not in family_references:
                raise ValueError("Unknown compact sweet-spot source family")
            compact_components.append(
                [
                    label_references[label],
                    family_references[family],
                    component["value"],
                    component["distance_atr"],
                    component["weight"],
                ]
            )
        sweet["components"] = compact_components
        nearest = sweet.get("nearest_support")
        if isinstance(nearest, dict):
            sweet["nearest_support"] = [
                label_references[nearest["label"]],
                family_references[nearest["source_family"]],
                nearest["value"],
                nearest["distance_atr"],
            ]
        tier = sweet.pop("zone_tier")
        sweet["zone_tier_ref"] = _SWEET_ZONE_TIERS.index(tier)
        scope = sweet.pop("anchor_scope")
        distance_class = sweet.pop("anchor_distance_class")
        sweet["anchor_scope_ref"] = (
            _SWEET_ANCHOR_SCOPES.index(scope) if scope is not None else None
        )
        sweet["anchor_distance_class_ref"] = (
            _SWEET_ANCHOR_DISTANCE_CLASSES.index(distance_class)
            if distance_class is not None
            else None
        )
        for key in (
            *_SWEET_ROW_DEFAULTS,
            "zone_width_atr",
            "zone_width_pct",
            "current_price",
            "current_distance_pct",
            "distance_to_zone_pct",
            "confluence_count",
            "available",
            "current_position",
            "independent_family_count",
            "tone",
            "investor_overlay_status",
        ):
            sweet.pop(key, None)
        invalidation = sweet.get("invalidation_reference")
        if isinstance(invalidation, dict):
            invalidation.pop("basis", None)
            invalidation.pop("technical_observation_only", None)
    return catalog


def _hydrate_compact_sweet(
    value: Any,
    *,
    reason_catalog: list[str],
    current_price: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Static sweet-spot group is missing")
    sweet = copy.deepcopy(value)
    for key, default in _SWEET_ROW_DEFAULTS.items():
        sweet.setdefault(key, default)
    for field in _SWEET_REASON_FIELDS:
        hydrated: list[str] = []
        for item in sweet.get(field) or []:
            if isinstance(item, int) and not isinstance(item, bool):
                if not 0 <= item < len(reason_catalog):
                    raise ValueError("Static sweet-spot reason reference is invalid")
                hydrated.append(reason_catalog[item])
            elif isinstance(item, str):
                hydrated.append(item)
            else:
                raise ValueError("Static sweet-spot reason is invalid")
        sweet[field] = hydrated
    alignment = sweet.get("valuation_alignment") or {}
    note_ref = alignment.pop("note_ref", None)
    if note_ref is not None:
        alignment["note"] = (
            reason_catalog[note_ref]
            if isinstance(note_ref, int) and not isinstance(note_ref, bool)
            else note_ref
        )
    hydrated_components = []
    for component in sweet.get("components") or []:
        try:
            label_ref, family_ref, level, distance_atr, weight = component
            hydrated_components.append(
                {
                    "label": _SWEET_COMPONENT_LABELS[label_ref],
                    "source_family": _SWEET_SOURCE_FAMILIES[family_ref],
                    "value": level,
                    "distance_atr": distance_atr,
                    "weight": weight,
                }
            )
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("Static sweet-spot component reference is invalid") from exc
        except ValueError as exc:
            raise ValueError("Static sweet-spot component shape is invalid") from exc
    sweet["components"] = hydrated_components
    nearest = sweet.get("nearest_support")
    if isinstance(nearest, list):
        try:
            label_ref, family_ref, level, distance_atr = nearest
            sweet["nearest_support"] = {
                "label": _SWEET_COMPONENT_LABELS[label_ref],
                "source_family": _SWEET_SOURCE_FAMILIES[family_ref],
                "value": level,
                "distance_atr": distance_atr,
            }
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("Static nearest-support reference is invalid") from exc
        except ValueError as exc:
            raise ValueError("Static nearest-support shape is invalid") from exc
    try:
        sweet["zone_tier"] = _SWEET_ZONE_TIERS[sweet.pop("zone_tier_ref")]
        scope_ref = sweet.pop("anchor_scope_ref")
        distance_ref = sweet.pop("anchor_distance_class_ref")
        sweet["anchor_scope"] = (
            _SWEET_ANCHOR_SCOPES[scope_ref] if scope_ref is not None else None
        )
        sweet["anchor_distance_class"] = (
            _SWEET_ANCHOR_DISTANCE_CLASSES[distance_ref]
            if distance_ref is not None
            else None
        )
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("Static sweet-spot tier reference is invalid") from exc
    sweet["current_price"] = current_price
    sweet["confluence_count"] = len(hydrated_components)
    sweet["independent_family_count"] = len(
        {component["source_family"] for component in hydrated_components}
    )
    sweet["available"] = sweet["zone_tier"] != "unavailable"
    sweet["tone"] = {
        "in_zone_confirmed": "green",
        "in_zone_risk_filtered": "amber",
        "approaching": "amber",
        "setup_waiting_confirmation": "amber",
        "safety_blocked": "red",
        "broken_below": "red",
        "far_above": "neutral",
        "reference_only_far": "neutral",
        "unavailable": "neutral",
    }.get(sweet.get("combined_status"))
    lower, ideal, upper = sweet.get("lower"), sweet.get("ideal"), sweet.get("upper")
    if all(isinstance(item, (int, float)) for item in (lower, ideal, upper)):
        sweet["current_position"] = (
            "below"
            if current_price < lower
            else "above"
            if current_price > upper
            else "in"
        )
        sweet["zone_width_pct"] = (upper - lower) / ideal * 100.0
        if isinstance(current_price, (int, float)):
            sweet["current_distance_pct"] = (current_price / ideal - 1.0) * 100.0
            sweet["distance_to_zone_pct"] = (
                (current_price / upper - 1.0) * 100.0
                if current_price > upper
                else (current_price / lower - 1.0) * 100.0
                if current_price < lower
                else 0.0
            )
    else:
        sweet["current_position"] = "unavailable"
    return sweet


def _static_validation_row(
    row: dict[str, Any],
    sweet: dict[str, Any],
) -> dict[str, Any]:
    evidence = sweet.get("gate_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(
        _SWEET_GATE_EVIDENCE_FIELDS
    ):
        raise ValueError("Static sweet-spot gate evidence is invalid")
    gate = dict(zip(_SWEET_GATE_EVIDENCE_FIELDS, evidence))
    hydrated = dict(row)
    hydrated.update(
        {
            key: gate[key]
            for key in _SWEET_GATE_EVIDENCE_FIELDS
            if key != "trend_phase_tone"
        }
    )
    phase = dict(row.get("trend_phase") or {})
    phase["tone"] = gate["trend_phase_tone"]
    hydrated["trend_phase"] = phase
    hydrated["sweet_spot"] = sweet
    return hydrated


def validate_static_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Static payload root must be an object")
    if payload.get("schema") != "stock-radar-static":
        raise ValueError("Unsupported static payload schema")
    if payload.get("schema_version") != STATIC_SCHEMA_VERSION:
        raise ValueError("Unsupported static payload version")
    if not isinstance(payload.get("instruments"), list):
        raise ValueError("Static payload instruments must be a list")
    if not isinstance(payload.get("rankings"), dict):
        raise ValueError("Static payload core rankings must be an object")
    expert_layer = payload.get("expert_layer")
    if expert_layer is not None and (
        not isinstance(expert_layer, dict)
        or expert_layer.get("model_status") != "heuristic_unvalidated"
        or expert_layer.get("actionable") is not False
        or _contains_actionable_true(expert_layer)
    ):
        raise ValueError("Static expert layer is invalid")
    contract = payload.get("instrument_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("model_status") != "heuristic_unvalidated"
        or contract.get("actionable") is not False
        or contract.get("group_provenance")
        != "insight_metadata.provenance_catalog"
        or _contains_actionable_true(contract)
    ):
        raise ValueError("Static instrument contract is invalid")
    sweet_contract = payload.get("sweet_spot_contract")
    reason_catalog = payload.get("sweet_spot_reason_catalog")
    if (
        not isinstance(sweet_contract, dict)
        or sweet_contract.get("model_status") != "heuristic_unvalidated"
        or sweet_contract.get("actionable") is not False
        or not isinstance(sweet_contract.get("formula"), str)
        or not isinstance(sweet_contract.get("thresholds"), dict)
        or sweet_contract.get("row_defaults") != _SWEET_ROW_DEFAULTS
        or sweet_contract.get("reason_catalog_fields")
        != list(_SWEET_REASON_FIELDS)
        or sweet_contract.get("gate_evidence_fields")
        != list(_SWEET_GATE_EVIDENCE_FIELDS)
        or sweet_contract.get("component_labels")
        != list(_SWEET_COMPONENT_LABELS)
        or sweet_contract.get("source_families")
        != list(_SWEET_SOURCE_FAMILIES)
        or sweet_contract.get("component_fields")
        != ["label_ref", "source_family_ref", "value", "distance_atr", "weight"]
        or sweet_contract.get("nearest_support_fields")
        != ["label_ref", "source_family_ref", "value", "distance_atr"]
        or sweet_contract.get("zone_tiers") != list(_SWEET_ZONE_TIERS)
        or sweet_contract.get("anchor_scopes") != list(_SWEET_ANCHOR_SCOPES)
        or sweet_contract.get("anchor_distance_classes")
        != list(_SWEET_ANCHOR_DISTANCE_CLASSES)
        or sweet_contract.get("invalidation_reference_basis")
        != "zone lower minus 0.35 ATR; technical observation only"
        or not isinstance(reason_catalog, list)
        or not all(isinstance(reason, str) for reason in reason_catalog)
        or _contains_actionable_true(sweet_contract)
    ):
        raise ValueError("Static sweet-spot contract is invalid")
    probability_contract = payload.get("probability_contract")
    probability_reason_catalog = payload.get("probability_reason_catalog")
    probability_model_catalog = payload.get("probability_model_catalog")
    probability_baseline_catalog = payload.get("probability_baseline_catalog")
    probability_validation = payload.get("probability_validation")
    forward_validation_status = payload.get("forward_validation_status")
    if (
        not isinstance(probability_contract, dict)
        or probability_contract.get("schema_version") != FORECAST_SCHEMA_VERSION
        or probability_contract.get("status_codes")
        != list(_PROBABILITY_STATUS_CODES)
        or probability_contract.get("class_order")
        != list(_PROBABILITY_CLASS_ORDER)
        or probability_contract.get("row_storage_key") != "pf"
        or probability_contract.get("publish_transform") != PUBLISH_TRANSFORM
        or probability_contract.get("row_defaults")
        != _PROBABILITY_ROW_DEFAULTS
        or probability_contract.get("forecast_fields")
        != [
            "down_pct",
            "up_pct",
            "down_interval_low",
            "down_interval_high",
            "middle_interval_low",
            "middle_interval_high",
            "up_interval_low",
            "up_interval_high",
        ]
        or probability_contract.get("forecast_header_fields")
        != [
            "model_mask_low",
            "model_mask_high",
            "ood_outside_count",
            "ood_distance_ratio_pct",
            "independent_threshold_tolerance_horizon_mask",
        ]
        or not isinstance(probability_reason_catalog, list)
        or not all(
            isinstance(reason, str) for reason in probability_reason_catalog
        )
        or not isinstance(probability_model_catalog, list)
        or not all(isinstance(model, dict) for model in probability_model_catalog)
        or not isinstance(probability_baseline_catalog, list)
        or not all(
            isinstance(baseline, dict)
            for baseline in probability_baseline_catalog
        )
        or not isinstance(probability_validation, dict)
    ):
        raise ValueError("Static probability contract is invalid")
    try:
        validate_forward_validation_status(forward_validation_status)
    except ValueError as exc:
        raise ValueError("Static forward-validation status is invalid") from exc

    def require_text_lists(group: dict[str, Any], keys: tuple[str, ...]) -> None:
        for key in keys:
            values = group.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError(f"Static group field {key!r} must be a text list")

    static_data_ready = bool(
        (payload.get("insight_rankings") or {}).get("enabled") is True
        and (payload.get("data_status") or {}).get("status") == "ok"
        and (payload.get("data_status") or {}).get("data_actionable") is True
        and not (payload.get("data_status") or {}).get("blocking_reasons")
    )
    validation_rows_by_symbol: dict[str, dict[str, Any]] = {}
    for row in payload["instruments"]:
        if not isinstance(row, dict) or any(key not in row for key in ROW_FIELDS):
            raise ValueError("Static payload instrument cockpit contract is invalid")
        if (
            not isinstance(row.get("display_name_full"), str)
            or not row["display_name_full"].strip()
            or not isinstance(row.get("sector_display"), str)
            or not isinstance(row.get("industry_display"), str)
        ):
            raise ValueError("Static payload instrument identity is invalid")
        if _contains_actionable_true(row):
            raise ValueError("Static payload instrument contains actionable=true")
        _hydrate_compact_probability(
            row.get("pf"),
            reason_catalog=probability_reason_catalog,
            model_catalog=probability_model_catalog,
            baseline_catalog=probability_baseline_catalog,
            contract=probability_contract,
            listing_currency=row.get("currency"),
            signal_timestamp=row.get("bar_date"),
        )
        jurisdiction = row.get("jurisdiction_risk")
        if (
            not isinstance(jurisdiction, dict)
            or jurisdiction.get("level") not in {"low", "medium", "high", "unknown"}
            or not isinstance(jurisdiction.get("penalty_points"), (int, float))
            or not 0 <= jurisdiction["penalty_points"] <= 20
            or not isinstance(jurisdiction.get("reasons"), list)
            or not all(
                isinstance(reason, str) for reason in jurisdiction["reasons"]
            )
        ):
            raise ValueError("Static jurisdiction context is invalid")
        valuation = row.get("valuation_context")
        if not isinstance(valuation, dict) or not isinstance(
            valuation.get("available"), bool
        ):
            raise ValueError("Static valuation context is invalid")
        valuation_thesis = row.get("valuation_thesis")
        if not isinstance(valuation_thesis, dict) or not isinstance(
            valuation_thesis.get("available"), bool
        ):
            raise ValueError("Static valuation thesis is invalid")
        require_text_lists(
            valuation_thesis,
            (
                "why_it_looks_cheap",
                "why_discount_may_be_justified",
                "strongest_positive_evidence",
                "strongest_counterarguments",
                "penalty_reasons",
            ),
        )
        evidence = valuation_thesis.get("penalty_evidence_ids")
        if not isinstance(evidence, dict) or any(
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            for values in evidence.values()
        ):
            raise ValueError("Static valuation evidence is invalid")
        entry = row.get("entry_thesis")
        if not isinstance(entry, dict) or not isinstance(entry.get("available"), bool):
            raise ValueError("Static entry thesis is invalid")
        require_text_lists(
            entry,
            (
                "why_timing_may_be_good",
                "what_confirms",
                "what_invalidates",
                "strongest_supporting_evidence",
                "strongest_counterarguments",
            ),
        )
        sweet = _hydrate_compact_sweet(
            row.get("sweet_spot"),
            reason_catalog=reason_catalog,
            current_price=row.get("price"),
        )
        validation_row = _static_validation_row(row, sweet)
        validation_rows_by_symbol[str(row.get("symbol") or "")] = validation_row
        if (
            not isinstance(sweet, dict)
            or not isinstance(sweet.get("available"), bool)
            or sweet.get("tone") not in {"green", "amber", "red", "neutral"}
            or sweet.get("combined_status")
            not in {
                "unavailable",
                "in_zone_confirmed",
                "in_zone_risk_filtered",
                "approaching",
                "setup_waiting_confirmation",
                "safety_blocked",
                "broken_below",
                "far_above",
                "reference_only_far",
            }
            or sweet.get("zone_tier")
            not in {
                "confirmed_confluence",
                "single_anchor",
                "reference_only",
                "unavailable",
            }
            or not isinstance(sweet.get("components"), list)
            or not isinstance(sweet.get("independent_family_count"), int)
            or sweet["independent_family_count"] < 0
            or not isinstance(sweet.get("reliability_score"), (int, float))
            or not 0 <= sweet["reliability_score"] <= 100
        ):
            raise ValueError("Static sweet-spot status is invalid")
        if any(
            not isinstance(component, dict)
            or not isinstance(component.get("label"), str)
            or component.get("source_family") not in _SWEET_SOURCE_FAMILIES
            or not isinstance(component.get("value"), (int, float))
            or not isinstance(component.get("distance_atr"), (int, float))
            or not isinstance(component.get("weight"), (int, float))
            for component in sweet["components"]
        ):
            raise ValueError("Static sweet-spot components are invalid")
        expected_tone = {
            "in_zone_confirmed": "green",
            "in_zone_risk_filtered": "amber",
            "approaching": "amber",
            "setup_waiting_confirmation": "amber",
            "safety_blocked": "red",
            "broken_below": "red",
            "far_above": "neutral",
            "reference_only_far": "neutral",
            "unavailable": "neutral",
        }[sweet["combined_status"]]
        if sweet["tone"] != expected_tone:
            raise ValueError("Static sweet-spot tone/status mapping is invalid")
        if sweet["available"]:
            lower, ideal, upper = (
                sweet.get("lower"),
                sweet.get("ideal"),
                sweet.get("upper"),
            )
            if (
                not all(
                    isinstance(value, (int, float))
                    and value > 0
                    for value in (lower, ideal, upper)
                )
                or not lower < ideal < upper
                or len(sweet["components"]) != sweet.get("confluence_count")
                or len(
                    {
                        component["source_family"]
                        for component in sweet["components"]
                    }
                )
                != sweet["independent_family_count"]
            ):
                raise ValueError("Static sweet-spot geometry is invalid")
            tier = sweet["zone_tier"]
            if tier == "confirmed_confluence" and sweet[
                "independent_family_count"
            ] < 2:
                raise ValueError(
                    "Static confirmed-confluence tier lacks independent families"
                )
            if tier in {"single_anchor", "reference_only"} and (
                sweet["confluence_count"] != 1
                or sweet["independent_family_count"] != 1
                or sweet["reliability_score"] > 49
                or sweet["combined_status"] == "in_zone_confirmed"
            ):
                raise ValueError("Static single-anchor tier safety is invalid")
            if tier == "reference_only" and sweet["combined_status"] not in {
                "reference_only_far",
                "safety_blocked",
            }:
                raise ValueError("Static reference-only status is invalid")
        elif sweet.get("zone_tier") != "unavailable":
            raise ValueError("Static unavailable sweet spot has invalid tier")
        expected_blockers = green_invariant_blockers(
            validation_row,
            sweet,
            data_ready=static_data_ready,
        )
        expected_reasons = list(
            dict.fromkeys(reason for _, reason in expected_blockers)
        )
        if expected_reasons:
            if sweet["why_green_or_not"] != expected_reasons:
                raise ValueError(
                    "Static sweet-spot blockers do not match recomputed gates"
                )
        elif sweet["why_green_or_not"] != [
            "Kurs liegt in der Zone und alle anwendbaren Sicherheits-Gates sind passiert."
        ]:
            raise ValueError("Static sweet-spot green explanation is inconsistent")
        if sweet.get("combined_status") == "in_zone_confirmed":
            violations = confirmed_status_violations(
                validation_row,
                sweet,
                data_ready=static_data_ready,
            )
            if violations:
                raise ValueError(
                    "Static confirmed sweet spot violates safety gates: "
                    + "; ".join(violations)
                )
        analyst = row.get("analyst_context")
        if not isinstance(analyst, dict) or not isinstance(
            analyst.get("available"), bool
        ):
            raise ValueError("Static analyst context is invalid")
        if not isinstance(row.get("scenario_long"), list) or len(
            row["scenario_long"]
        ) > 4:
            raise ValueError("Static scenarios are invalid")
        if not isinstance(row.get("news"), list) or len(row["news"]) > 3:
            raise ValueError("Static news context is invalid")
    instrument_symbols = {
        row.get("symbol") for row in payload["instruments"] if isinstance(row, dict)
    }
    for by_asset in payload["rankings"].values():
        if not isinstance(by_asset, dict):
            raise ValueError("Static payload ranking partition is invalid")
        for symbols in by_asset.values():
            if (
                not isinstance(symbols, list)
                or any(not isinstance(symbol, str) for symbol in symbols)
                or any(symbol not in instrument_symbols for symbol in symbols)
            ):
                raise ValueError(
                    "Static ranking references missing/inconsistent instrument context"
                )
    try:
        validate_insight_contract(payload.get("insight_rankings"), None)
    except Exception as exc:
        raise ValueError(f"Static payload insight contract is invalid: {exc}") from exc
    metadata = payload.get("insight_metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_status") != "heuristic_unvalidated"
        or metadata.get("actionable") is not False
        or _contains_actionable_true(metadata)
    ):
        raise ValueError("Static payload insight metadata is invalid")
    for category in payload["insight_rankings"]["categories"].values():
        for items in category["items_by_currency"].values():
            if any(item.get("symbol") not in instrument_symbols for item in items):
                raise ValueError(
                    "Static insight ranking references missing instrument"
                )
    for items in payload["insight_rankings"]["categories"]["in_sweet_spot"][
        "items_by_currency"
    ].values():
        for item in items:
            validation_row = validation_rows_by_symbol[item["symbol"]]
            violations = confirmed_status_violations(
                validation_row,
                validation_row["sweet_spot"],
                data_ready=static_data_ready,
            )
            if violations:
                raise ValueError(
                    "Static confirmed category violates safety gates: "
                    + "; ".join(violations)
                )
    for category_key in ("in_sweet_spot", "approaching_sweet_spot"):
        for items in payload["insight_rankings"]["categories"][category_key][
            "items_by_currency"
        ].values():
            for item in items:
                if (
                    validation_rows_by_symbol[item["symbol"]]["sweet_spot"].get(
                        "zone_tier"
                    )
                    == "reference_only"
                ):
                    raise ValueError(
                        "Static reference-only zone leaked into sweet category"
                    )
    return payload


def export_static(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    snapshot = load_json(input_path, required=True, expected_type=dict)
    validate_output_contract(snapshot)

    rankings = {}
    for currency, by_asset in snapshot["rankings_by_currency_asset"].items():
        rankings[currency] = {
            asset_type: [
                row.get("symbol")
                for row in rows
                if isinstance(row, dict) and row.get("symbol")
            ]
            for asset_type, rows in by_asset.items()
            if isinstance(rows, list)
        }

    compact_rows = [
        _compact_row(row)
        for row in snapshot["all"]
        if isinstance(row, dict) and row.get("symbol")
    ]
    shared_probability_baselines = list(
        snapshot.get("probability_baselines") or []
    )
    if shared_probability_baselines:
        for row in compact_rows:
            forecast = row.get("probability_forecast")
            if (
                isinstance(forecast, dict)
                and not forecast.get("forecasts")
                and not forecast.get("baselines")
                and row.get("asset_type") == "company_equity"
                and row.get("currency") == "USD"
            ):
                forecast["baselines"] = shared_probability_baselines
    sweet_spot_reason_catalog = _intern_sweet_spot_text(compact_rows)
    (
        probability_reason_catalog,
        probability_model_catalog,
        probability_baseline_catalog,
    ) = _compact_probability_data(compact_rows)
    probability_validation = (
        snapshot.get("probability_validation")
        or load_probability_validation_summary()
    )
    forward_validation_status = (
        snapshot.get("forward_validation_status")
        or load_forward_validation_status()
    )
    validate_forward_validation_status(forward_validation_status)
    payload = {
        "_meta": schema_meta(
            "stock-radar-static-export",
            insight_contract=INSIGHT_CONTRACT_VERSION,
        ),
        "schema": "stock-radar-static",
        "schema_version": STATIC_SCHEMA_VERSION,
        "generated_at": snapshot["generated_at"],
        "data_status": snapshot["data_status"],
        "model_status": snapshot["model_status"],
        "insight_metadata": snapshot["insight_metadata"],
        "instrument_contract": STATIC_INSTRUMENT_CONTRACT,
        "sweet_spot_contract": {
            "model_status": SWEET_SPOT_MODEL_STATUS,
            "model_version": SWEET_SPOT_MODEL_VERSION,
            "actionable": False,
            "formula": SWEET_SPOT_FORMULA,
            "thresholds": SWEET_SPOT_THRESHOLDS,
            "row_defaults": _SWEET_ROW_DEFAULTS,
            "reason_catalog_fields": list(_SWEET_REASON_FIELDS),
            "gate_evidence_fields": list(_SWEET_GATE_EVIDENCE_FIELDS),
            "component_labels": list(_SWEET_COMPONENT_LABELS),
            "source_families": list(_SWEET_SOURCE_FAMILIES),
            "component_fields": [
                "label_ref",
                "source_family_ref",
                "value",
                "distance_atr",
                "weight",
            ],
            "nearest_support_fields": [
                "label_ref",
                "source_family_ref",
                "value",
                "distance_atr",
            ],
            "zone_tiers": list(_SWEET_ZONE_TIERS),
            "anchor_scopes": list(_SWEET_ANCHOR_SCOPES),
            "anchor_distance_classes": list(_SWEET_ANCHOR_DISTANCE_CLASSES),
            "derived_on_hydration": [
                "current_price",
                "current_distance_pct",
                "distance_to_zone_pct",
                "zone_width_pct",
                "confluence_count",
                "available",
                "current_position",
                "independent_family_count",
                "tone",
            ],
            "invalidation_reference_basis": (
                "zone lower minus 0.35 ATR; technical observation only"
            ),
            "provenance": (
                "Completed-daily USD-normalized absolute levels; repeated per-row "
                "formula/provenance is compacted here while visible reasons remain per row."
            ),
        },
        "sweet_spot_reason_catalog": sweet_spot_reason_catalog,
        "probability_contract": {
            "schema_version": FORECAST_SCHEMA_VERSION,
            "status_codes": list(_PROBABILITY_STATUS_CODES),
            "class_order": list(_PROBABILITY_CLASS_ORDER),
            "publish_transform": PUBLISH_TRANSFORM,
            "row_fields": ["status_ref", "reason_refs", "forecast_bytes_b64"],
            "row_storage_key": "pf",
            "probability_precision": (
                "whole-percent presentation quantization of exact raw probabilities "
                "retained in data/output/latest.json"
            ),
            "forecast_fields": [
                "down_pct",
                "up_pct",
                "down_interval_low",
                "down_interval_high",
                "middle_interval_low",
                "middle_interval_high",
                "up_interval_low",
                "up_interval_high",
            ],
            "forecast_header_fields": [
                "model_mask_low",
                "model_mask_high",
                "ood_outside_count",
                "ood_distance_ratio_pct",
                "independent_threshold_tolerance_horizon_mask",
            ],
            "forecast_encoding": (
                "base64 bytes: 16-bit model mask, shared conservative OOD summary, "
                "four-bit tolerated-inversion horizon mask, "
                "then seven-bit packed down/up plus six directly quantized validated "
                "interval bounds in model-catalog order; middle=100-down-up"
            ),
            "row_defaults": _PROBABILITY_ROW_DEFAULTS,
            "model_interval_scope": (
                "95% aggregate calibration-error interval approximation from "
                "fixed OOS predictions; not an individual stock outcome interval"
            ),
            "ranking_separation": (
                "probability fields are excluded from radar scores, insight "
                "rankings, Sweet Spot, and colors"
            ),
        },
        "probability_reason_catalog": probability_reason_catalog,
        "probability_model_catalog": probability_model_catalog,
        "probability_baseline_catalog": probability_baseline_catalog,
        "probability_validation": probability_validation,
        "forward_validation_status": forward_validation_status,
        "market_data_contract": snapshot.get("market_data_contract") or {},
        "rankings": rankings,
        "insight_rankings": snapshot["insight_rankings"],
        "today": snapshot.get("today") or {
            "headline": "Heute keine belastbare Aussage",
            "summary": "Der Heute-Presenter ist im Snapshot nicht verfügbar.",
            "candidate_count": 0,
            "candidates": [],
            "changes": [],
        },
        "expert_layer": snapshot.get("expert_layer") or {
            "model_status": "heuristic_unvalidated",
            "actionable": False,
            "core_ranking_unchanged": True,
            "rankings": {"long_term": [], "short_term": []},
            "recommendation_journal": {},
        },
        "instruments": compact_rows,
    }
    validate_static_payload(payload)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    encoded_size = len(encoded)
    if encoded_size > TARGET_STATIC_BYTES:
        raise ValueError(
            f"Static payload is {encoded_size / 1024 / 1024:.2f} MiB; "
            "must remain at or below the 8.5 MiB publication target"
        )
    atomic_write_bytes(output_path, encoded)
    return payload


def main() -> None:
    payload = export_static()
    size_mb = DEFAULT_OUTPUT.stat().st_size / 1024 / 1024
    print(
        f"Static dashboard export: {len(payload['instruments'])} instruments, "
        f"{size_mb:.2f} MiB -> {DEFAULT_OUTPUT}"
    )


if __name__ == "__main__":
    main()
