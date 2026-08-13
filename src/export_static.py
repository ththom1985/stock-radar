"""Export a compact, login-free GitHub Pages insight payload."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data_quality import validate_insight_contract, validate_output_contract
from .insights import INSIGHT_CONTRACT_VERSION
from .persistence import atomic_write_bytes, load_json, schema_meta
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
    sweet_spot_reason_catalog = _intern_sweet_spot_text(compact_rows)
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
        "market_data_contract": snapshot.get("market_data_contract") or {},
        "rankings": rankings,
        "insight_rankings": snapshot["insight_rankings"],
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
