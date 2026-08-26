"""Output-contract validation and conservative completed-bar quality gates."""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from statistics import median
from typing import Any

from .persistence import SCHEMA_VERSION
from .probability_inference import validate_probability_forecast
from .sweet_spot import confirmed_status_violations, green_invariant_blockers


OUTPUT_SCHEMA = "stock-radar-output"
OUTPUT_SCHEMA_VERSION = 3
INSIGHT_CONTRACT_VERSION = 3
REQUIRED_INSIGHT_CATEGORIES = (
    "in_sweet_spot",
    "approaching_sweet_spot",
    "daily_setups",
    "undervalued_quality",
    "analyst_potential",
    "entry_watchlist",
    "falling_knives",
    "bottoming_watch",
    "risk_watch",
    "quality_momentum",
)
MIN_COVERAGE_PCT = float(os.environ.get("STOCK_RADAR_MIN_COVERAGE_PCT", "97.0"))
MAX_BAR_AGE_DAYS = int(os.environ.get("STOCK_RADAR_MAX_BAR_AGE_DAYS", "4"))
MAX_OUTPUT_AGE_HOURS = int(os.environ.get("STOCK_RADAR_MAX_OUTPUT_AGE_HOURS", "36"))
MIN_RANK_COVERAGE_PCT = {
    "company_equity": float(
        os.environ.get("STOCK_RADAR_MIN_RANK_COVERAGE_COMPANY_PCT", "70")
    ),
    "etf_fund": float(
        os.environ.get("STOCK_RADAR_MIN_RANK_COVERAGE_FUND_PCT", "70")
    ),
    "crypto": float(
        os.environ.get("STOCK_RADAR_MIN_RANK_COVERAGE_CRYPTO_PCT", "70")
    ),
    "index_other": float(
        os.environ.get("STOCK_RADAR_MIN_RANK_COVERAGE_OTHER_PCT", "70")
    ),
}
MIN_COMPANY_FUNDAMENTAL_COVERAGE_PCT = float(
    os.environ.get("STOCK_RADAR_MIN_COMPANY_FUNDAMENTAL_COVERAGE_PCT", "60")
)
FORBIDDEN_RESEARCH_PHRASES = (
    "tipps des tages",
    "guter einstieg",
    "kaufen/halten",
    "attraktiver einstieg",
    "klassischer einstiegspunkt",
    "einstieg wird interessanter",
    "sofort (heute)",
    "kaufen",
    "jetzt kaufen",
    "gestaffelt kaufen",
    "kaufsetup",
    "kaufzone",
    "meiden",
    "starker kauf",
    "konsens „kauf",
    "konsens „halten",
)
_RESEARCH_LANGUAGE_FIELDS = (
    "research_summary",
    "plain_summary",
    "research_actions",
    "entry_timing_reason",
    "longterm_reasons",
    "daily_signal_reasons",
    "entry_thesis",
    "valuation_thesis",
    "trend_phase",
    "weinstein_label",
    "risk_warnings",
    "falling_knife",
    "bottoming",
    "bull_thesis",
    "priced_in_note",
    "sweet_spot",
    "trade_plan_long",
    "trade_plan_short",
    "urgency",
    "actions",
)


class DataContractError(RuntimeError):
    pass


def _has_actionable_true(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("actionable") is True:
            return True
        return any(_has_actionable_true(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_actionable_true(item) for item in value)
    return False


def _forbidden_research_phrases(row: dict[str, Any]) -> list[str]:
    text = json.dumps(
        {key: row.get(key) for key in _RESEARCH_LANGUAGE_FIELDS},
        ensure_ascii=False,
        default=str,
    ).casefold()
    return [phrase for phrase in FORBIDDEN_RESEARCH_PHRASES if phrase in text]


def _require_heuristic_group(
    value: Any,
    label: str,
    *,
    allow_descriptive_status: bool = False,
) -> None:
    status_valid = isinstance(value, dict) and (
        value.get("actionable") is False
        or (
            allow_descriptive_status
            and value.get("assessment_status") == "descriptive_only"
            and isinstance(value.get("assessment_status_label"), str)
        )
    )
    if (
        not isinstance(value, dict)
        or value.get("model_status") != "heuristic_unvalidated"
        or not status_valid
        or not isinstance(value.get("inputs_used"), list)
        or not isinstance(value.get("missing_inputs"), list)
    ):
        raise DataContractError(f"{label} insight group is invalid")


def validate_insight_contract(
    insight: Any,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(insight, dict):
        raise DataContractError("Insight ranking root must be an object")
    if (
        insight.get("contract_version") != INSIGHT_CONTRACT_VERSION
        or insight.get("model_status") != "heuristic_unvalidated"
        or insight.get("actionable") is not False
        or not isinstance(insight.get("enabled"), bool)
        or not isinstance(insight.get("blocking_reasons"), list)
        or not isinstance(insight.get("categories"), dict)
    ):
        raise DataContractError("Insight ranking contract is invalid")
    if insight["enabled"] and insight["blocking_reasons"]:
        raise DataContractError("Enabled insight rankings cannot have blocking reasons")
    if not insight["enabled"] and not insight["blocking_reasons"]:
        raise DataContractError("Disabled insight rankings require blocking reasons")
    categories = insight["categories"]
    missing = [key for key in REQUIRED_INSIGHT_CATEGORIES if key not in categories]
    if missing:
        raise DataContractError(f"Missing insight categories: {missing}")
    for key in REQUIRED_INSIGHT_CATEGORIES:
        category = categories[key]
        if (
            not isinstance(category, dict)
            or not isinstance(category.get("label"), str)
            or not isinstance(category.get("formula"), str)
            or category.get("partitioned_by_currency") is not True
            or category.get("model_status") != "heuristic_unvalidated"
            or category.get("actionable") is not False
            or not isinstance(category.get("eligible_count"), int)
            or category["eligible_count"] < 0
            or not isinstance(category.get("items_by_currency"), dict)
        ):
            raise DataContractError(f"Insight category {key!r} is invalid")
        for currency, items in category["items_by_currency"].items():
            if not isinstance(currency, str) or not isinstance(items, list):
                raise DataContractError(f"Insight category {key!r} partition is invalid")
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("symbol"), str)
                    or not isinstance(item.get("reasons"), list)
                    or not all(isinstance(reason, str) for reason in item["reasons"])
                    or not isinstance(item.get("components"), dict)
                    or not all(
                        isinstance(name, str)
                        and (
                            value is None
                            or isinstance(value, (str, bool))
                            or (
                                isinstance(value, (int, float))
                                and math.isfinite(value)
                            )
                        )
                        for name, value in item.get("components", {}).items()
                    )
                    or item.get("model_status") != "heuristic_unvalidated"
                    or item.get("actionable") is not False
                    or not isinstance(item.get("score"), (int, float))
                    or not math.isfinite(item["score"])
                    or not 0 <= item["score"] <= 100
                ):
                    raise DataContractError(
                        f"Insight category {key!r} contains an invalid item"
                    )
                if key == "undervalued_quality":
                    components = item["components"]
                    required_penalties = (
                        "raw_value_quality_score",
                        "jurisdiction_penalty",
                        "size_liquidity_penalty",
                        "cyclical_peak_penalty",
                        "shrinking_fundamentals_penalty",
                        "weak_trend_downside_penalty",
                        "total_risk_penalty",
                        "risk_adjusted_score",
                        "score_formula",
                    )
                    if any(name not in components for name in required_penalties):
                        raise DataContractError(
                            "Undervalued item is missing visible risk-adjustment components"
                        )
                    if (
                        not 0 <= components["total_risk_penalty"] <= 45
                        or components["risk_adjusted_score"] != item["score"]
                    ):
                        raise DataContractError(
                            "Undervalued item risk adjustment is invalid"
                        )
                    evidence = item.get("penalty_evidence_ids")
                    if not isinstance(evidence, dict) or any(
                        not isinstance(values, list)
                        or not all(isinstance(value, str) for value in values)
                        for values in evidence.values()
                    ):
                        raise DataContractError(
                            "Undervalued item penalty evidence is invalid"
                        )
    if _has_actionable_true(insight):
        raise DataContractError("Nested actionable=true is forbidden in insights")
    category_language = json.dumps(
        {
            key: {
                "label": value.get("label"),
                "formula": value.get("formula"),
            }
            for key, value in categories.items()
        },
        ensure_ascii=False,
    ).casefold()
    category_forbidden = [
        phrase for phrase in FORBIDDEN_RESEARCH_PHRASES if phrase in category_language
    ]
    if category_forbidden:
        raise DataContractError(
            f"Insight labels contain recommendation-oriented language: {category_forbidden}"
        )

    if rows is not None:
        row_symbols = {
            row.get("symbol") for row in rows if isinstance(row, dict)
        }
        required_groups = (
            "entry_timing",
            "entry_thesis",
            "sweet_spot",
            "analyst_context",
            "valuation_context",
            "valuation_thesis",
            "potential_context",
            "risk_context",
            "jurisdiction_risk",
            "thesis_context",
            "research_context",
            "insight_provenance",
        )
        required_fields = (
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
            "research_actions",
            "display_name_full",
            "short_name",
            "provider_country",
            "headquarters_country",
            "headquarters_country_code",
            "legal_domicile",
            "legal_domicile_code",
            "legal_domicile_verified",
            "legal_domicile_source",
            "issuer_country",
            "listing_country",
            "listing_market",
            "economic_exposure_country",
            "economic_exposure_country_code",
            "economic_exposure_region",
            "economic_exposure_classification_status",
            "jurisdiction_code",
            "sector_display",
            "industry_display",
            "identity_source",
            "identity_semantics",
            "identity_status",
        )
        for row in rows:
            if not isinstance(row, dict):
                raise DataContractError("Insight row must be an object")
            absent = [key for key in (*required_groups, *required_fields) if key not in row]
            if absent:
                raise DataContractError(
                    f"Insight row {row.get('symbol')!r} is missing fields: {absent}"
                )
            for group in required_groups:
                _require_heuristic_group(
                    row[group],
                    f"{row.get('symbol')}.{group}",
                    allow_descriptive_status=group == "valuation_context",
                )
            for optional_group in (
                "falling_knife",
                "bottoming",
                "downside_structure",
                "trend_phase",
            ):
                if row[optional_group] is not None:
                    _require_heuristic_group(
                        row[optional_group],
                        f"{row.get('symbol')}.{optional_group}",
                    )
            if not isinstance(row["research_summary"], str):
                raise DataContractError("research_summary must be text")
            if not isinstance(row["research_actions"], list) or not all(
                isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item.get("tone") in {"pos", "neg", "neutral"}
                for item in row["research_actions"]
            ):
                raise DataContractError("research_actions are invalid")
            if not isinstance(row["risk_warnings"], list) or not all(
                isinstance(item, str) for item in row["risk_warnings"]
            ):
                raise DataContractError("risk_warnings are invalid")
            if (
                not isinstance(row["display_name_full"], str)
                or not row["display_name_full"].strip()
                or row["identity_status"] not in {"complete", "partial", "fallback"}
                or not isinstance(row["identity_source"], dict)
                or not isinstance(row["identity_semantics"], dict)
                or row["economic_exposure_classification_status"]
                not in {"classified", "unclassified", "unavailable"}
                or not isinstance(row["sector_display"], str)
                or not isinstance(row["industry_display"], str)
            ):
                raise DataContractError("row identity fields are invalid")
            jurisdiction = row["jurisdiction_risk"]
            if (
                jurisdiction.get("level") not in {"low", "medium", "high", "unknown"}
                or not isinstance(jurisdiction.get("penalty_points"), (int, float))
                or not 0 <= jurisdiction["penalty_points"] <= 20
                or not isinstance(jurisdiction.get("reasons"), list)
                or not all(isinstance(reason, str) for reason in jurisdiction["reasons"])
            ):
                raise DataContractError("jurisdiction risk context is invalid")
            valuation_thesis = row["valuation_thesis"]
            evidence = valuation_thesis.get("penalty_evidence_ids")
            if not isinstance(evidence, dict) or any(
                not isinstance(values, list)
                or not all(isinstance(value, str) for value in values)
                for values in evidence.values()
            ):
                raise DataContractError("valuation penalty evidence is invalid")
            if set(evidence.get("cyclical_peak_penalty") or []) & set(
                evidence.get("shrinking_fundamentals_penalty") or []
            ):
                raise DataContractError(
                    "cyclical and shrinking penalties cannot share evidence"
                )
            for key in (
                "why_it_looks_cheap",
                "why_discount_may_be_justified",
                "strongest_positive_evidence",
                "strongest_counterarguments",
                "penalty_reasons",
            ):
                if not isinstance(valuation_thesis.get(key), list) or not all(
                    isinstance(item, str) for item in valuation_thesis[key]
                ):
                    raise DataContractError("valuation thesis narratives are invalid")
            if valuation_thesis.get("available"):
                raw = valuation_thesis.get("raw_score")
                penalty = valuation_thesis.get("risk_penalty")
                adjusted = valuation_thesis.get("risk_adjusted_score")
                if (
                    not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in (raw, penalty, adjusted)
                    )
                    or not 0 <= raw <= 100
                    or not 0 <= penalty <= 45
                    or adjusted != round(max(0.0, min(100.0, raw - penalty)), 2)
                    or valuation_thesis.get("value_trap_risk")
                    not in {"low", "medium", "high"}
                ):
                    raise DataContractError("valuation thesis scores are invalid")
            elif any(
                valuation_thesis.get(key) is not None
                for key in ("raw_score", "risk_penalty", "risk_adjusted_score")
            ):
                raise DataContractError(
                    "Unavailable valuation thesis must not expose scores"
                )
            entry_thesis = row["entry_thesis"]
            for key in (
                "why_timing_may_be_good",
                "what_confirms",
                "what_invalidates",
                "strongest_supporting_evidence",
                "strongest_counterarguments",
            ):
                if not isinstance(entry_thesis.get(key), list) or not all(
                    isinstance(item, str) for item in entry_thesis[key]
                ):
                    raise DataContractError("entry thesis narratives are invalid")
            sweet = row["sweet_spot"]
            for key in (
                "why_zone_here",
                "why_green_or_not",
                "confirmation_needed",
                "invalidation_signals",
                "investor_overlay_reasons",
            ):
                if not isinstance(sweet.get(key), list) or not all(
                    isinstance(item, str) for item in sweet[key]
                ):
                    raise DataContractError("sweet-spot narratives are invalid")
            if (
                not isinstance(sweet.get("available"), bool)
                or sweet.get("currency") != "USD"
                or sweet.get("technical_status")
                not in {
                    "unavailable",
                    "in_zone_confirmed",
                    "approaching",
                    "setup_waiting_confirmation",
                    "safety_blocked",
                    "broken_below",
                    "far_above",
                    "reference_only_far",
                }
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
                or sweet.get("tone") not in {"green", "amber", "red", "neutral"}
                or sweet.get("zone_tier")
                not in {
                    "confirmed_confluence",
                    "single_anchor",
                    "reference_only",
                    "unavailable",
                }
                or not isinstance(sweet.get("confluence_count"), int)
                or sweet["confluence_count"] < 0
                or not isinstance(sweet.get("independent_family_count"), int)
                or sweet["independent_family_count"] < 0
                or not isinstance(sweet.get("reliability_score"), (int, float))
                or not math.isfinite(sweet["reliability_score"])
                or not 0 <= sweet["reliability_score"] <= 100
                or not isinstance(sweet.get("components"), list)
                or sweet.get("current_position")
                not in {"below", "in", "above", "unavailable"}
            ):
                raise DataContractError("sweet-spot status contract is invalid")
            if any(
                not isinstance(component, dict)
                or not isinstance(component.get("label"), str)
                or component.get("source_family")
                not in {
                    "pivot",
                    "moving_average_fast",
                    "moving_average_medium",
                    "moving_average_long",
                    "price_structure",
                }
                or not isinstance(component.get("value"), (int, float))
                or not math.isfinite(component["value"])
                or component["value"] <= 0
                or not isinstance(component.get("distance_atr"), (int, float))
                or not math.isfinite(component["distance_atr"])
                or not isinstance(component.get("weight"), (int, float))
                or not math.isfinite(component["weight"])
                or component["weight"] <= 0
                for component in sweet["components"]
            ):
                raise DataContractError("sweet-spot components are invalid")
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
                raise DataContractError("sweet-spot tone/status mapping is invalid")
            if sweet["available"]:
                lower, ideal, upper = (
                    sweet.get("lower"),
                    sweet.get("ideal"),
                    sweet.get("upper"),
                )
                if (
                    not all(
                        isinstance(value, (int, float))
                        and math.isfinite(value)
                        and value > 0
                        for value in (lower, ideal, upper)
                    )
                    or not lower < ideal < upper
                    or not isinstance(sweet.get("zone_width_pct"), (int, float))
                    or sweet["zone_width_pct"] <= 0
                    or len(sweet["components"]) != sweet["confluence_count"]
                    or len(
                        {
                            component["source_family"]
                            for component in sweet["components"]
                        }
                    )
                    != sweet["independent_family_count"]
                ):
                    raise DataContractError("sweet-spot zone geometry is invalid")
                tier = sweet["zone_tier"]
                if tier == "confirmed_confluence" and sweet[
                    "independent_family_count"
                ] < 2:
                    raise DataContractError(
                        "confirmed-confluence tier needs two independent families"
                    )
                if tier in {"single_anchor", "reference_only"} and (
                    sweet["confluence_count"] != 1
                    or sweet["independent_family_count"] != 1
                    or sweet["reliability_score"] > 49
                    or sweet["combined_status"] == "in_zone_confirmed"
                ):
                    raise DataContractError("single-anchor tier safety is invalid")
                if tier == "reference_only" and sweet["combined_status"] not in {
                    "reference_only_far",
                    "safety_blocked",
                }:
                    raise DataContractError(
                        "reference-only tier has an invalid combined status"
                    )
            elif any(
                sweet.get(key) is not None for key in ("lower", "ideal", "upper")
            ):
                raise DataContractError("unavailable sweet spot exposes numeric zone")
            elif sweet.get("zone_tier") != "unavailable":
                raise DataContractError("unavailable sweet spot has invalid tier")
            expected_blockers = green_invariant_blockers(
                row,
                sweet,
                data_ready=insight["enabled"],
            )
            expected_reasons = list(
                dict.fromkeys(reason for _, reason in expected_blockers)
            )
            actual_reasons = sweet["why_green_or_not"]
            if expected_reasons:
                if actual_reasons != expected_reasons:
                    raise DataContractError(
                        "sweet-spot blockers do not match recomputed failed gates"
                    )
            elif actual_reasons != [
                "Kurs liegt in der Zone und alle anwendbaren Sicherheits-Gates sind passiert."
            ]:
                raise DataContractError("sweet-spot green explanation is inconsistent")
            if sweet.get("combined_status") == "in_zone_confirmed":
                violations = confirmed_status_violations(
                    row,
                    sweet,
                    data_ready=insight["enabled"],
                )
                if violations:
                    raise DataContractError(
                        "confirmed sweet spot violates safety gates: "
                        + "; ".join(violations)
                    )
            if _has_actionable_true(
                {key: row[key] for key in (*required_groups, *required_fields)}
            ):
                raise DataContractError("Nested row actionable=true is forbidden")
            forbidden = _forbidden_research_phrases(row)
            if forbidden:
                raise DataContractError(
                    f"Research row contains recommendation-oriented language: {forbidden}"
                )
        for category in categories.values():
            for items in category["items_by_currency"].values():
                if any(item["symbol"] not in row_symbols for item in items):
                    raise DataContractError(
                        "Insight category references an unknown instrument"
                    )
        confirmed_symbols = {
            item["symbol"]
            for items in categories["in_sweet_spot"]["items_by_currency"].values()
            for item in items
        }
        approaching_symbols = {
            item["symbol"]
            for items in categories["approaching_sweet_spot"][
                "items_by_currency"
            ].values()
            for item in items
        }
        for row in rows:
            if (
                (row.get("sweet_spot") or {}).get("zone_tier")
                == "reference_only"
                and row.get("symbol")
                in confirmed_symbols | approaching_symbols
            ):
                raise DataContractError(
                    "reference-only zone leaked into a sweet-spot category"
                )
            if row.get("symbol") in confirmed_symbols:
                violations = confirmed_status_violations(
                    row,
                    row.get("sweet_spot") or {},
                    data_ready=insight["enabled"],
                )
                if violations:
                    raise DataContractError(
                        "confirmed sweet-spot category violates row safety contract: "
                        + "; ".join(violations)
                    )
    return insight


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def build_data_status(
    *,
    universe_size: int,
    rows: list[dict[str, Any]],
    failed_symbols: dict[str, str],
    now: datetime | None = None,
    min_coverage_pct: float = MIN_COVERAGE_PCT,
    max_bar_age_days: int = MAX_BAR_AGE_DAYS,
    extra_blockers: list[str] | None = None,
    feature_coverage: dict[str, dict[str, int]] | None = None,
    min_rank_coverage_pct: dict[str, float] | None = None,
    min_company_fundamental_coverage_pct: float = MIN_COMPANY_FUNDAMENTAL_COVERAGE_PCT,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    today = now.date()
    ages: list[float] = []
    missing_bar_date: list[str] = []
    stale_symbols: list[str] = []
    future_bar_symbols: list[str] = []
    for row in rows:
        raw = row.get("bar_date")
        try:
            age = float((today - date.fromisoformat(str(raw))).days)
            ages.append(age)
            if age < 0:
                future_bar_symbols.append(str(row.get("symbol") or ""))
            if age > max_bar_age_days:
                stale_symbols.append(str(row.get("symbol") or ""))
        except (TypeError, ValueError):
            missing_bar_date.append(str(row.get("symbol") or ""))

    analyzed = len(rows)
    coverage = analyzed / universe_size * 100 if universe_size else 0.0
    fresh_count = sum(age <= max_bar_age_days for age in ages)
    fresh_pct = fresh_count / analyzed * 100 if analyzed else 0.0
    blockers = list(extra_blockers or [])
    if coverage < min_coverage_pct:
        blockers.append(
            f"price coverage {coverage:.2f}% is below the {min_coverage_pct:.2f}% SLA"
        )
    if missing_bar_date:
        blockers.append(f"{len(missing_bar_date)} rows have no completed-bar date")
    if future_bar_symbols:
        blockers.append(f"{len(future_bar_symbols)} rows have future completed-bar dates")
    if fresh_pct < min_coverage_pct:
        blockers.append(
            f"fresh completed-bar coverage {fresh_pct:.2f}% is below the {min_coverage_pct:.2f}% SLA"
        )

    minimums = min_rank_coverage_pct or MIN_RANK_COVERAGE_PCT
    feature_status = {}
    for asset_type, counts in (feature_coverage or {}).items():
        total = int(counts.get("total") or 0)
        analyzed_successfully = int(counts.get("analyzed_successfully") or 0)
        technical = int(counts.get("technical_complete") or 0)
        eligible = int(counts.get("rank_eligible") or 0)
        fundamental = int(counts.get("fundamental_complete_current") or 0)
        technical_pct = technical / total * 100 if total else 0.0
        eligible_pct = eligible / total * 100 if total else 0.0
        minimum = float(minimums.get(asset_type, 100.0))
        status = {
            **counts,
            "analyzed_successfully_pct": (
                analyzed_successfully / total * 100 if total else 0.0
            ),
            "technical_complete_pct": technical_pct,
            "rank_eligible_pct": eligible_pct,
            "minimum_rank_eligible_pct": minimum,
        }
        if total and technical_pct < minimum:
            blockers.append(
                f"{asset_type} technical coverage {technical_pct:.2f}% is below {minimum:.2f}%"
            )
        if total and eligible_pct < minimum:
            blockers.append(
                f"{asset_type} rank-eligible coverage {eligible_pct:.2f}% is below {minimum:.2f}%"
            )
        if asset_type == "company_equity":
            fundamental_pct = fundamental / total * 100 if total else 0.0
            status["fundamental_complete_current_pct"] = fundamental_pct
            status[
                "minimum_fundamental_complete_current_pct"
            ] = min_company_fundamental_coverage_pct
            if total and fundamental_pct < min_company_fundamental_coverage_pct:
                blockers.append(
                    "company fundamental descriptive coverage "
                    f"{fundamental_pct:.2f}% is below "
                    f"{min_company_fundamental_coverage_pct:.2f}%"
                )
        feature_status[asset_type] = status

    distribution = {
        "min_days": min(ages) if ages else None,
        "median_days": median(ages) if ages else None,
        "p95_days": _percentile(ages, 0.95),
        "max_days": max(ages) if ages else None,
    }
    return {
        "status": "blocked" if blockers else "ok",
        "data_actionable": not blockers,
        "actionable": False,
        "actionable_reason": "deployed model is unvalidated; output is research context only",
        "coverage_pct": coverage,
        "coverage_sla_pct": min_coverage_pct,
        "fresh_bar_coverage_pct": fresh_pct,
        "max_bar_age_days": max_bar_age_days,
        "bar_age_distribution": distribution,
        "failed_symbol_count": len(failed_symbols),
        "failed_symbols": failed_symbols,
        "stale_symbols": stale_symbols,
        "missing_bar_date_symbols": missing_bar_date,
        "future_bar_symbols": future_bar_symbols,
        "feature_coverage": feature_status,
        "blocking_reasons": blockers,
    }


def validate_output_contract(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DataContractError("Output root must be an object")
    if data.get("schema") != OUTPUT_SCHEMA:
        raise DataContractError(f"Unsupported output schema: {data.get('schema')!r}")
    if data.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise DataContractError(
            f"Unsupported schema version {data.get('schema_version')!r}; "
            f"expected {OUTPUT_SCHEMA_VERSION}"
        )
    for key, kind in (
        ("generated_at", str),
        ("data_status", dict),
        ("model_status", dict),
        ("rankings_by_currency_asset", dict),
        ("insight_rankings", dict),
        ("insight_metadata", dict),
        ("all", list),
    ):
        if not isinstance(data.get(key), kind):
            raise DataContractError(f"Output field {key!r} must be {kind.__name__}")
    status = data["data_status"]
    for key, kind in (
        ("status", str),
        ("data_actionable", bool),
        ("blocking_reasons", list),
        ("feature_coverage", dict),
    ):
        if not isinstance(status.get(key), kind):
            raise DataContractError(f"data_status field {key!r} has invalid type")
    for key in ("coverage_pct", "fresh_bar_coverage_pct"):
        value = status.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise DataContractError(f"data_status field {key!r} must be a finite percentage")
    if not isinstance(status.get("failed_symbol_count"), int):
        raise DataContractError("data_status.failed_symbol_count must be an integer")
    for asset_type, coverage in status["feature_coverage"].items():
        if not isinstance(asset_type, str) or not isinstance(coverage, dict):
            raise DataContractError("data_status.feature_coverage has invalid shape")
        for key, value in coverage.items():
            if key.endswith(("_pct", "_count")) or key in {
                "total",
                "technical_complete",
                "rank_eligible",
                "fundamental_complete_current",
                "analyzed_successfully",
            }:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise DataContractError(
                        f"feature coverage field {asset_type}.{key} must be numeric"
                    )
                if key.endswith("_pct") and not 0 <= value <= 100:
                    raise DataContractError(
                        f"feature coverage field {asset_type}.{key} is out of range"
                    )
    model = data["model_status"]
    if model.get("validation") != "unvalidated" or model.get("actionable") is not False:
        raise DataContractError(
            "Deployed output must remain explicitly unvalidated and non-actionable"
        )
    probability_validation = data.get("probability_validation")
    if probability_validation is not None:
        if (
            not isinstance(probability_validation, dict)
            or probability_validation.get("status") not in {
                "not_run",
                "running",
                "no_model_passed",
                "accepted_models_available",
                "accepted_full_grid",
                "accepted_partial_grid",
                "unavailable",
            }
            or not isinstance(
                probability_validation.get("accepted_model_count"), int
            )
            or not isinstance(
                probability_validation.get("tested_model_count"), int
            )
            or not isinstance(probability_validation.get("reasons"), list)
        ):
            raise DataContractError("Probability validation summary is invalid")
        provider = probability_validation.get("provider_coverage")
        if provider is not None and (
            not isinstance(provider, dict)
            or not isinstance(provider.get("requested_issuer_count"), int)
            or not isinstance(provider.get("successful_issuer_count"), int)
            or not isinstance(provider.get("success_coverage"), (int, float))
            or not 0 <= provider["success_coverage"] <= 1
            or not isinstance(provider.get("unavailable_symbols"), dict)
        ):
            raise DataContractError("Probability provider coverage summary is invalid")
        missing_forecasts = [
            row.get("symbol")
            for row in data["all"]
            if not isinstance(row, dict) or "probability_forecast" not in row
        ]
        if missing_forecasts:
            raise DataContractError(
                "Probability-enabled output has rows without a forecast contract: "
                f"{missing_forecasts[:10]}"
            )
    for row in data["all"]:
        if isinstance(row, dict) and "probability_forecast" in row:
            try:
                validate_probability_forecast(row["probability_forecast"])
            except (TypeError, ValueError) as exc:
                raise DataContractError(
                    f"Probability forecast for {row.get('symbol')!r} is invalid: {exc}"
                ) from exc
    validate_insight_contract(data["insight_rankings"], data["all"])
    for key in ("aschenbrenner_holdings",):
        for row in data.get(key) or []:
            if isinstance(row, dict):
                forbidden = _forbidden_research_phrases(row)
                if forbidden:
                    raise DataContractError(
                        f"{key} contains recommendation-oriented language: {forbidden}"
                    )
    insight_metadata = data["insight_metadata"]
    if (
        insight_metadata.get("model_status") != "heuristic_unvalidated"
        or insight_metadata.get("actionable") is not False
        or _has_actionable_true(insight_metadata)
    ):
        raise DataContractError("Insight metadata contract is invalid")
    sweet_metadata = insight_metadata.get("sweet_spot_contract")
    if sweet_metadata is not None and (
        not isinstance(sweet_metadata, dict)
        or sweet_metadata.get("model_status") != "heuristic_unvalidated"
        or sweet_metadata.get("actionable") is not False
        or not isinstance(sweet_metadata.get("formula"), str)
        or not isinstance(sweet_metadata.get("thresholds"), dict)
    ):
        raise DataContractError("Sweet-spot metadata contract is invalid")
    expected_insight_enabled = (
        status["status"] == "ok"
        and status["data_actionable"] is True
        and not status["blocking_reasons"]
    )
    if data["insight_rankings"]["enabled"] is not expected_insight_enabled:
        raise DataContractError(
            "Insight enabled state is inconsistent with the output data gate"
        )
    for currency, by_asset in data["rankings_by_currency_asset"].items():
        if not isinstance(currency, str) or not isinstance(by_asset, dict):
            raise DataContractError("Currency ranking partitions have invalid shape")
        if not all(
            isinstance(asset_type, str)
            and isinstance(members, list)
            and all(isinstance(member, dict) for member in members)
            for asset_type, members in by_asset.items()
        ):
            raise DataContractError("Asset ranking partitions have invalid shape")
        required_ranking_fields = (
            "display_name_full",
            "provider_country",
            "headquarters_country",
            "legal_domicile",
            "economic_exposure_country",
            "listing_country",
            "listing_market",
            "industry_display",
            "sector_display",
            "identity_semantics",
        )
        required_ranking_groups = (
            "jurisdiction_risk",
            "valuation_context",
            "valuation_thesis",
            "entry_thesis",
            "sweet_spot",
        )
        for members in by_asset.values():
            for member in members:
                absent = [
                    key
                    for key in (*required_ranking_fields, *required_ranking_groups)
                    if key not in member
                ]
                if absent:
                    raise DataContractError(
                        f"Ranking row {member.get('symbol')!r} is missing context: {absent}"
                    )
                for group in required_ranking_groups:
                    _require_heuristic_group(
                        member[group],
                        f"ranking.{member.get('symbol')}.{group}",
                        allow_descriptive_status=group == "valuation_context",
                    )
                if _has_actionable_true(member):
                    raise DataContractError(
                        "Nested ranking-row actionable=true is forbidden"
                    )
    return data


def validate_portfolio_contract(data: Any) -> dict[str, Any]:
    """Validate optional portfolio state without affecting the main dashboard."""
    if not isinstance(data, dict):
        raise DataContractError("Portfolio root must be an object")
    if data.get("schema") is None:
        if not isinstance(data.get("positions", {}), dict) or not isinstance(
            data.get("cash", 0), (int, float)
        ):
            raise DataContractError("Malformed legacy portfolio")
        return data
    if data.get("schema") != "stock-radar-paper-portfolio":
        raise DataContractError(f"Unsupported portfolio schema: {data.get('schema')!r}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError("Unsupported portfolio schema version")
    required = {
        "cash": (int, float),
        "positions": dict,
        "pending_orders": list,
        "ledger": list,
        "equity_curve": list,
    }
    for key, kind in required.items():
        if not isinstance(data.get(key), kind):
            raise DataContractError(f"Portfolio field {key!r} has invalid type")
    for symbol, position in data["positions"].items():
        if not isinstance(symbol, str) or not isinstance(position, dict):
            raise DataContractError("Portfolio positions must map symbols to objects")
        for key in ("quantity", "entry_price", "last_price"):
            if not isinstance(position.get(key), (int, float)) or position[key] <= 0:
                raise DataContractError(f"Position {symbol!r} has invalid {key}")
    for order in data["pending_orders"]:
        if not isinstance(order, dict):
            raise DataContractError("Pending orders must be objects")
        if order.get("action") not in {"BUY", "SELL"} or not all(
            isinstance(order.get(key), str)
            for key in ("order_id", "symbol", "created_at", "not_before_bar_date", "status")
        ):
            raise DataContractError("Pending order has invalid nested fields")
        numeric_key = "target_notional" if order["action"] == "BUY" else "quantity"
        if (
            not isinstance(order.get(numeric_key), (int, float))
            or order[numeric_key] <= 0
        ):
            raise DataContractError(f"Pending order has invalid {numeric_key}")
    for entry in data["ledger"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            raise DataContractError("Ledger entry has invalid nested fields")
        if entry["type"] == "FILL":
            for key in (
                "quantity",
                "raw_fill_price",
                "execution_price",
                "gross_value",
                "commission",
            ):
                value = entry.get(key)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise DataContractError(f"FILL ledger entry has invalid {key}")
            if entry["quantity"] <= 0 or entry["raw_fill_price"] <= 0 or entry[
                "execution_price"
            ] <= 0:
                raise DataContractError("FILL ledger entry has non-positive quantity/price")
    for point in data["equity_curve"]:
        if (
            not isinstance(point, dict)
            or not isinstance(point.get("date"), str)
            or not isinstance(point.get("equity"), (int, float))
        ):
            raise DataContractError("Equity-curve point has invalid nested fields")
    return data


def dashboard_gate(
    data: dict[str, Any],
    *,
    now: datetime | None = None,
    max_output_age_hours: int = MAX_OUTPUT_AGE_HOURS,
) -> tuple[bool, list[str]]:
    """Return whether research cards may be rendered and blocking reasons."""
    validate_output_contract(data)
    now = now or datetime.now(timezone.utc)
    reasons = list(data["data_status"].get("blocking_reasons") or [])
    if data["data_status"].get("status") != "ok":
        reasons.append("data_status.status is not 'ok'")
    if data["data_status"].get("data_actionable") is not True:
        reasons.append("data_status.data_actionable is not true")
    try:
        generated = datetime.fromisoformat(data["generated_at"])
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        age_h = (now - generated.astimezone(timezone.utc)).total_seconds() / 3600
        if age_h > max_output_age_hours:
            reasons.append(
                f"output is {age_h:.1f} hours old (limit {max_output_age_hours} hours)"
            )
        if age_h < -1:
            reasons.append("output timestamp is in the future")
    except ValueError:
        reasons.append("generated_at is not a valid ISO timestamp")
    # Avoid duplicate messages from intentionally redundant consistency checks.
    return not reasons, list(dict.fromkeys(reasons))
