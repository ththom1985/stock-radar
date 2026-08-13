"""Transparent, unvalidated research insights derived from existing v2 fields."""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

from .data_quality import (
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    REQUIRED_INSIGHT_CATEGORIES,
)
from .persistence import schema_meta, utc_now
from .rating import (
    bottoming_signal,
    bull_thesis,
    downside_analysis,
    entry_label,
    entry_reason,
    entry_score,
    falling_knife,
    plain_summary,
    priced_in_note,
    risk_warnings,
    suggest_actions,
    trend_phase,
)

INSIGHT_STATUS = "heuristic_unvalidated"
INSIGHT_CONTRACT_VERSION = 1
MAX_CATEGORY_ITEMS = 20
PROVENANCE_CATALOG = {
    "entry": ["completed-daily trend/momentum/volatility/support features"],
    "analyst": ["analyst count, consensus, target and target gap"],
    "valuation": ["current complete company value/quality/growth fields"],
    "potential": ["analyst context and unvalidated scenario ranges"],
    "risk": ["downside structure, volatility, earnings, distress and knife flags"],
    "thesis": ["complete fundamentals when available, trend, analysts and news"],
    "research": ["timing, trend, valuation, analyst, risk and news contexts"],
    "overall": ["all insight groups; core ranking remains unchanged"],
    "knife": ["5d/20d deterioration, fast averages, MACD and daily direction"],
    "bottom": ["decline plus multi-signal stabilization observations"],
    "downside": ["price supports and ATR-scaled distance"],
    "trend": ["Weinstein stage, RSI, SMA50, MACD and earnings proximity"],
    "zone": ["nearest support, ATR and current price"],
}

_GENERIC_FUNDAMENTAL_EXCLUSIONS = (
    "bank",
    "insurance",
    "reit",
    "mortgage",
    "financial services",
    "real estate",
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _technical_complete(row: dict[str, Any]) -> bool:
    return (row.get("feature_coverage") or {}).get("technical_complete") is True


def _fundamentals_complete(row: dict[str, Any]) -> bool:
    coverage = row.get("feature_coverage") or {}
    return (
        row.get("asset_type") == "company_equity"
        and coverage.get("fundamental_complete") is True
        and coverage.get("fundamental_current") is True
    )


def _generic_fundamental_excluded(row: dict[str, Any]) -> str | None:
    text = f"{row.get('sector') or ''} {row.get('industry') or ''}".lower()
    matched = next((word for word in _GENERIC_FUNDAMENTAL_EXCLUSIONS if word in text), None)
    return (
        f"generic cross-sector value comparison excluded for {matched}"
        if matched
        else None
    )


def _narrative_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    if _fundamentals_complete(row) and not _generic_fundamental_excluded(row):
        return row
    safe = dict(row)
    if not _fundamentals_complete(row):
        for key in (
            "value_score",
            "quality_score",
            "growth_score",
            "fundamental_score",
            "pe",
            "forward_pe",
            "roe_pct",
            "revenue_growth_pct",
            "earnings_growth",
            "piotroski",
            "altman_z",
        ):
            safe[key] = None
        safe["fundamental_reasons"] = []
    else:
        safe["value_score"] = None
    return safe


def _provenance(inputs: list[str], missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_status": INSIGHT_STATUS,
        "actionable": False,
        "inputs_used": inputs,
        "missing_inputs": list(missing or []),
    }


def _analyst_context(row: dict[str, Any]) -> dict[str, Any]:
    count = row.get("analyst_n")
    upside = row.get("analyst_upside_pct")
    target = row.get("target_price")
    eligible = (
        row.get("asset_type") == "company_equity"
        and isinstance(count, int)
        and count >= 5
        and _finite(upside)
        and _finite(target)
        and target > 0
    )
    missing = []
    if row.get("asset_type") != "company_equity":
        missing.append("company equity required")
    if not isinstance(count, int) or count < 5:
        missing.append("at least five analyst opinions required")
    if not _finite(upside) or not _finite(target):
        missing.append("analyst target/upside unavailable")
    return {
        **_provenance(
            ["analyst_n", "analyst_rating", "target_price", "analyst_upside_pct"],
            missing,
        ),
        "available": eligible,
        "analyst_count": count if isinstance(count, int) else None,
        "consensus": row.get("analyst_rating"),
        "target_price": target if _finite(target) else None,
        "upside_pct": upside if _finite(upside) else None,
        "note": (
            "Analyst consensus context; not a model forecast."
            if eligible
            else "Analyst context withheld because coverage is insufficient."
        ),
    }


def _valuation_context(row: dict[str, Any]) -> dict[str, Any]:
    complete = _fundamentals_complete(row)
    excluded = _generic_fundamental_excluded(row) if complete else None
    eligible = complete and excluded is None
    missing = []
    if not complete:
        missing.append("current complete company fundamentals required")
    if excluded:
        missing.append(excluded)
    reasons = list(row.get("fundamental_reasons") or []) if complete else []
    return {
        **_provenance(
            [
                "asset_type",
                "fundamental_source_status",
                "value_score",
                "quality_score",
                "growth_score",
                "fundamental_reasons",
            ],
            missing,
        ),
        "available": complete,
        "ranking_eligible": eligible,
        "unavailable_reason": (
            None
            if complete
            else "Aktuelle vollständige Unternehmensfundamentaldaten fehlen; "
            "Bewertungs- und Qualitätsaussagen werden unterdrückt."
        ),
        "value_score": row.get("value_score") if complete else None,
        "quality_score": row.get("quality_score") if complete else None,
        "growth_score": row.get("growth_score") if complete else None,
        "fundamental_score": row.get("fundamental_score") if complete else None,
        "reasons": reasons,
        "why_undervalued": [
            reason
            for reason in reasons
            if any(word in reason.lower() for word in ("günst", "niedrig", "attraktiv", "peg"))
        ],
        "comparison_note": excluded or (
            "Descriptive absolute bands; excluded from the conservative core ranking."
            if complete
            else "No valuation claim."
        ),
    }


def _potential_context(
    row: dict[str, Any],
    analyst: dict[str, Any],
) -> dict[str, Any]:
    scenarios = []
    for scenario in row.get("scenario_long") or []:
        if scenario.get("label") in {"1 Monat", "6 Monate", "12 Monate", "24 Monate"}:
            scenarios.append(
                {
                    key: scenario.get(key)
                    for key in (
                        "label",
                        "reference_change_pct",
                        "reference_price",
                        "range_low_pct",
                        "range_high_pct",
                        "range_low_price",
                        "range_high_price",
                        "model_status",
                        "interpretation",
                    )
                }
            )
    missing = [] if scenarios else ["heuristic scenario ranges unavailable"]
    if not analyst["available"]:
        missing.append("meaningful analyst coverage unavailable")
    return {
        **_provenance(
            ["scenario_long", "analyst_context", "atr_pct", "vol_daily"],
            missing,
        ),
        "analyst_target": analyst if analyst["available"] else None,
        "scenario_ranges": scenarios,
        "note": (
            "Reference paths and ranges are heuristic illustrations, not expected, "
            "median or probable outcomes."
        ),
    }


def _observation_zone(
    row: dict[str, Any],
    downside: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not downside or not _finite(downside.get("support1")):
        return None
    support = downside["support1"]
    atr = row.get("atr")
    price = row.get("price")
    if not (_finite(atr) and atr > 0 and _finite(price) and price > 0):
        return None
    return {
        **_provenance(["support1", "atr", "price"]),
        "label": "Technische Beobachtungszone",
        "lower": support,
        "upper": min(price, support + 0.5 * atr),
        "currency_display": "USD",
        "note": "Support/ATR observation only; not an entry order, target or stop.",
    }


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add insight fields without changing conservative core scores/rankings."""
    technical_ok = _technical_complete(row)
    missing_technical = [] if technical_ok else ["complete technical feature set required"]

    phase = trend_phase(row) if technical_ok else None
    if phase:
        phase = {
            **phase,
            **_provenance(
                ["weinstein_stage", "rsi", "sma50", "macd_hist", "earnings_in_days"]
            ),
        }
    row["trend_phase"] = phase

    knife_warning = falling_knife(row) if technical_ok else None
    knife = None
    if knife_warning:
        severity = min(
            100,
            max(
                abs(min(row.get("ret_5d") or 0, 0)) * 4,
                abs(min(row.get("ret_20d") or 0, 0)) * 2,
            ),
        )
        knife = {
            **_provenance(
                ["ret_5d", "ret_20d", "sma20", "ema9", "macd_hist", "daily_signal_direction"]
            ),
            "warning": knife_warning,
            "severity": round(severity, 1),
        }
    row["falling_knife"] = knife
    row["knife_warn"] = knife_warning

    bottom = bottoming_signal(row) if technical_ok else None
    if bottom:
        bottom = {
            **bottom,
            **_provenance(
                [
                    "pct_from_high52",
                    "ret_20d",
                    "ret_60d",
                    "ema9",
                    "macd_hist",
                    "stoch_k",
                    "rsi",
                    "aroon_up",
                ]
            ),
            "speculative": True,
            "note": "Spekulative Bodenbildungsbeobachtung; keine bestätigte Trendwende.",
        }
    row["bottoming"] = bottom

    downside = downside_analysis(row) if technical_ok else None
    if downside:
        downside = {
            **downside,
            **_provenance(
                ["price", "sma20", "sma50", "sma150", "sma200", "pivot_s1", "low20", "atr_pct"]
            ),
        }
    row["downside_structure"] = downside
    row["downside"] = downside

    narrative_row = _narrative_safe_row(row)
    warnings = list(risk_warnings(narrative_row)) if technical_ok else []
    if knife_warning:
        warnings.append(knife_warning)
    if downside and downside.get("risk") == "hoch":
        warnings.append("⚠️ Hoher Abstand bzw. schwache Struktur bis zur nächsten Unterstützung")
    if _finite(row.get("vol_annual_pct")) and row["vol_annual_pct"] >= 60:
        warnings.append(f"⚠️ Hohe annualisierte Schwankung ({row['vol_annual_pct']:.0f}%)")
    if _finite(row.get("atr_pct")) and row["atr_pct"] >= 5:
        warnings.append(f"⚠️ Große typische Tagesspanne (ATR {row['atr_pct']:.1f}%)")
    earnings_days = row.get("earnings_in_days")
    if isinstance(earnings_days, int) and 0 <= earnings_days <= 7:
        warnings.append(f"📅 Unternehmenszahlen in {earnings_days} Tagen: Ereignisrisiko")
    if _fundamentals_complete(row) and _finite(row.get("altman_z")) and row["altman_z"] < 1.81:
        warnings.append(f"⚠️ Kritischer Altman-Z-Wert ({row['altman_z']:.2f})")
    row["risk_warnings"] = list(dict.fromkeys(warnings))
    row["risk_context"] = {
        **_provenance(
            [
                "risk_warnings",
                "downside_structure",
                "vol_annual_pct",
                "atr_pct",
                "earnings_in_days",
                "altman_z",
            ],
            missing_technical,
        ),
        "warnings": row["risk_warnings"],
        "critical": bool(
            knife_warning
            or (downside and downside.get("risk") == "hoch")
            or (
                _fundamentals_complete(row)
                and _finite(row.get("altman_z"))
                and row["altman_z"] < 1.81
            )
        ),
    }

    timing_score = entry_score(row) if technical_ok else None
    timing_label, _tone = entry_label(timing_score)
    timing_reason = entry_reason(row) if technical_ok else ""
    row["entry_timing_score"] = timing_score
    row["entry_timing_label"] = timing_label
    row["entry_timing_reason"] = timing_reason
    row["entry_timing"] = {
        **_provenance(
            [
                "feature_coverage.technical_complete",
                "rsi",
                "sma20",
                "sma50",
                "ema9",
                "macd_hist",
                "stoch_k",
                "atr_pct",
                "daily_signal_direction",
                "longterm_score",
            ],
            missing_technical,
        ),
        "available": timing_score is not None,
        "score": timing_score,
        "label": timing_label,
        "reason": timing_reason,
    }

    analyst = _analyst_context(row)
    valuation = _valuation_context(row)
    potential = _potential_context(row, analyst)
    row["analyst_context"] = analyst
    row["valuation_context"] = valuation
    row["potential_context"] = potential
    row["technical_observation_zone"] = _observation_zone(row, downside)

    row["bull_thesis"] = bull_thesis(narrative_row)
    row["priced_in_note"] = priced_in_note(narrative_row)
    row["priced_in"] = row["priced_in_note"]
    thesis_inputs = [
        "minervini_score",
        "weinstein_stage",
        "analyst_context",
        "news_sentiment",
    ]
    if _fundamentals_complete(row):
        thesis_inputs.extend(["growth_score", "value_score", "piotroski"])
    row["thesis_context"] = {
        **_provenance(
            thesis_inputs
        ),
        "bull_thesis": row["bull_thesis"],
        "priced_in_note": row["priced_in_note"],
    }
    row["research_actions"] = suggest_actions(narrative_row)
    row["research_summary"] = plain_summary(narrative_row)
    row["plain_summary"] = row["research_summary"]
    row["research_context"] = {
        **_provenance(
            [
                "entry_timing",
                "trend_phase",
                "valuation_context",
                "analyst_context",
                "risk_context",
                "news_sentiment",
            ]
        ),
        "summary": row["research_summary"],
        "observations": row["research_actions"],
    }
    row["insight_provenance"] = {
        **_provenance(
            [
                "technical completed-daily features",
                "current complete fundamentals when available",
                "age-filtered news",
                "analyst context with coverage gate",
                "unvalidated heuristic scenarios",
            ],
            missing_technical,
        ),
        "technical_missing": missing_technical,
        "fundamental_complete_current": _fundamentals_complete(row),
        "analyst_coverage_sufficient": analyst["available"],
    }
    return row


def _item(
    row: dict[str, Any],
    score: float,
    components: dict[str, Any],
    reasons: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "symbol": row["symbol"],
        "currency": row.get("currency"),
        "asset_type": row.get("asset_type"),
        "score": round(max(0, min(100, score)), 2),
        "components": components,
        "reasons": [reason for reason in reasons if reason][:6],
        "model_status": INSIGHT_STATUS,
        "actionable": False,
        **extra,
    }


def _category(
    label: str,
    formula: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_currency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        by_currency[item.get("currency") or "UNKNOWN"].append(item)
    sorted_partitions = {
        currency: sorted(
            items,
            key=lambda item: (-item["score"], item["symbol"]),
        )[:MAX_CATEGORY_ITEMS]
        for currency, items in sorted(by_currency.items())
    }
    return {
        "label": label,
        "formula": formula,
        "partitioned_by_currency": True,
        "model_status": INSIGHT_STATUS,
        "actionable": False,
        "items_by_currency": sorted_partitions,
        "eligible_count": sum(len(items) for items in by_currency.values()),
    }


def build_insight_rankings(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {
        "daily_setups": [],
        "undervalued_quality": [],
        "analyst_potential": [],
        "entry_watchlist": [],
        "falling_knives": [],
        "bottoming_watch": [],
        "risk_watch": [],
        "quality_momentum": [],
    }
    if enabled:
        for row in rows:
            technical = _technical_complete(row)
            timing = row.get("entry_timing_score")
            trend = row.get("longterm_score")
            daily = row.get("daily_signal_score")
            direction = row.get("daily_signal_direction")
            downside = row.get("downside_structure") or {}
            warnings = row.get("risk_warnings") or []
            knife = row.get("falling_knife")
            bottom = row.get("bottoming")

            critical_warning = bool(
                knife
                or downside.get("risk") == "hoch"
                or (
                    _fundamentals_complete(row)
                    and _finite(row.get("altman_z"))
                    and row["altman_z"] < 1.81
                )
            )
            if (
                technical
                and _finite(timing)
                and timing >= 48
                and _finite(trend)
                and trend >= 55
                and direction != "NEGATIVE"
                and not critical_warning
                and not bottom
            ):
                daily_component = daily if direction == "POSITIVE" and _finite(daily) else 50
                score = 0.45 * trend + 0.35 * timing + 0.20 * daily_component
                categories["daily_setups"].append(
                    _item(
                        row,
                        score,
                        {
                            "trend": trend,
                            "entry_timing": timing,
                            "daily_context": daily_component,
                        },
                        [row.get("entry_timing_reason"), *(row.get("longterm_reasons") or [])],
                    )
                )

            valuation = row.get("valuation_context") or {}
            value, quality = valuation.get("value_score"), valuation.get("quality_score")
            if (
                valuation.get("ranking_eligible")
                and _finite(value)
                and value >= 65
                and _finite(quality)
                and quality >= 60
            ):
                categories["undervalued_quality"].append(
                    _item(
                        row,
                        0.55 * value + 0.45 * quality,
                        {"value": value, "quality": quality},
                        valuation.get("why_undervalued") or valuation.get("reasons") or [],
                    )
                )

            analyst = row.get("analyst_context") or {}
            upside = analyst.get("upside_pct")
            if analyst.get("available") and _finite(upside) and upside >= 10:
                upside_component = min(100, max(0, 50 + upside))
                trend_component = trend if _finite(trend) else 0
                timing_component = timing if _finite(timing) else 0
                penalty = 0
                flags = []
                if _finite(row.get("rsi")) and row["rsi"] >= 70:
                    penalty += 10
                    flags.append("technisch überkauft")
                if trend_component < 45:
                    penalty += 10
                    flags.append("schwacher Trend")
                score = (
                    0.55 * upside_component
                    + 0.25 * trend_component
                    + 0.20 * timing_component
                    - penalty
                )
                categories["analyst_potential"].append(
                    _item(
                        row,
                        score,
                        {
                            "analyst_upside": upside,
                            "analyst_count": analyst.get("analyst_count"),
                            "trend": trend_component,
                            "entry_timing": timing_component,
                            "penalty": penalty,
                        },
                        [
                            f"{analyst.get('analyst_count')} Analysten; Zielabstand {upside:+.1f}%",
                            *flags,
                        ],
                    )
                )

            if (
                technical
                and _finite(timing)
                and timing >= 55
                and direction != "NEGATIVE"
                and downside.get("risk") != "hoch"
                and not knife
                and not bottom
                and (row.get("trend_phase") or {}).get("tone") != "down"
            ):
                score = 0.65 * timing + 0.35 * (trend if _finite(trend) else 0)
                categories["entry_watchlist"].append(
                    _item(
                        row,
                        score,
                        {"entry_timing": timing, "trend": trend},
                        [row.get("entry_timing_reason"), downside.get("verdict")],
                    )
                )

            if knife:
                categories["falling_knives"].append(
                    _item(
                        row,
                        knife["severity"],
                        {
                            "severity": knife["severity"],
                            "ret_5d": row.get("ret_5d"),
                            "ret_20d": row.get("ret_20d"),
                        },
                        [knife["warning"]],
                        warning_only=True,
                    )
                )

            if bottom:
                categories["bottoming_watch"].append(
                    _item(
                        row,
                        bottom.get("strength") or 0,
                        {
                            "strength": bottom.get("strength"),
                            "signal_count": bottom.get("n"),
                        },
                        list(bottom.get("signals") or []),
                        speculative=True,
                    )
                )

            risk_components = {}
            severity = 0
            if warnings:
                severity += min(40, len(warnings) * 10)
                risk_components["warning_count"] = len(warnings)
            if downside.get("risk") == "hoch":
                severity += 25
                risk_components["downside_structure"] = "hoch"
            if _finite(row.get("vol_annual_pct")) and row["vol_annual_pct"] >= 60:
                severity += min(25, (row["vol_annual_pct"] - 50) / 2)
                risk_components["vol_annual_pct"] = row["vol_annual_pct"]
            if isinstance(row.get("earnings_in_days"), int) and 0 <= row["earnings_in_days"] <= 7:
                severity += 15
                risk_components["earnings_in_days"] = row["earnings_in_days"]
            if knife:
                severity += 25
            if severity:
                categories["risk_watch"].append(
                    _item(
                        row,
                        severity,
                        risk_components,
                        warnings or [downside.get("verdict")],
                        warning_only=True,
                    )
                )

            valuation = row.get("valuation_context") or {}
            quality = valuation.get("quality_score")
            if (
                valuation.get("available")
                and _finite(quality)
                and quality >= 70
                and _finite(trend)
                and trend >= 60
                and direction == "POSITIVE"
                and not knife
            ):
                categories["quality_momentum"].append(
                    _item(
                        row,
                        0.50 * quality + 0.35 * trend + 0.15 * (daily or 0),
                        {"quality": quality, "trend": trend, "daily_context": daily},
                        ["vollständige Qualitätsdaten", "positiver abgeschlossener Tagesimpuls"],
                    )
                )

    definitions = {
        "daily_setups": (
            "Setups des Tages",
            "45% trend + 35% entry timing + 20% completed-daily context; "
            "requires no falling knife/high critical structure",
        ),
        "undervalued_quality": (
            "Unterbewertete Qualität",
            "55% value + 45% quality; current complete company fundamentals only; "
            "generic banks/insurers/REITs excluded",
        ),
        "analyst_potential": (
            "Analysten-Potenzial",
            "55% normalized analyst target gap + 25% trend + 20% timing, "
            "minus visible overbought/weak-trend penalties; >=5 analysts",
        ),
        "entry_watchlist": (
            "Timing-Beobachtung",
            "65% entry timing + 35% trend; no negative daily context, falling knife "
            "or high downside structure",
        ),
        "falling_knives": (
            "Fallende Messer",
            "warning severity from dimensionless 5d/20d deterioration and missing stabilization",
        ),
        "bottoming_watch": (
            "Bodenbildung (spekulativ)",
            "multi-signal bottoming strength; explicitly speculative",
        ),
        "risk_watch": (
            "Risiko-Watch",
            "additive warning severity from downside structure, volatility, earnings and knife flags",
        ),
        "quality_momentum": (
            "Qualität + Momentum",
            "50% complete quality + 35% trend + 15% positive completed-daily context",
        ),
    }
    output = {
        "contract_version": INSIGHT_CONTRACT_VERSION,
        "model_status": INSIGHT_STATUS,
        "actionable": False,
        "recommendation_status": "research lists, not recommendations",
        "enabled": enabled,
        "blocking_reasons": list(blockers or []) if not enabled else [],
        "categories": {},
    }
    if set(definitions) != set(REQUIRED_INSIGHT_CATEGORIES):
        raise RuntimeError("Insight category definitions do not match the output contract")
    for key in REQUIRED_INSIGHT_CATEGORIES:
        label, formula = definitions[key]
        output["categories"][key] = _category(label, formula, categories[key])
    return output


def enrich_rows_and_rankings(
    rows: list[dict[str, Any]],
    *,
    rankings_enabled: bool,
    blockers: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for row in rows:
        enrich_row(row)
    return rows, build_insight_rankings(
        rows,
        enabled=rankings_enabled,
        blockers=blockers,
    )


def enrich_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Pure-provider-free migration from a v2/v3 snapshot to insight contract v3."""
    if snapshot.get("schema") != OUTPUT_SCHEMA:
        raise ValueError(f"Unsupported snapshot schema: {snapshot.get('schema')!r}")
    if snapshot.get("schema_version") not in {2, OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported source schema version: {snapshot.get('schema_version')!r}")
    if not isinstance(snapshot.get("all"), list):
        raise ValueError("Snapshot rows are missing")
    model = snapshot.get("model_status") or {}
    if model.get("validation") != "unvalidated" or model.get("actionable") is not False:
        raise ValueError("Snapshot must remain explicitly unvalidated/non-actionable")

    enriched = copy.deepcopy(snapshot)
    status = enriched.get("data_status") or {}
    enabled = status.get("status") == "ok" and status.get("data_actionable") is True
    rows, rankings = enrich_rows_and_rankings(
        enriched["all"],
        rankings_enabled=enabled,
        blockers=status.get("blocking_reasons") or [],
    )
    enriched["all"] = rows
    enriched["insight_rankings"] = rankings
    enriched["insight_metadata"] = {
        "contract_version": INSIGHT_CONTRACT_VERSION,
        "model_status": INSIGHT_STATUS,
        "actionable": False,
        "enriched_at": utc_now(),
        "core_ranking_unchanged": True,
        "scenario_ranges_used_in_core_ranking": False,
        "provenance_catalog": PROVENANCE_CATALOG,
    }
    enriched["schema_version"] = OUTPUT_SCHEMA_VERSION
    enriched["_meta"] = schema_meta(
        "stock-radar-output",
        schema_version=OUTPUT_SCHEMA_VERSION,
        insight_contract=1,
    )
    return enriched
