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
from .identity import enrich_identity
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
INSIGHT_CONTRACT_VERSION = 2
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
    "identity": ["provider identity, configured short name and listing metadata"],
    "jurisdiction": ["issuer domicile, listing venue and conservative exposure overrides"],
    "valuation_thesis": ["raw value/quality score and bounded visible risk penalties"],
    "entry_thesis": ["completed-daily momentum, trend, support, ATR and event context"],
}

_GENERIC_FUNDAMENTAL_EXCLUSIONS = (
    "bank",
    "insurance",
    "reit",
    "mortgage",
    "financial services",
    "real estate",
)
_LEGACY_LANGUAGE_REPLACEMENTS = (
    ("Phase 2 – Aufwärtstrend (kaufen/halten)", "Phase 2 – Aufwärtstrend / technisch konstruktiv"),
    ("Phase 4 – Abwärtstrend (meiden)", "Phase 4 – Abwärtstrend / negative Struktur"),
    ("Phase 3 – Topbildung (Gewinne sichern)", "Phase 3 – Topbildungsrisiko"),
    ("attraktiver Einstieg statt überkauft", "konstruktive Timing-Beobachtung"),
    (
        "Kurs testet die 50-Tage-Linie im Aufwärtstrend – klassischer Einstiegspunkt",
        "Test der 50-Tage-Linie im Aufwärtstrend",
    ),
    ("schlechter Langfrist-Einstieg", "ungünstiger längerfristiger Timing-Kontext"),
    ("Einstieg wird interessanter", "Timing verbessert sich"),
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _neutralize_legacy_language(value: Any) -> Any:
    if isinstance(value, str):
        for old, new in _LEGACY_LANGUAGE_REPLACEMENTS:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_neutralize_legacy_language(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _neutralize_legacy_language(item)
            for key, item in value.items()
        }
    return value


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
    consensus_raw = row.get("analyst_rating")
    consensus = {
        "strong_buy": "Strong Buy-Konsens",
        "buy": "Buy-Konsens",
        "hold": "Hold-Konsens",
        "underperform": "Underperform-Konsens",
        "sell": "Sell-Konsens",
        "none": "Kein Analystenkonsens",
    }.get(consensus_raw, consensus_raw)
    return {
        **_provenance(
            ["analyst_n", "analyst_rating", "target_price", "analyst_upside_pct"],
            missing,
        ),
        "available": eligible,
        "analyst_count": count if isinstance(count, int) else None,
        "consensus": consensus,
        "consensus_raw": consensus_raw,
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


def _pct_from_ratio(value: Any) -> float | None:
    return value * 100 if _finite(value) else None


def _valuation_penalties(
    row: dict[str, Any],
) -> tuple[dict[str, float], list[str], dict[str, list[str]]]:
    jurisdiction = row.get("jurisdiction_risk") or {}
    jurisdiction_penalty = min(
        20.0,
        max(0.0, float(jurisdiction.get("penalty_points") or 0)),
    )
    reasons = [
        f"Jurisdiktionsabschlag {jurisdiction_penalty:.0f} Punkte "
        f"({jurisdiction.get('level') or 'low'}; begrenzte Heuristik 0–20)."
    ]
    penalty_evidence_ids: dict[str, list[str]] = {
        "jurisdiction_penalty": (
            [f"jurisdiction:{row.get('jurisdiction_code') or 'unclassified'}"]
            if jurisdiction_penalty
            else []
        )
    }

    size_penalty = 0.0
    market_cap = row.get("market_cap_usd")
    if _finite(market_cap) and market_cap > 0:
        if market_cap < 300_000_000:
            size_penalty = 12.0
            reasons.append(
                f"Microcap-/Liquiditätskontext 12 Punkte: verlässliche Marktkapitalisierung ca. {market_cap / 1e6:.0f} Mio. USD."
            )
        elif market_cap < 2_000_000_000:
            size_penalty = 7.0
            reasons.append(
                f"Small-Cap-/Liquiditätskontext 7 Punkte: verlässliche Marktkapitalisierung ca. {market_cap / 1e9:.2f} Mrd. USD."
            )
        elif market_cap < 10_000_000_000:
            size_penalty = 3.0
            reasons.append(
                f"Mid-Cap-Kontext 3 Punkte: verlässliche Marktkapitalisierung ca. {market_cap / 1e9:.2f} Mrd. USD."
            )
        else:
            reasons.append("Größen-/Liquiditätsabschlag 0 Punkte: Marktkapitalisierung mindestens 10 Mrd. USD.")
    else:
        reasons.append("Größen-/Liquiditätsabschlag 0 Punkte: keine verlässlich in USD normalisierte Marktkapitalisierung.")
    penalty_evidence_ids["size_liquidity_penalty"] = (
        ["market_cap_usd:size_bucket"] if size_penalty else []
    )

    sector_text = f"{row.get('sector') or ''} {row.get('industry') or ''}".casefold()
    commodity = any(
        token in sector_text
        for token in (
            "energy",
            "basic materials",
            "oil",
            "gas",
            "mining",
            "metal",
            "steel",
            "commodity",
        )
    )
    earnings_growth_pct = _pct_from_ratio(row.get("earnings_growth"))
    revenue_growth_pct = row.get("revenue_growth_pct")
    cyclical_penalty = 0.0
    cyclical_evidence_ids: list[str] = []
    if (
        commodity
        and _finite(row.get("pe"))
        and row["pe"] < 12
        and _finite(earnings_growth_pct)
        and earnings_growth_pct > 40
    ):
        cyclical_penalty += 3.0
        cyclical_evidence_ids.extend(
            [
                "peak_cycle:commodity_sector",
                "peak_cycle:low_positive_trailing_pe",
                "peak_cycle:earnings_growth_gt_40pct",
            ]
        )
    if (
        commodity
        and _finite(revenue_growth_pct)
        and revenue_growth_pct > 40
        and _finite(earnings_growth_pct)
        and earnings_growth_pct > 40
    ):
        cyclical_penalty += 2.0
        cyclical_evidence_ids.append("peak_cycle:revenue_growth_gt_40pct")
    cyclical_penalty = min(8.0, cyclical_penalty)
    if cyclical_penalty:
        evidence = []
        if (
            _finite(row.get("pe"))
            and row["pe"] < 12
            and _finite(earnings_growth_pct)
            and earnings_growth_pct > 40
        ):
            evidence.append(
                f"KGV {row['pe']:.1f} bei Ergebniswachstum {earnings_growth_pct:+.1f}% (mögliche zyklisch erhöhte Gewinnbasis)"
            )
        if (
            _finite(revenue_growth_pct)
            and revenue_growth_pct > 40
            and _finite(earnings_growth_pct)
            and earnings_growth_pct > 40
        ):
            evidence.append(
                f"gleichzeitig starkes Umsatz-/Ergebniswachstum {revenue_growth_pct:+.1f}%/{earnings_growth_pct:+.1f}%"
            )
        reasons.append(
            f"Zyklik-/Peak-Earnings-Abschlag {cyclical_penalty:.0f} Punkte: "
            + ", ".join(evidence)
            + "."
        )
    else:
        reasons.append(
            "Zyklik-/Peak-Earnings-Abschlag 0 Punkte: kein eigenständiger Peak-Basis-Hinweis (Rohstoffsektor plus niedriges positives KGV und außergewöhnlich hohes positives Wachstum)."
        )
    penalty_evidence_ids["cyclical_peak_penalty"] = list(
        dict.fromkeys(cyclical_evidence_ids)
    )

    shrinking_penalty = 0.0
    shrinking_evidence = []
    shrinking_evidence_ids: list[str] = []
    if _finite(revenue_growth_pct) and revenue_growth_pct < 0:
        shrinking_penalty += 4.0
        shrinking_evidence.append(f"Umsatz {revenue_growth_pct:+.1f}%")
        shrinking_evidence_ids.append("shrink:revenue_growth_negative")
    if _finite(earnings_growth_pct) and earnings_growth_pct < 0:
        shrinking_penalty += 4.0
        shrinking_evidence.append(f"Ergebnis {earnings_growth_pct:+.1f}%")
        shrinking_evidence_ids.append("shrink:earnings_growth_negative")
    if any(
        _finite(value) and value < -25
        for value in (revenue_growth_pct, earnings_growth_pct)
    ):
        shrinking_penalty += 2.0
        shrinking_evidence_ids.append("shrink:growth_below_minus_25pct")
    shrinking_penalty = min(10.0, shrinking_penalty)
    reasons.append(
        f"Schrumpfungsabschlag {shrinking_penalty:.0f} Punkte"
        + (": " + ", ".join(shrinking_evidence) if shrinking_evidence else ": keine negativen Wachstumswerte")
        + "."
    )
    penalty_evidence_ids["shrinking_fundamentals_penalty"] = list(
        dict.fromkeys(shrinking_evidence_ids)
    )

    trend_penalty = 0.0
    trend_evidence = []
    if row.get("falling_knife"):
        trend_penalty += 8.0
        trend_evidence.append("Falling-Knife-Warnung")
    if (row.get("downside_structure") or {}).get("risk") == "hoch":
        trend_penalty += 5.0
        trend_evidence.append("hohe Abwärtsstruktur")
    if _finite(row.get("longterm_score")) and row["longterm_score"] < 40:
        trend_penalty += 4.0
        trend_evidence.append(f"Trend-Score {row['longterm_score']:.0f}/100")
    if _finite(row.get("ret_60d")) and row["ret_60d"] < -15:
        trend_penalty += 3.0
        trend_evidence.append(f"60T {row['ret_60d']:+.1f}%")
    trend_penalty = min(10.0, trend_penalty)
    reasons.append(
        f"Trend-/Downside-Abschlag {trend_penalty:.0f} Punkte"
        + (": " + ", ".join(trend_evidence) if trend_evidence else ": keine der definierten Schwächeschwellen erreicht")
        + "."
    )
    penalty_evidence_ids["weak_trend_downside_penalty"] = [
        f"trend:{item}" for item in trend_evidence
    ]
    overlap = set(penalty_evidence_ids["cyclical_peak_penalty"]) & set(
        penalty_evidence_ids["shrinking_fundamentals_penalty"]
    )
    if overlap:
        raise RuntimeError(f"Penalty evidence overlap is forbidden: {sorted(overlap)}")

    return (
        {
            "jurisdiction_penalty": jurisdiction_penalty,
            "size_liquidity_penalty": size_penalty,
            "cyclical_peak_penalty": cyclical_penalty,
            "shrinking_fundamentals_penalty": shrinking_penalty,
            "weak_trend_downside_penalty": trend_penalty,
        },
        reasons,
        penalty_evidence_ids,
    )


def _valuation_thesis(
    row: dict[str, Any],
    valuation: dict[str, Any],
) -> dict[str, Any]:
    if not valuation.get("ranking_eligible"):
        return {
            **_provenance(
                ["valuation_context", "jurisdiction_risk", "market_cap_usd", "growth and technical fields"],
                ["current complete comparable company fundamentals required"],
            ),
            "available": False,
            "why_it_looks_cheap": [],
            "why_discount_may_be_justified": [],
            "strongest_positive_evidence": [],
            "strongest_counterarguments": [],
            "raw_score": None,
            "risk_penalty": None,
            "risk_adjusted_score": None,
            "value_trap_risk": None,
            "penalty_components": {},
            "penalty_evidence_ids": {},
            "penalty_reasons": [],
            "formula": "not calculated without current complete comparable fundamentals",
        }

    value = valuation["value_score"]
    quality = valuation["quality_score"]
    raw = round(0.55 * value + 0.45 * quality, 2)
    penalties, penalty_reasons, penalty_evidence_ids = _valuation_penalties(row)
    total_penalty = round(min(45.0, sum(penalties.values())), 2)
    adjusted = round(max(0.0, min(100.0, raw - total_penalty)), 2)

    cheap = list(valuation.get("why_undervalued") or [])
    if _finite(row.get("pe")) and row["pe"] > 0:
        cheap.append(f"Trailing-KGV {row['pe']:.2f} (deskriptiver Ist-Wert).")
    if _finite(row.get("forward_pe")) and row["forward_pe"] > 0:
        cheap.append(f"Forward-KGV {row['forward_pe']:.2f} (Provider-Schätzung).")
    if _finite(row.get("pb")) and row["pb"] > 0:
        cheap.append(f"Kurs-Buchwert-Verhältnis {row['pb']:.2f}.")
    cheap.append(f"Raw Value/Quality {raw:.2f}/100 = 55% Value + 45% Quality.")

    positives = [f"Qualitäts-Score {quality:.1f}/100 bei aktuellen vollständigen Daten."]
    if _finite(row.get("roe_pct")):
        positives.append(f"Eigenkapitalrendite {row['roe_pct']:.1f}%.")
    if _finite(row.get("revenue_growth_pct")) and row["revenue_growth_pct"] > 0:
        positives.append(f"Umsatzwachstum {row['revenue_growth_pct']:+.1f}%.")
    earnings_growth_pct = _pct_from_ratio(row.get("earnings_growth"))
    if _finite(earnings_growth_pct) and earnings_growth_pct > 0:
        positives.append(f"Ergebniswachstum {earnings_growth_pct:+.1f}%.")

    jurisdiction_reasons = list((row.get("jurisdiction_risk") or {}).get("reasons") or [])
    justified = jurisdiction_reasons + penalty_reasons
    forward_multiple_counterargument = (
        f"Forward-KGV {row['forward_pe']:.2f} ist wegen erwarteter Verluste kein sinnvoller Bewertungsmultiplikator."
        if _finite(row.get("forward_pe")) and row["forward_pe"] <= 0
        else None
    )
    counterarguments = [
        *jurisdiction_reasons,
        forward_multiple_counterargument,
        *[
            reason
            for reason in penalty_reasons
            if not reason.endswith("0 Punkte: keine der definierten Schwächeschwellen erreicht.")
            and " 0 Punkte:" not in reason
        ],
        "Absolute Bewertungsbänder sind nicht robust sektor- und zeitpunktneutral peer-adjustiert.",
    ]
    risk_level = (row.get("jurisdiction_risk") or {}).get("level")
    trap = "high" if total_penalty >= 20 or risk_level == "high" else "medium" if total_penalty >= 8 else "low"
    return {
        **_provenance(
            [
                "valuation_context.value_score",
                "valuation_context.quality_score",
                "jurisdiction_risk.penalty_points",
                "market_cap_usd",
                "sector",
                "industry",
                "revenue_growth_pct",
                "earnings_growth",
                "longterm_score",
                "ret_60d",
                "downside_structure",
                "falling_knife",
            ]
        ),
        "available": True,
        "why_it_looks_cheap": list(dict.fromkeys(cheap))[:8],
        "why_discount_may_be_justified": list(dict.fromkeys(justified))[:14],
        "strongest_positive_evidence": positives[:6],
        "strongest_counterarguments": list(
            dict.fromkeys(argument for argument in counterarguments if argument)
        )[:12],
        "raw_score": raw,
        "risk_penalty": total_penalty,
        "risk_adjusted_score": adjusted,
        "value_trap_risk": trap,
        "penalty_components": penalties,
        "penalty_evidence_ids": penalty_evidence_ids,
        "penalty_reasons": penalty_reasons,
        "formula": (
            "risk_adjusted_score = clamp(55% value_score + 45% quality_score "
            "- min(45, jurisdiction + size/liquidity + cyclical peak + "
            "shrinking fundamentals + weak trend/downside), 0, 100)"
        ),
    }


def _entry_thesis(
    row: dict[str, Any],
    *,
    timing_score: float | None,
    phase: dict[str, Any] | None,
    downside: dict[str, Any] | None,
    analyst: dict[str, Any],
) -> dict[str, Any]:
    if timing_score is None or not _technical_complete(row):
        return {
            **_provenance(
                ["completed-daily technical fields"],
                ["complete technical feature set required"],
            ),
            "available": False,
            "why_timing_may_be_good": [],
            "what_confirms": [],
            "what_invalidates": [],
            "strongest_supporting_evidence": [],
            "strongest_counterarguments": [],
            "timing_score": None,
            "trend": None,
            "regime": None,
            "support_atr_context": {},
            "falling_knife_bottoming_status": "unavailable",
        }

    price = row.get("price")
    sma50, sma200 = row.get("sma50"), row.get("sma200")
    rsi = row.get("rsi")
    macd_hist, macd_prev = row.get("macd_hist"), row.get("macd_hist_prev")
    ret20, ret60 = row.get("ret_20d"), row.get("ret_60d")
    atr_pct = row.get("atr_pct")
    support = (downside or {}).get("support1")
    support_pct = (downside or {}).get("support1_pct")

    why: list[str] = []
    confirms: list[str] = []
    invalidates: list[str] = []
    support_evidence: list[str] = []
    counter: list[str] = []

    if _finite(rsi):
        if 35 <= rsi <= 60:
            why.append(f"RSI {rsi:.1f}: weder überkauft noch extrem überverkauft.")
        elif rsi >= 70:
            counter.append(f"RSI {rsi:.1f}: technisch überkauft.")
        elif rsi < 30:
            counter.append(f"RSI {rsi:.1f}: starke Schwäche, keine bestätigte Wende.")
    if all(_finite(value) for value in (price, sma50)):
        if price >= sma50:
            why.append(f"Schlusskurs {price:.2f} liegt über SMA50 {sma50:.2f}.")
        else:
            counter.append(f"Schlusskurs {price:.2f} liegt unter SMA50 {sma50:.2f}.")
        confirms.append(f"Nächste abgeschlossene Tagesbars halten/erobern SMA50 {sma50:.2f}.")
    if all(_finite(value) for value in (price, sma200)):
        if price >= sma200:
            support_evidence.append(f"Schlusskurs {price:.2f} liegt über SMA200 {sma200:.2f}.")
        else:
            counter.append(f"Schlusskurs {price:.2f} liegt unter SMA200 {sma200:.2f}.")
        invalidates.append(f"Nachhaltiger Schluss unter SMA200 {sma200:.2f} schwächt das Regime.")
    if all(_finite(value) for value in (macd_hist, macd_prev)):
        if macd_hist > macd_prev:
            why.append(f"MACD-Histogramm steigt von {macd_prev:.3f} auf {macd_hist:.3f}.")
        else:
            counter.append(f"MACD-Histogramm fällt von {macd_prev:.3f} auf {macd_hist:.3f}.")
        confirms.append(f"MACD-Histogramm bleibt über {macd_prev:.3f} bzw. dreht weiter nach oben.")
        invalidates.append(f"MACD-Histogramm fällt unter {macd_prev:.3f} und bestätigt keine Verbesserung.")
    if _finite(ret20) and _finite(ret60):
        if ret20 >= 0:
            why.append(f"20T-Performance {ret20:+.1f}% bei 60T {ret60:+.1f}%.")
        else:
            counter.append(f"20T-Performance {ret20:+.1f}% bei 60T {ret60:+.1f}%.")
        confirms.append("20T-Performance bleibt in abgeschlossenen Tagesdaten nicht negativ.")
        invalidates.append("20T-Performance fällt unter -10% ohne Stabilisierung.")
    if _finite(support) and _finite(support_pct) and _finite(atr_pct):
        distance = abs(support_pct)
        support_text = (
            f"Nächster Support {support:.2f} liegt {distance:.1f}% entfernt; ATR {atr_pct:.1f}%."
        )
        if distance <= max(atr_pct * 2, 3):
            why.append(support_text)
            support_evidence.append(support_text)
        else:
            counter.append(support_text)
        confirms.append(f"Support {support:.2f} hält auf abgeschlossener Tagesbasis.")
        invalidates.append(f"Schluss unter Support {support:.2f} negiert den Support-Kontext.")
    if isinstance(row.get("earnings_in_days"), int) and 0 <= row["earnings_in_days"] <= 7:
        counter.append(f"Ergebnistermin in {row['earnings_in_days']} Tagen: erhöhter Ereigniskontext.")
    if row.get("falling_knife"):
        counter.append(row["falling_knife"].get("warning") or "Falling-Knife-Warnung aktiv.")
    if row.get("bottoming"):
        counter.append("Nur spekulative Bodenbildungsbeobachtung; keine bestätigte Trendwende.")
    if analyst.get("available"):
        support_evidence.append(
            f"Separater Analystenkontext: {analyst['analyst_count']} Stimmen, Zielabstand {analyst['upside_pct']:+.1f}%; nicht Teil des Timing-Scores."
        )

    status = (
        "falling_knife"
        if row.get("falling_knife")
        else "bottoming_watch"
        if row.get("bottoming")
        else "none"
    )
    return {
        **_provenance(
            [
                "entry_timing_score",
                "rsi",
                "price",
                "sma50",
                "sma200",
                "macd_hist",
                "macd_hist_prev",
                "ret_20d",
                "ret_60d",
                "downside_structure.support1",
                "atr_pct",
                "earnings_in_days",
                "analyst_context (separate only)",
            ]
        ),
        "available": True,
        "why_timing_may_be_good": why[:8],
        "what_confirms": list(dict.fromkeys(confirms))[:8],
        "what_invalidates": list(dict.fromkeys(invalidates))[:8],
        "strongest_supporting_evidence": support_evidence[:6],
        "strongest_counterarguments": list(dict.fromkeys(counter))[:8],
        "timing_score": timing_score,
        "trend": row.get("longterm_score"),
        "regime": (phase or {}).get("phase"),
        "support_atr_context": {
            "support": support if _finite(support) else None,
            "support_distance_pct": abs(support_pct) if _finite(support_pct) else None,
            "atr_pct": atr_pct if _finite(atr_pct) else None,
        },
        "falling_knife_bottoming_status": status,
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
    enrich_identity(row)
    for key in (
        "longterm_reasons",
        "daily_signal_reasons",
        "weinstein_label",
        "heuristic_summary",
        "plain_summary",
        "research_summary",
        "research_actions",
        "trade_plan_long",
        "trade_plan_short",
        "urgency",
        "actions",
    ):
        if key in row:
            row[key] = _neutralize_legacy_language(row[key])
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
                "jurisdiction_risk",
            ],
            missing_technical,
        ),
        "warnings": row["risk_warnings"],
        "jurisdiction_level": row["jurisdiction_risk"]["level"],
        "jurisdiction_reasons": row["jurisdiction_risk"]["reasons"],
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
    row["valuation_thesis"] = _valuation_thesis(row, valuation)
    if row["valuation_thesis"]["available"]:
        valuation.update(
            {
                "raw_value_quality_score": row["valuation_thesis"]["raw_score"],
                "risk_penalty": row["valuation_thesis"]["risk_penalty"],
                "risk_adjusted_score": row["valuation_thesis"]["risk_adjusted_score"],
                "value_trap_risk": row["valuation_thesis"]["value_trap_risk"],
            }
        )
    row["entry_thesis"] = _entry_thesis(
        row,
        timing_score=timing_score,
        phase=phase,
        downside=downside,
        analyst=analyst,
    )
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
                "valuation_thesis",
                "entry_thesis",
                "jurisdiction_risk",
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
        "reasons": [reason for reason in reasons if reason][:20],
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
            valuation_thesis = row.get("valuation_thesis") or {}
            value, quality = valuation.get("value_score"), valuation.get("quality_score")
            if (
                valuation.get("ranking_eligible")
                and valuation_thesis.get("available")
                and _finite(value)
                and value >= 65
                and _finite(quality)
                and quality >= 60
            ):
                penalties = valuation_thesis["penalty_components"]
                categories["undervalued_quality"].append(
                    _item(
                        row,
                        valuation_thesis["risk_adjusted_score"],
                        {
                            "value": value,
                            "quality": quality,
                            "raw_value_quality_score": valuation_thesis["raw_score"],
                            **penalties,
                            "total_risk_penalty": valuation_thesis["risk_penalty"],
                            "risk_adjusted_score": valuation_thesis["risk_adjusted_score"],
                            "score_formula": "raw - min(45, sum visible penalties)",
                        },
                        [
                            f"Raw {valuation_thesis['raw_score']:.2f} minus Risikoabschlag {valuation_thesis['risk_penalty']:.2f} = adjustiert {valuation_thesis['risk_adjusted_score']:.2f}.",
                            *valuation_thesis["penalty_reasons"],
                            *valuation_thesis["why_it_looks_cheap"],
                        ],
                        raw_score=valuation_thesis["raw_score"],
                        risk_penalty=valuation_thesis["risk_penalty"],
                        risk_level=(row.get("jurisdiction_risk") or {}).get("level"),
                        value_trap_risk=valuation_thesis["value_trap_risk"],
                        penalty_evidence_ids=valuation_thesis["penalty_evidence_ids"],
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
                entry_thesis = row.get("entry_thesis") or {}
                categories["entry_watchlist"].append(
                    _item(
                        row,
                        score,
                        {"entry_timing": timing, "trend": trend},
                        [
                            *entry_thesis.get("why_timing_may_be_good", []),
                            *entry_thesis.get("what_confirms", []),
                            *entry_thesis.get("strongest_counterarguments", []),
                        ],
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
            jurisdiction = row.get("jurisdiction_risk") or {}
            if (jurisdiction.get("penalty_points") or 0) > 0:
                severity += jurisdiction["penalty_points"]
                risk_components["jurisdiction_penalty"] = jurisdiction["penalty_points"]
            if severity:
                categories["risk_watch"].append(
                    _item(
                        row,
                        severity,
                        risk_components,
                        [
                            *warnings,
                            *jurisdiction.get("reasons", []),
                            downside.get("verdict"),
                        ],
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
            "Tages-Setups",
            "45% trend + 35% entry timing + 20% completed-daily context; "
            "requires no falling knife/high critical structure",
        ),
        "undervalued_quality": (
            "Unterbewertete Qualität",
            "raw = 55% value + 45% quality; adjusted = clamp(raw - min(45, "
            "jurisdiction[0..20] + size/liquidity[0..12] + cyclical peak[0..8] + "
            "shrinking fundamentals[0..10] + weak trend/downside[0..10]), 0, 100); "
            "current complete company fundamentals only; generic banks/insurers/REITs excluded",
        ),
        "analyst_potential": (
            "Analysten-Potenzial",
            "55% normalized analyst target gap + 25% trend + 20% timing, "
            "minus visible overbought/weak-trend penalties; >=5 analysts",
        ),
        "entry_watchlist": (
            "Timing-Beobachtung",
            "65% entry timing + 35% trend; no negative daily context, falling knife "
            "or high downside structure; valuation is not an entry reason",
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


def rehydrate_rankings(
    rankings: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace ranking members with enriched rows without changing rank order."""
    by_symbol = {
        row.get("symbol"): row
        for row in rows
        if isinstance(row, dict) and row.get("symbol")
    }
    for currency, by_asset in rankings.items():
        if not isinstance(by_asset, dict):
            raise ValueError(f"Invalid ranking partition for {currency!r}")
        for asset_type, members in by_asset.items():
            if not isinstance(members, list):
                raise ValueError(
                    f"Invalid ranking members for {currency!r}/{asset_type!r}"
                )
            hydrated = []
            for member in members:
                symbol = member.get("symbol") if isinstance(member, dict) else None
                if symbol not in by_symbol:
                    raise ValueError(
                        f"Ranking references missing enriched symbol {symbol!r}"
                    )
                hydrated.append(copy.deepcopy(by_symbol[symbol]))
            by_asset[asset_type] = hydrated
    return rankings


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
    enriched["rankings_by_currency_asset"] = rehydrate_rankings(
        enriched.get("rankings_by_currency_asset") or {},
        rows,
    )
    by_symbol = {row["symbol"]: row for row in rows if row.get("symbol")}
    for key in ("aschenbrenner_holdings",):
        if isinstance(enriched.get(key), list):
            enriched[key] = [
                copy.deepcopy(by_symbol[member["symbol"]])
                for member in enriched[key]
                if isinstance(member, dict) and member.get("symbol") in by_symbol
            ]
    enriched["insight_rankings"] = rankings
    enriched["insight_metadata"] = {
        "contract_version": INSIGHT_CONTRACT_VERSION,
        "model_status": INSIGHT_STATUS,
        "actionable": False,
        "enriched_at": utc_now(),
        "core_ranking_unchanged": True,
        "core_ranking_symbol_order_unchanged": True,
        "core_ranking_rows_rehydrated": True,
        "scenario_ranges_used_in_core_ranking": False,
        "provenance_catalog": PROVENANCE_CATALOG,
    }
    enriched["schema_version"] = OUTPUT_SCHEMA_VERSION
    enriched["_meta"] = schema_meta(
        "stock-radar-output",
        schema_version=OUTPUT_SCHEMA_VERSION,
        insight_contract=INSIGHT_CONTRACT_VERSION,
    )
    return enriched
