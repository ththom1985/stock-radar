"""Additive, unvalidated expert composites with explicit data coverage."""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone

from .config import DATA
from .persistence import load_json

WEIGHTS_PATH = DATA / "expert_score_weights.json"
EXPERT_MODEL_STATUS = "heuristic_unvalidated"

DEFAULT_WEIGHTS = {
    "long_term": {
        "value": 27.0,
        "quality": 22.0,
        "growth": 18.0,
        "momentum": 12.0,
        "sentiment": 9.0,
        "alternative_data": 12.0,
    },
    "short_term": {
        "technical_momentum": 35.0,
        "catalysts": 20.0,
        "sentiment": 10.0,
        "valuation": 15.0,
        "alternative_data": 20.0,
    },
}

SOURCE_CATALOG = {
    "yfinance_price": {
        "source": "Yahoo Finance via yfinance",
        "cost": "free",
        "expected_delay": "completed daily bar; exchange close plus safety buffer",
        "coverage": "global listed instruments where Yahoo has data",
        "official": False,
    },
    "yfinance_fundamentals": {
        "source": "Yahoo Finance via yfinance",
        "cost": "free",
        "expected_delay": "provider dependent; refreshed weekly",
        "coverage": "global, field coverage varies",
        "official": False,
    },
    "sec_companyfacts": {
        "source": "SEC EDGAR Companyfacts",
        "cost": "free",
        "expected_delay": "filing publication time",
        "coverage": "SEC filers only",
        "official": True,
    },
    "sec_forms_345": {
        "source": "SEC EDGAR Forms 3/4/5",
        "cost": "free",
        "expected_delay": "normally up to 2 business days after transaction",
        "coverage": "SEC filers only",
        "official": True,
    },
    "sec_13f": {
        "source": "SEC EDGAR 13F-HR",
        "cost": "free",
        "expected_delay": "up to 45 days after quarter end",
        "coverage": "reporting institutional managers; US holdings",
        "official": True,
    },
    "finra_short_interest": {
        "source": "FINRA",
        "cost": "free registration",
        "expected_delay": "twice monthly publication",
        "coverage": "US market",
        "official": True,
    },
    "congress_disclosures": {
        "source": "US House and Senate disclosure portals",
        "cost": "free",
        "expected_delay": "statutory disclosure can lag up to 45 days",
        "coverage": "reported US congressional transactions",
        "official": True,
    },
    "fred": {
        "source": "Federal Reserve Bank of St. Louis FRED",
        "cost": "free API key",
        "expected_delay": "series dependent",
        "coverage": "US and selected global macro series",
        "official": True,
    },
    "yfinance_options": {
        "source": "Yahoo Finance options chain via yfinance",
        "cost": "free",
        "expected_delay": "provider dependent; not exchange-certified real time",
        "coverage": "symbols with Yahoo option chains",
        "official": False,
    },
    "wikipedia_pageviews": {
        "source": "Wikimedia Pageviews API",
        "cost": "free",
        "expected_delay": "daily aggregates",
        "coverage": "mapped company articles only",
        "official": True,
    },
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(*values):
    clean = [_number(value) for value in values]
    clean = [value for value in clean if value is not None]
    return sum(clean) / len(clean) if clean else None


def _validate_group(group: dict, expected: set[str]) -> dict[str, float]:
    if not isinstance(group, dict) or set(group) != expected:
        raise ValueError(f"score weights must contain exactly {sorted(expected)}")
    clean = {}
    for key, value in group.items():
        numeric = _number(value)
        if numeric is None or numeric < 0:
            raise ValueError(f"score weight {key} must be a finite non-negative number")
        clean[key] = numeric
    if sum(clean.values()) <= 0:
        raise ValueError("score weights must have a positive total")
    return clean


def load_score_weights(path=WEIGHTS_PATH) -> dict[str, dict[str, float]]:
    payload = load_json(path, expected_type=dict, default={})
    if not payload:
        payload = DEFAULT_WEIGHTS
    return {
        "long_term": _validate_group(
            payload.get("long_term", payload.get("long_term_score", {})),
            set(DEFAULT_WEIGHTS["long_term"]),
        ),
        "short_term": _validate_group(
            payload.get("short_term", payload.get("short_term_score", {})),
            set(DEFAULT_WEIGHTS["short_term"]),
        ),
    }


def _analyst_sentiment(row):
    mean = _number(row.get("analyst_mean"))
    count = _number(row.get("analyst_n"))
    if mean is None or count is None or count < 2:
        return None
    return max(0.0, min(100.0, (5.0 - mean) / 4.0 * 100.0))


def _alternative_score(row):
    signals = row.get("alternative_signals") or {}
    return _number(signals.get("confluence_score"))


def _component_values(row):
    sentiment = _mean(_analyst_sentiment(row), row.get("news_score"))
    technical = _mean(
        row.get("entry_timing_score"),
        row.get("tech_momentum"),
        row.get("tech_trend"),
        row.get("tech_volume"),
    )
    catalysts = _number((row.get("catalyst_context") or {}).get("score"))
    return {
        "long_term": {
            "value": _number(row.get("value_score")),
            "quality": _number(row.get("quality_score")),
            "growth": _number(row.get("growth_score")),
            "momentum": _mean(row.get("longterm_score"), row.get("rs_rating")),
            "sentiment": sentiment,
            "alternative_data": _alternative_score(row),
        },
        "short_term": {
            "technical_momentum": technical,
            "catalysts": catalysts,
            "sentiment": _mean(row.get("news_score"), row.get("hype_score")),
            "valuation": _number(row.get("value_score")),
            "alternative_data": _alternative_score(row),
        },
    }


VALUATION_METRICS = (
    "pe",
    "forward_pe",
    "peg",
    "ev_ebitda",
    "price_to_fcf",
    "price_to_sales",
)


def sector_valuation_medians(rows, minimum_peers=5):
    grouped = {}
    for row in rows:
        sector = row.get("sector")
        if not sector:
            continue
        bucket = grouped.setdefault(
            sector, {metric: [] for metric in VALUATION_METRICS}
        )
        for metric in VALUATION_METRICS:
            value = _number(row.get(metric))
            if value is not None and value > 0:
                bucket[metric].append(value)
    medians = {}
    for sector, metrics in grouped.items():
        medians[sector] = {
            metric: round(statistics.median(values), 4)
            if len(values) >= minimum_peers
            else None
            for metric, values in metrics.items()
        }
        medians[sector]["peer_counts"] = {
            metric: len(values) for metric, values in metrics.items()
        }
    return medians


def build_valuation_assessment(row, sector_medians=None, five_year=None):
    sector_medians = sector_medians or {}
    five_year = five_year or {"metrics": {}, "months_available": 0, "complete": False}
    peers = sector_medians.get(row.get("sector")) or {}
    metrics = {}
    implied_prices = []
    current_price = _number(row.get("price_local"))
    for metric in VALUATION_METRICS:
        current = _number(row.get(metric))
        sector = _number(peers.get(metric))
        own_average = _number((five_year.get("metrics") or {}).get(metric))
        metrics[metric] = {
            "current": round(current, 4) if current is not None else None,
            "own_5y_average": (
                round(own_average, 4) if own_average is not None else None
            ),
            "sector_median": round(sector, 4) if sector is not None else None,
            "sector_peer_count": (peers.get("peer_counts") or {}).get(metric, 0),
        }
        reference_values = [
            value for value in (own_average, sector) if value is not None and value > 0
        ]
        if current_price and current and current > 0 and reference_values:
            implied_prices.extend(
                current_price * reference / current for reference in reference_values
            )
    implied_prices = sorted(
        value for value in implied_prices if math.isfinite(value) and value > 0
    )
    fair_range = None
    verdict = "unavailable"
    if len(implied_prices) >= 2 and current_price:
        low_index = max(0, round((len(implied_prices) - 1) * 0.25))
        high_index = min(
            len(implied_prices) - 1,
            round((len(implied_prices) - 1) * 0.75),
        )
        lower = implied_prices[low_index]
        upper = implied_prices[high_index]
        fair_range = {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "currency": row.get("currency"),
            "method": (
                "interquartile implied-price range from available sector medians "
                "and complete point-in-time five-year averages"
            ),
            "input_count": len(implied_prices),
        }
        if current_price < lower * 0.85:
            verdict = "clearly_undervalued"
        elif current_price <= upper:
            verdict = "fair"
        elif current_price <= upper * 1.15:
            verdict = "expensive"
        else:
            verdict = "overpriced"
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "actionable": False,
        "verdict": verdict,
        "fair_value_range": fair_range,
        "metrics": metrics,
        "own_history_months": five_year.get("months_available", 0),
        "own_5y_complete": bool(five_year.get("complete")),
        "missing_note": (
            None
            if five_year.get("complete")
            else "Eigener 5-Jahres-Schnitt noch nicht verfügbar; Monats-Snapshots werden ab jetzt gesammelt."
        ),
    }


def build_entry_assessment(row):
    sweet = row.get("sweet_spot") or {}
    status = sweet.get("combined_status")
    falling_value = row.get("falling_knife")
    falling = (
        bool(falling_value.get("active"))
        if isinstance(falling_value, dict)
        else bool(falling_value)
    )
    if falling or status in {"safety_blocked", "broken_below"}:
        signal = "risk_too_high"
    elif status == "in_zone_confirmed":
        signal = "positive_setup"
    elif sweet.get("available"):
        signal = "wait_for_pullback"
    else:
        signal = "insufficient_data"
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "actionable": False,
        "signal": signal,
        "zone": {
            "lower": sweet.get("lower"),
            "ideal": sweet.get("ideal"),
            "upper": sweet.get("upper"),
            "currency": "USD",
        } if sweet.get("available") else None,
        "technical_inputs": {
            "rsi": row.get("rsi"),
            "macd": row.get("macd"),
            "macd_signal": row.get("macd_signal"),
            "sma50": row.get("sma50"),
            "sma200": row.get("sma200"),
            "bollinger_lower": row.get("bb_lower"),
            "bollinger_upper": row.get("bb_upper"),
            "relative_volume": row.get("rvol"),
        },
        "reasons": (
            list(sweet.get("why_green_or_not") or [])
            + list(sweet.get("confirmation_needed") or [])
        )[:8],
    }


def build_risk_assessment(row):
    risks = list(row.get("risk_warnings") or [])
    if _number(row.get("debt_to_equity_pct")) is not None:
        risks.append(f"Debt/Equity: {row['debt_to_equity_pct']:.1f}%")
    if _number(row.get("beta")) is not None:
        risks.append(f"Beta: {row['beta']:.2f}")
    if row.get("next_earnings"):
        risks.append(
            f"Nächster Ergebnistermin: {row['next_earnings']} "
            f"({row.get('earnings_in_days')} Tage)"
        )
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "actionable": False,
        "top_risks": risks[:3],
        "debt_to_equity_pct": row.get("debt_to_equity_pct"),
        "beta": row.get("beta"),
        "short_interest": (
            (row.get("alternative_signals") or {}).get("signals") or {}
        ).get("short_interest"),
        "next_earnings": row.get("next_earnings"),
    }


def build_scenario_assessment(row):
    return {
        "model_status": "uncalibrated_scenarios",
        "actionable": False,
        "probabilities_status": "withheld_no_validated_model",
        "probabilities": None,
        "scenarios": row.get("scenario_long") or [],
        "why": (
            list(row.get("macro_notes") or [])
            + list(row.get("longterm_reasons") or [])
            + list(row.get("daily_signal_reasons") or [])
        )[:8],
    }


def _score_group(values, weights):
    total_weight = sum(weights.values())
    available = {
        key: value for key, value in values.items() if _number(value) is not None
    }
    used_weight = sum(weights[key] for key in available)
    score = None
    if used_weight:
        score = sum(available[key] * weights[key] for key in available) / used_weight
        score = round(max(0.0, min(100.0, score)), 1)
    return {
        "score": score,
        "coverage_pct": round(used_weight / total_weight * 100.0, 1),
        "components": {
            key: {
                "value": round(value, 1) if value is not None else None,
                "configured_weight_pct": round(weights[key] / total_weight * 100.0, 2),
                "used": key in available,
            }
            for key, value in values.items()
        },
        "missing_components": [key for key in values if key not in available],
    }


def build_expert_analysis(
    row,
    weights=None,
    *,
    sector_medians=None,
    five_year=None,
):
    weights = weights or load_score_weights()
    values = _component_values(row)
    long_term = _score_group(values["long_term"], weights["long_term"])
    short_term = _score_group(values["short_term"], weights["short_term"])

    sweet_status = (row.get("sweet_spot") or {}).get("combined_status")
    falling_value = row.get("falling_knife")
    falling = (
        bool(falling_value.get("active"))
        if isinstance(falling_value, dict)
        else bool(falling_value)
    )
    short_score = short_term["score"]
    if falling or (short_score is not None and short_score < 35):
        signal = "risk_too_high"
    elif sweet_status == "in_zone_confirmed" and short_score is not None and short_score >= 65:
        signal = "positive_setup"
    elif short_score is not None:
        signal = "wait_for_pullback"
    else:
        signal = "insufficient_data"

    evidence_coverage = min(long_term["coverage_pct"], short_term["coverage_pct"])
    evidence_quality = (
        "high" if evidence_coverage >= 85 else
        "medium" if evidence_coverage >= 65 else
        "low"
    )
    non_us = row.get("listing_country") not in {
        None, "United States", "USA", "US"
    }
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "actionable": False,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "long_term": long_term,
        "short_term": short_term,
        "signal": signal,
        "evidence_quality": evidence_quality,
        "coverage_basis": {
            "status": "narrower_non_us" if non_us else "standard_us",
            "weights_renormalized": True,
            "structurally_unavailable": (
                ["SEC Form 4", "SEC 13F", "FINRA short interest", "U.S. Congress trades"]
                if non_us else []
            ),
            "eu_source_status": (
                "EQS/DGAP and Bundesanzeiger expose no stable documented free API; "
                "missing fields are omitted, never scored neutral."
                if non_us else None
            ),
        },
        "valuation": build_valuation_assessment(
            row, sector_medians=sector_medians, five_year=five_year
        ),
        "entry": build_entry_assessment(row),
        "outlook": build_scenario_assessment(row),
        "risks": build_risk_assessment(row),
        "disclaimer": "Analyse-Unterstützung, keine Anlageberatung.",
    }


def attach_expert_analysis(rows, weights=None, valuation_history=None):
    weights = weights or load_score_weights()
    sector_medians = sector_valuation_medians(rows)
    from .valuation_history import five_year_averages

    for row in rows:
        row["expert_analysis"] = build_expert_analysis(
            row,
            weights,
            sector_medians=sector_medians,
            five_year=five_year_averages(valuation_history, row.get("symbol")),
        )
    return rows


def build_expert_rankings(rows, top_n=10):
    def rank(horizon):
        eligible = []
        for row in rows:
            analysis = row.get("expert_analysis") or {}
            score = ((analysis.get(horizon) or {}).get("score"))
            coverage = ((analysis.get(horizon) or {}).get("coverage_pct") or 0)
            if _number(score) is not None and coverage >= 60:
                eligible.append(
                    {
                        "symbol": row.get("symbol"),
                        "name": row.get("display_name_full") or row.get("name"),
                        "currency": row.get("currency"),
                        "score": score,
                        "coverage_pct": coverage,
                        "signal": analysis.get("signal"),
                        "evidence_quality": analysis.get("evidence_quality"),
                    }
                )
        return sorted(
            eligible,
            key=lambda item: (-item["score"], -item["coverage_pct"], item["symbol"]),
        )[:top_n]

    return {
        "model_status": EXPERT_MODEL_STATUS,
        "actionable": False,
        "weights": load_score_weights(),
        "long_term": rank("long_term"),
        "short_term": rank("short_term"),
    }
