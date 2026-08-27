"""Additive, unvalidated expert composites with explicit data coverage."""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone

from .config import DATA
from .financial_sector_history import (
    financial_history_baseline,
    financial_model_group,
    financial_peer_benchmarks,
)
from .persistence import load_json

WEIGHTS_PATH = DATA / "expert_score_weights.json"
EXPERT_MODEL_STATUS = "heuristic_unvalidated"
MAX_FAIR_VALUE_DEVIATION_PCT = 50.0
MAX_FAIR_VALUE_WIDTH_FACTOR = 1.5
TRADING_DAYS_PER_YEAR = 252

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
    "yfinance_financial_statements": {
        "source": "Yahoo Finance annual financial statements via yfinance",
        "cost": "free",
        "expected_delay": "provider dependent; refreshed monthly",
        "coverage": "four annual Common Equity periods for supported banks and insurers",
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


def _requires_special_sector_model(row):
    return financial_model_group(row) != "standard"


def _sector_model_reason(row):
    group = financial_model_group(row)
    if group == "reit":
        return (
            "Bewertung sektorbedingt nicht möglich: Für REITs fehlen belastbare "
            "mehrjährige FFO-/AFFO- oder direkte NAV-Daten."
        )
    return (
        "Bewertung sektorbedingt nicht möglich: Für dieses Finanzgeschäft ist "
        "noch kein fachlich passendes Bewertungsmodell zugeordnet."
    )


def _apply_minimum_fair_band(row, lower, upper):
    midpoint = (lower + upper) / 2.0
    atr_pct = _number(row.get("atr_pct"))
    annual_volatility_pct = _number(row.get("vol_annual_pct"))
    daily_volatility_pct = (
        annual_volatility_pct / math.sqrt(TRADING_DAYS_PER_YEAR)
        if annual_volatility_pct is not None
        else None
    )
    anchors = [
        value
        for value in (atr_pct, daily_volatility_pct)
        if value is not None and value > 0
    ]
    original_half_width_pct = (
        (upper - lower) / (2.0 * midpoint) * 100.0 if midpoint > 0 else None
    )
    minimum_half_width_pct = max(anchors) if anchors else None
    expanded = (
        midpoint > 0
        and original_half_width_pct is not None
        and minimum_half_width_pct is not None
        and original_half_width_pct < minimum_half_width_pct
    )
    adjusted_lower, adjusted_upper = lower, upper
    if expanded:
        half_width = midpoint * minimum_half_width_pct / 100.0
        adjusted_lower = max(midpoint * 0.01, midpoint - half_width)
        adjusted_upper = midpoint + half_width
    return adjusted_lower, adjusted_upper, {
        "status": "expanded" if expanded else "already_wide_enough",
        "rule": (
            "Mindestens plus/minus eine typische Tagesbewegung um den Mittelpunkt; "
            "Halbbreite = max(ATR%, annualisierte Volatilität/sqrt(252))."
        ),
        "atr_pct": round(atr_pct, 4) if atr_pct is not None else None,
        "daily_volatility_pct": (
            round(daily_volatility_pct, 4)
            if daily_volatility_pct is not None
            else None
        ),
        "minimum_half_width_pct": (
            round(minimum_half_width_pct, 4)
            if minimum_half_width_pct is not None
            else None
        ),
        "original_half_width_pct": (
            round(original_half_width_pct, 4)
            if original_half_width_pct is not None
            else None
        ),
        "original_range": {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
        },
    }


def _financial_valuation_assessment(row, history, peer):
    group = financial_model_group(row)
    current_price = _number(row.get("price_local"))
    book_value_per_share = _number(row.get("bvps"))
    price_to_book = _number(row.get("pb"))
    roe_pct = _number(row.get("roe_pct"))
    own_ratio = _number((history or {}).get("pb_to_roe_median"))
    peer_ratio = _number((peer or {}).get("pb_to_roe_median"))
    peer_count = int((peer or {}).get("peer_count") or 0)
    history_complete = bool((history or {}).get("complete"))
    reference_families = []
    if history_complete and own_ratio is not None:
        reference_families.append(
            {
                "id": "issuer_history",
                "label": "Eigene Vierjahresbewertung",
                "source": (
                    "Yahoo Finance Common Equity, Ordinary Shares und Net Income "
                    "+ historische Tageskurse"
                ),
                "metrics": ["pb_to_roe"],
                "observation_count": (history or {}).get("annual_points_available", 0),
            }
        )
    if peer_ratio is not None and peer_count >= 5:
        reference_families.append(
            {
                "id": "sector_peers",
                "label": (
                    "Aktuelle Bankenvergleichsgruppe"
                    if group == "bank"
                    else "Aktuelle Versicherungsvergleichsgruppe"
                ),
                "source": "Yahoo Finance aktuelles P/B und ROE anderer Unternehmen",
                "metrics": ["pb_to_roe"],
                "observation_count": peer_count,
            }
        )
    family_ids = {family["id"] for family in reference_families}
    basis_quality = {
        "status": (
            "broad"
            if {"issuer_history", "sector_peers"}.issubset(family_ids)
            else "narrow"
        ),
        "independent_family_count": len(reference_families),
        "reference_families": reference_families,
        "definition": (
            "Breit verlangt für Banken und Versicherer vier saubere Jahresperioden "
            "mit Common Equity sowie eine ausreichend besetzte aktuelle P/B-zu-ROE-"
            "Peergroup. Bewertung auf 4-Jahres-Basis, ein Jahr kürzer als bei "
            "Industriewerten."
        ),
    }
    fair_range = None
    raw_fair_range = None
    minimum_band = None
    verdict = "unavailable"
    current_roe = roe_pct / 100.0 if roe_pct is not None else None
    current_pb_to_roe = (
        price_to_book / current_roe
        if price_to_book is not None
        and price_to_book > 0
        and current_roe is not None
        and current_roe > 0
        else None
    )
    expected_pb = {
        "own_4y": current_roe * own_ratio
        if current_roe is not None and own_ratio is not None
        else None,
        "peer": current_roe * peer_ratio
        if current_roe is not None and peer_ratio is not None
        else None,
    }
    implied_prices = sorted(
        book_value_per_share * multiple
        for multiple in expected_pb.values()
        if book_value_per_share is not None
        and multiple is not None
        and multiple > 0
    )
    plausibility_gate = {
        "status": "withheld_narrow_basis",
        "max_deviation_pct": MAX_FAIR_VALUE_DEVIATION_PCT,
        "max_width_factor": MAX_FAIR_VALUE_WIDTH_FACTOR,
        "observed_deviation_pct": None,
        "observed_width_factor": None,
        "checks": {
            "sector_model_supported": True,
            "basis_broad": basis_quality["status"] == "broad",
            "width_within_limit": False,
            "deviation_within_limit": False,
        },
        "reason": (
            "Für das P/B-zu-ROE-Modell fehlen vier vollständige Common-Equity-"
            "Jahresperioden, eine ausreichend große Peergroup oder aktuelle "
            "positive P/B-, ROE- und Buchwertdaten."
        ),
    }
    if (
        len(implied_prices) == 2
        and current_price is not None
        and current_price > 0
        and basis_quality["status"] == "broad"
    ):
        lower, upper = implied_prices
        lower, upper, minimum_band = _apply_minimum_fair_band(
            row,
            lower,
            upper,
        )
        fair_range = {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "currency": row.get("currency"),
            "method": (
                "P/B relativ zum ROE, abgeleitet aus eigener Vierjahreshistorie "
                "und aktueller Sektor-Peergroup"
            ),
            "input_count": len(reference_families),
            "implied_price_count": len(implied_prices),
        }
        raw_fair_range = dict(fair_range)
        if current_price < lower * 0.85:
            verdict = "clearly_undervalued"
        elif current_price <= upper:
            verdict = "fair"
        elif current_price <= upper * 1.15:
            verdict = "expensive"
        else:
            verdict = "overpriced"
        deviation_pct = (
            (lower / current_price - 1.0) * 100.0
            if current_price < lower
            else (current_price / upper - 1.0) * 100.0
            if current_price > upper
            else 0.0
        )
        width_factor = upper / lower
        checks = {
            "sector_model_supported": True,
            "basis_broad": True,
            "width_within_limit": width_factor <= MAX_FAIR_VALUE_WIDTH_FACTOR,
            "deviation_within_limit": deviation_pct <= MAX_FAIR_VALUE_DEVIATION_PCT,
        }
        plausibility_gate.update(
            {
                "status": "pass",
                "observed_deviation_pct": round(deviation_pct, 2),
                "observed_width_factor": round(width_factor, 4),
                "checks": checks,
                "reason": None,
            }
        )
        failures = [key for key, passed in checks.items() if not passed]
        if failures:
            status = (
                "withheld_wide_range"
                if not checks["width_within_limit"]
                else "withheld_extreme_deviation"
            )
            reasons = []
            if not checks["width_within_limit"]:
                reasons.append(
                    f"Der faire Bereich ist breiter als Faktor "
                    f"{MAX_FAIR_VALUE_WIDTH_FACTOR:.1f}."
                )
            if not checks["deviation_within_limit"]:
                reasons.append(
                    "Der faire Bereich liegt mehr als "
                    f"{MAX_FAIR_VALUE_DEVIATION_PCT:.0f}% vom aktuellen Kurs entfernt."
                )
            plausibility_gate.update({"status": status, "reason": " ".join(reasons)})
            fair_range = None
            verdict = "data_review_required"
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "valuation_status": (
            "evidence_qualified_unbacktested"
            if plausibility_gate["status"] == "pass"
            else "withheld"
        ),
        "valuation_status_label": (
            "Bewertung belastbar; Modell noch nicht rückgeprüft"
            if plausibility_gate["status"] == "pass"
            else "Bewertung zurückgehalten"
        ),
        "sector_model": f"{group}_pb_to_roe_4y",
        "verdict": verdict,
        "fair_value_range": fair_range,
        "raw_fair_value_range": (
            raw_fair_range if plausibility_gate["status"] != "pass" else None
        ),
        "plausibility_gate": plausibility_gate,
        "minimum_band": minimum_band,
        "basis_quality": basis_quality,
        "metrics": {
            "book_value_per_share": book_value_per_share,
            "price_to_book": price_to_book,
            "roe_pct": roe_pct,
            "pb_to_roe": {
                "current": round(current_pb_to_roe, 4)
                if current_pb_to_roe is not None
                else None,
                "own_4y_median": own_ratio,
                "peer_median": peer_ratio,
                "peer_count": peer_count,
            },
            "expected_price_to_book": {
                key: round(value, 4) if value is not None else None
                for key, value in expected_pb.items()
            },
            "annual_calculation": (history or {}).get("annual_points") or [],
        },
        "own_history_months": 0,
        "own_history_annual_points": (history or {}).get(
            "annual_points_available", 0
        ),
        "own_history_type": "yfinance_common_equity_4y",
        "own_5y_complete": False,
        "history_note": (
            "Bewertung auf 4-Jahres-Basis, ein Jahr kürzer als bei Industriewerten."
        ),
        "missing_note": (
            plausibility_gate.get("reason")
            if plausibility_gate["status"] != "pass"
            else None
        ),
    }


def build_valuation_assessment(
    row,
    sector_medians=None,
    five_year=None,
    *,
    financial_history=None,
    financial_peer=None,
):
    if financial_model_group(row) in {"bank", "insurance"}:
        return _financial_valuation_assessment(
            row,
            financial_history or {},
            financial_peer or {},
        )
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
    raw_fair_range = None
    minimum_band = None
    verdict = "unavailable"
    own_metrics = [
        metric
        for metric, values in metrics.items()
        if values.get("own_5y_average") is not None
    ]
    peer_metrics = [
        metric
        for metric, values in metrics.items()
        if values.get("sector_median") is not None
        and (values.get("sector_peer_count") or 0) >= 5
    ]
    reference_families = []
    if five_year.get("complete") and len(own_metrics) >= 2:
        reference_families.append(
            {
                "id": "issuer_history",
                "label": "Eigene Mehrjahresbewertung",
                "source": "SEC EDGAR Companyfacts + historische Tageskurse",
                "metrics": own_metrics,
                "observation_count": five_year.get("annual_points_available")
                or five_year.get("months_available"),
            }
        )
    if len(peer_metrics) >= 2:
        reference_families.append(
            {
                "id": "sector_peers",
                "label": "Aktuelle Sektorvergleichsgruppe",
                "source": "Yahoo Finance Fundamentaldaten anderer Unternehmen",
                "metrics": peer_metrics,
                "observation_count": min(
                    (metrics[metric].get("sector_peer_count") or 0)
                    for metric in peer_metrics
                ),
            }
        )
    family_ids = {family["id"] for family in reference_families}
    basis_quality = {
        "status": (
            "broad"
            if {"issuer_history", "sector_peers"}.issubset(family_ids)
            else "narrow"
        ),
        "independent_family_count": len(reference_families),
        "reference_families": reference_families,
        "definition": (
            "Breit verlangt zwei unabhängige Referenzfamilien: eine eigene "
            "Mehrjahresbewertung aus offiziellen Filings plus historischen Kursen "
            "und eine ausreichend besetzte aktuelle Peer-Verteilung. Mehrere "
            "Kennzahlen derselben Familie zählen nicht mehrfach."
        ),
    }
    plausibility_gate = {
        "status": "pass",
        "max_deviation_pct": MAX_FAIR_VALUE_DEVIATION_PCT,
        "max_width_factor": MAX_FAIR_VALUE_WIDTH_FACTOR,
        "observed_deviation_pct": None,
        "observed_width_factor": None,
        "checks": {},
        "reason": None,
    }
    if len(implied_prices) >= 2 and current_price:
        low_index = max(0, round((len(implied_prices) - 1) * 0.25))
        high_index = min(
            len(implied_prices) - 1,
            round((len(implied_prices) - 1) * 0.75),
        )
        lower = implied_prices[low_index]
        upper = implied_prices[high_index]
        lower, upper, minimum_band = _apply_minimum_fair_band(
            row,
            lower,
            upper,
        )
        fair_range = {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "currency": row.get("currency"),
            "method": (
                "interquartile implied-price range from available sector medians "
                "and multi-year issuer valuation history"
            ),
            "input_count": len(reference_families),
            "implied_price_count": len(implied_prices),
        }
        raw_fair_range = dict(fair_range)
        if current_price < lower * 0.85:
            verdict = "clearly_undervalued"
        elif current_price <= upper:
            verdict = "fair"
        elif current_price <= upper * 1.15:
            verdict = "expensive"
        else:
            verdict = "overpriced"
        deviation_pct = (
            (lower / current_price - 1.0) * 100.0
            if current_price < lower
            else (current_price / upper - 1.0) * 100.0
            if current_price > upper
            else 0.0
        )
        width_factor = upper / lower
        plausibility_gate["observed_deviation_pct"] = round(deviation_pct, 2)
        plausibility_gate["observed_width_factor"] = round(width_factor, 4)
        checks = {
            "sector_model_supported": not _requires_special_sector_model(row),
            "basis_broad": basis_quality["status"] == "broad",
            "width_within_limit": width_factor <= MAX_FAIR_VALUE_WIDTH_FACTOR,
            "deviation_within_limit": deviation_pct <= MAX_FAIR_VALUE_DEVIATION_PCT,
        }
        plausibility_gate["checks"] = checks
        failures = [key for key, passed in checks.items() if not passed]
        if failures:
            status = (
                "withheld_sector_model"
                if not checks["sector_model_supported"]
                else "withheld_wide_range"
                if not checks["width_within_limit"]
                else "withheld_extreme_deviation"
                if not checks["deviation_within_limit"]
                else "withheld_narrow_basis"
            )
            reasons = {
                "sector_model_supported": (
                    _sector_model_reason(row)
                ),
                "basis_broad": (
                    "Die Bewertung besitzt noch nicht beide unabhängigen "
                    "Referenzfamilien."
                ),
                "width_within_limit": (
                    f"Der faire Bereich ist breiter als Faktor "
                    f"{MAX_FAIR_VALUE_WIDTH_FACTOR:.1f}."
                ),
                "deviation_within_limit": (
                    "Der faire Bereich liegt mehr als "
                    f"{MAX_FAIR_VALUE_DEVIATION_PCT:.0f}% vom aktuellen Kurs entfernt."
                ),
            }
            plausibility_gate.update(
                {
                    "status": status,
                    "reason": " ".join(reasons[key] for key in failures),
                }
            )
            fair_range = None
            verdict = "data_review_required"
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "valuation_status": (
            "evidence_qualified_unbacktested"
            if plausibility_gate["status"] == "pass"
            else "withheld"
        ),
        "valuation_status_label": (
            "Bewertung belastbar; Modell noch nicht rückgeprüft"
            if plausibility_gate["status"] == "pass"
            else "Bewertung zurückgehalten"
        ),
        "sector_model": (
            "standard_multiples"
            if financial_model_group(row) == "standard"
            else f"{financial_model_group(row)}_unavailable"
        ),
        "verdict": verdict,
        "fair_value_range": fair_range,
        "raw_fair_value_range": (
            raw_fair_range
            if plausibility_gate["status"] != "pass"
            else None
        ),
        "plausibility_gate": plausibility_gate,
        "minimum_band": minimum_band,
        "basis_quality": basis_quality,
        "metrics": metrics,
        "own_history_months": five_year.get("months_available", 0),
        "own_history_annual_points": five_year.get("annual_points_available", 0),
        "own_history_type": five_year.get("history_type", "insufficient"),
        "own_5y_complete": bool(five_year.get("complete")),
        "missing_note": (
            None
            if five_year.get("complete")
            else (
                "Keine belastbare eigene Mehrjahresbewertung aus mindestens vier "
                "Jahresperioden und zwei Kennzahlen."
            )
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
    debt = _number(row.get("debt_to_equity_pct"))
    beta = _number(row.get("beta"))
    if debt is not None and debt >= 150:
        risks.append(
            f"Die Verschuldung ist mit {debt:.0f}% des Eigenkapitals sehr hoch."
        )
    elif debt is not None and debt >= 100:
        risks.append(
            f"Die Verschuldung ist mit {debt:.0f}% des Eigenkapitals erhöht."
        )
    if beta is not None and beta >= 1.5:
        risks.append(
            f"Der Kurs schwankt mit einem Beta von {beta:.2f} deutlich stärker als der Markt."
        )
    short_interest = (
        (row.get("alternative_signals") or {}).get("signals") or {}
    ).get("short_interest") or {}
    days_to_cover = _number((short_interest.get("evidence") or {}).get("days_to_cover"))
    if days_to_cover is not None and days_to_cover >= 7:
        risks.append(
            f"Die gemeldeten Leerverkäufe entsprechen etwa {days_to_cover:.1f} Handelstagen."
        )
    earnings_days = row.get("earnings_in_days")
    if (
        row.get("next_earnings")
        and isinstance(earnings_days, int)
        and 0 <= earnings_days <= 14
    ):
        risks.append(
            f"Der nächste Ergebnistermin ist bereits in {earnings_days} Tagen."
        )
    if not risks:
        risks.append(
            "Das Modell erkennt aktuell kein einzelnes dominantes Risiko; "
            "Verschuldung und Kursschwankung sind unauffällig."
        )
    return {
        "model_status": EXPERT_MODEL_STATUS,
        "actionable": False,
        "top_risks": risks[:3],
        "debt_to_equity_pct": row.get("debt_to_equity_pct"),
        "beta": row.get("beta"),
        "short_interest": short_interest or None,
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
    financial_history=None,
    financial_peer=None,
):
    weights = weights or load_score_weights()
    valuation_assessment = build_valuation_assessment(
        row,
        sector_medians=sector_medians,
        five_year=five_year,
        financial_history=financial_history,
        financial_peer=financial_peer,
    )
    values = _component_values(row)
    if (
        valuation_assessment.get("plausibility_gate") or {}
    ).get("status") != "pass":
        values["long_term"]["value"] = None
        values["short_term"]["valuation"] = None
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
        "valuation": valuation_assessment,
        "entry": build_entry_assessment(row),
        "outlook": build_scenario_assessment(row),
        "risks": build_risk_assessment(row),
        "disclaimer": "Analyse-Unterstützung, keine Anlageberatung.",
    }


def attach_expert_analysis(
    rows,
    weights=None,
    valuation_history=None,
    *,
    financial_history_records=None,
    price_histories=None,
):
    weights = weights or load_score_weights()
    sector_medians = sector_valuation_medians(rows)
    financial_peers = financial_peer_benchmarks(rows)
    from .valuation_history import five_year_averages

    for row in rows:
        group = financial_model_group(row)
        financial_history = (
            financial_history_baseline(
                (financial_history_records or {}).get(row.get("symbol")),
                (price_histories or {}).get(row.get("symbol")),
            )
            if group in {"bank", "insurance"}
            else None
        )
        row["expert_analysis"] = build_expert_analysis(
            row,
            weights,
            sector_medians=sector_medians,
            five_year=five_year_averages(valuation_history, row.get("symbol")),
            financial_history=financial_history,
            financial_peer=financial_peers.get(group),
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
