"""Unvalidated heuristic scenario ranges with strictly positive prices.

These are volatility-scaled research scenarios, not probabilities, forecasts,
expected returns, confidence intervals or ranking inputs.
"""
from __future__ import annotations

import math


HORIZONS = {
    "short": [("1 Woche", 5)],
    "long": [("1 Monat", 21), ("6 Monate", 126), ("12 Monate", 252), ("24 Monate", 504)],
}


def _has(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _reference_annual_change(row, mode):
    """Small bounded heuristic tilt; deliberately excluded from all rankings."""
    core = row.get("investment_score")
    if not _has(core):
        core = row.get("longterm_score")
    tilt = ((core - 50) * 0.30) if _has(core) else 0.0
    daily_direction = row.get("daily_signal_direction")
    daily_score = row.get("daily_signal_score")
    if _has(daily_score):
        sign = 1 if daily_direction == "POSITIVE" else -1 if daily_direction == "NEGATIVE" else 0
        tilt += sign * daily_score * (0.04 if mode == "long" else 0.10)
    return _clamp(tilt, -25.0, 25.0)


def project(row, mode="long"):
    """Return positive-price scenario ranges labelled as unvalidated heuristics."""
    if mode not in HORIZONS:
        raise ValueError(f"Unsupported scenario mode: {mode}")
    price = row.get("price")
    daily_vol = row.get("vol_daily")
    if not _has(daily_vol) or daily_vol <= 0:
        atr_pct = row.get("atr_pct")
        daily_vol = atr_pct / 100 if _has(atr_pct) and atr_pct > 0 else None
    if not _has(price) or price <= 0 or not _has(daily_vol) or daily_vol <= 0:
        return []

    annual_change = _reference_annual_change(row, mode)
    annual_log_tilt = math.log1p(annual_change / 100)
    scenarios = []
    for label, days in HORIZONS[mode]:
        years = days / 252
        sigma = daily_vol * math.sqrt(days)
        reference_multiple = math.exp(annual_log_tilt * years)
        low_multiple = math.exp(annual_log_tilt * years - sigma)
        high_multiple = math.exp(annual_log_tilt * years + sigma)
        reference_price = price * reference_multiple
        low_price = price * low_multiple
        high_price = price * high_multiple
        reference_pct = (reference_multiple - 1) * 100
        scenarios.append(
            {
                "label": label,
                "days": days,
                "model_status": "unvalidated",
                "interpretation": "heuristic scenario range; not statistically calibrated",
                "reference_change_pct": reference_pct,
                "reference_price": reference_price,
                "range_low_pct": (low_multiple - 1) * 100,
                "range_high_pct": (high_multiple - 1) * 100,
                "range_low_price": low_price,
                "range_high_price": high_price,
                # Compatibility for older consumers. These keys are deprecated
                # and are never used for ranking or probability language.
                "center_pct": reference_pct,
                "low_pct": (low_multiple - 1) * 100,
                "high_pct": (high_multiple - 1) * 100,
                "low_price": low_price,
                "high_price": high_price,
            }
        )
    return scenarios
