"""Time-horizon projection per stock — honest, model-based, NOT a guarantee.

Two contexts:
  - "short" (Trading): 1 Tag, 1 Woche — direction driven mainly by momentum.
  - "long"  (Investment): 1/3/12/24 Monate — driven mainly by trend + fundamentals.

Two ingredients, kept transparent:
  1. Expected RANGE from actual volatility (random-walk, ±1 sigma ≈ 2 of 3
     cases). A legitimate statement about spread, not a price prediction.
  2. Directional LEAN + confidence from the signals, capped modestly.

Note: ranges use daily volatility scaled by sqrt(trading days). Sub-daily
(hours) horizons are NOT supported — that needs intraday data we don't fetch.
"""
import math

HORIZONS = {
    "short": [("1 Tag", 1), ("1 Woche", 5)],
    "long": [("1 Monat", 21), ("3 Monate", 63), ("12 Monate", 252), ("24 Monate", 504)],
}


def _has(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def _bias(row, mode):
    """-100..+100 directional lean, weighted for the context."""
    inv = row.get("investment_score")
    inv = inv if _has(inv) else 50
    trend = inv - 50
    dt = row.get("daytrade_score") or 0
    ddir = row.get("daytrade_direction")
    dt_signed = dt * (1 if ddir == "LONG" else -1 if ddir == "SHORT" else 0)
    if mode == "short":
        return dt_signed * 0.7 + trend * 0.3
    return trend * 1.0 + dt_signed * 0.25


def project(row, mode="long"):
    """Return list of horizon dicts for the given context, or []."""
    price = row.get("price")
    vd = row.get("vol_daily")
    if not _has(vd) or vd <= 0:
        atrp = row.get("atr_pct")
        vd = (atrp / 100) if _has(atrp) and atrp > 0 else None
    if not _has(price) or price <= 0 or not vd:
        return []

    bias = _bias(row, mode)
    if bias > 12:
        direction = "eher aufwärts"
    elif bias < -12:
        direction = "eher abwärts"
    else:
        direction = "eher seitwärts"
    conf = 50 + min(28, abs(bias) * 0.5)
    conf_label = "hoch" if conf >= 70 else "mittel" if conf >= 60 else "niedrig"

    out = []
    for label, days in HORIZONS[mode]:
        sigma = vd * math.sqrt(days)
        high_pct = sigma * 100
        out.append({
            "label": label,
            "days": days,
            "direction": direction,
            "low_pct": round(-high_pct, 1),
            "high_pct": round(high_pct, 1),
            "low_price": round(price * (1 - sigma), 2),
            "high_price": round(price * (1 + sigma), 2),
            "confidence_pct": int(round(conf)),
            "confidence_label": conf_label,
        })
    return out
