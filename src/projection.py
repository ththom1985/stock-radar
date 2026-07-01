"""Time-horizon projection per stock — honest, model-based, NOT a guarantee.

Two ingredients, kept transparent:
  1. Expected RANGE from actual volatility (random-walk, ±1 sigma ≈ 2 out of 3
     cases). This is a legitimate statistical statement about spread, not a
     price prediction.
  2. Directional LEAN + confidence derived from the signals (trend, momentum,
     fundamentals). Clearly a model estimate, capped modestly so it never
     pretends to be a certainty.
"""
import math

HORIZONS = [("1 Monat", 21), ("3 Monate", 63)]


def _has(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def _direction_bias(row):
    """-100..+100: how bullish/bearish the combined signals lean."""
    inv = row.get("investment_score")
    base = (inv - 50) if _has(inv) else 0            # long-term attractiveness
    dt = row.get("daytrade_score") or 0
    ddir = row.get("daytrade_direction")
    mom = dt * 0.35 * (1 if ddir == "LONG" else -1 if ddir == "SHORT" else 0)
    return base + mom


def project(row):
    """Return list of horizon dicts, or [] if volatility unknown."""
    price = row.get("price")
    vd = row.get("vol_daily")
    if not _has(vd) or vd <= 0:
        atrp = row.get("atr_pct")
        vd = (atrp / 100) if _has(atrp) and atrp > 0 else None
    if not _has(price) or price <= 0 or not vd:
        return []

    bias = _direction_bias(row)
    if bias > 12:
        direction = "eher aufwärts"
    elif bias < -12:
        direction = "eher abwärts"
    else:
        direction = "eher seitwärts"

    # confidence in the DIRECTION: 50% (coin flip) up to ~78%, from signal strength
    conf = 50 + min(28, abs(bias) * 0.5)
    conf_label = "hoch" if conf >= 70 else "mittel" if conf >= 60 else "niedrig"

    out = []
    for label, days in HORIZONS:
        sigma = vd * math.sqrt(days)           # horizon volatility (fraction)
        low_pct = -sigma * 100
        high_pct = sigma * 100
        out.append({
            "label": label,
            "days": days,
            "direction": direction,
            "low_pct": round(low_pct, 1),
            "high_pct": round(high_pct, 1),
            "low_price": round(price * (1 + low_pct / 100), 2),
            "high_price": round(price * (1 + high_pct / 100), 2),
            "confidence_pct": int(round(conf)),
            "confidence_label": conf_label,
        })
    return out
