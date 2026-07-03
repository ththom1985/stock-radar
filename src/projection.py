"""Time-horizon projection per stock — honest, model-based, NOT a guarantee.

Each horizon genuinely differs:
  - Expected value (center) drifts with time, driven by signal bias + the
    analyst consensus price target (an ~12-month anchor).
  - The uncertainty band widens with sqrt(time) from real volatility (±1σ ≈
    2 of 3 cases).
  - Confidence in the direction decays with the horizon (further out = less sure).

Contexts:
  - "short" (Trading): 1 Tag, 1 Woche — direction driven mainly by momentum.
  - "long"  (Investment): 1/6/12/24 Monate — driven by trend, fundamentals, analysts.

Sub-daily (hours) is NOT supported — needs intraday data we don't fetch.
"""
import math

HORIZONS = {
    "short": [("1 Tag", 1), ("1 Woche", 5)],
    "long": [("1 Monat", 21), ("6 Monate", 126), ("12 Monate", 252), ("24 Monate", 504)],
}


def _has(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _bias(row, mode):
    """-100..+100 directional lean, weighted for the context."""
    inv = row.get("investment_score")
    trend = (inv - 50) if _has(inv) else 0
    dt = row.get("daytrade_score") or 0
    ddir = row.get("daytrade_direction")
    dt_signed = dt * (1 if ddir == "LONG" else -1 if ddir == "SHORT" else 0)
    if mode == "short":
        return dt_signed * 0.7 + trend * 0.3
    return trend * 1.0 + dt_signed * 0.25


def _annual_drift(row, bias, mode):
    """Expected annualised % drift (capped). Blends signal bias with the
    analyst price target for the long view."""
    signal_annual = _clamp(bias * 0.25, -25, 25)          # -25%..+25% p.a. from signals
    upside = row.get("analyst_upside_pct")
    if mode == "long" and _has(upside):
        analyst_annual = _clamp(upside, -35, 50)           # analyst target ≈ 12M view
        return _clamp(0.4 * analyst_annual + 0.6 * signal_annual, -35, 55)   # analysts down-weighted
    return signal_annual


def project(row, mode="long"):
    price = row.get("price")
    vd = row.get("vol_daily")
    if not _has(vd) or vd <= 0:
        atrp = row.get("atr_pct")
        vd = (atrp / 100) if _has(atrp) and atrp > 0 else None
    if not _has(price) or price <= 0 or not vd:
        return []

    bias = _bias(row, mode)
    annual_drift = _annual_drift(row, bias, mode)
    if bias > 12:
        direction = "eher aufwärts"
    elif bias < -12:
        direction = "eher abwärts"
    else:
        direction = "eher seitwärts"
    base_conf = 50 + min(28, abs(bias) * 0.5)

    out = []
    for label, days in HORIZONS[mode]:
        t_years = days / 252
        center = _clamp(annual_drift * t_years, -60, 90)     # expected % move (grows with time)
        sigma = vd * math.sqrt(days) * 100                   # ±1σ band (widens with time)
        low_pct = center - sigma
        high_pct = center + sigma
        conf = 50 + (base_conf - 50) * (1 / (1 + days / 45))  # decays with horizon
        conf = int(round(conf))
        out.append({
            "label": label,
            "days": days,
            "direction": direction,
            "center_pct": round(center, 1),
            "expected_price": round(price * (1 + center / 100), 2),
            "low_pct": round(low_pct, 1),
            "high_pct": round(high_pct, 1),
            "low_price": round(price * (1 + low_pct / 100), 2),
            "high_price": round(price * (1 + high_pct / 100), 2),
            "confidence_pct": conf,
            "confidence_label": "hoch" if conf >= 68 else "mittel" if conf >= 58 else "niedrig",
        })
    return out
