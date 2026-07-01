"""Opportunity scoring. Each factor contributes points AND a plain-German reason,
so every ranking is fully explainable ("Begründung")."""
import numpy as np


def _clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def _ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def score_daytrade(f):
    """Short-term trading opportunity.

    Returns (score 0-100, direction LONG/SHORT/NEUTRAL, list[str] reasons).
    Driven by momentum, breakouts, volume spikes and volatility.
    """
    score = 0.0
    reasons = []
    long_sig = 0
    short_sig = 0
    price = f["price"]

    rvol = f.get("rvol")
    if _ok(rvol) and rvol >= 1.5:
        score += min(25, (rvol - 1) * 20)
        reasons.append(f"Hohes relatives Volumen (RVOL {rvol:.1f}×) – ungewöhnliches Marktinteresse")

    if _ok(f.get("high20")) and price > f["high20"]:
        score += 20
        long_sig += 1
        reasons.append(f"Ausbruch über 20-Tage-Hoch ({f['high20']:.2f}) – Long-Momentum")
    if _ok(f.get("low20")) and price < f["low20"]:
        score += 20
        short_sig += 1
        reasons.append(f"Bruch unter 20-Tage-Tief ({f['low20']:.2f}) – Short-Momentum")

    mh, mhp = f.get("macd_hist"), f.get("macd_hist_prev")
    if _ok(mh) and _ok(mhp):
        if mh > 0 and mhp <= 0:
            score += 15
            long_sig += 1
            reasons.append("MACD dreht bullisch (Histogramm kreuzt über Null)")
        elif mh < 0 and mhp >= 0:
            score += 15
            short_sig += 1
            reasons.append("MACD dreht bärisch (Histogramm kreuzt unter Null)")

    rsi = f.get("rsi")
    if _ok(rsi):
        if rsi <= 30:
            score += 12
            long_sig += 1
            reasons.append(f"RSI überverkauft ({rsi:.0f}) – Erholungschance")
        elif rsi >= 70:
            score += 12
            short_sig += 1
            reasons.append(f"RSI überkauft ({rsi:.0f}) – Rücksetzer-Chance (Short)")

    atrp = f.get("atr_pct")
    if _ok(atrp):
        if atrp >= 2:
            score += 8
            reasons.append(f"Genug Tagesvolatilität (ATR {atrp:.1f}%) für eine Trading-Range")
        elif atrp < 0.8:
            score -= 5
            reasons.append(f"Sehr niedrige Volatilität (ATR {atrp:.1f}%) – wenig Bewegung")

    dr = f.get("daily_return")
    if _ok(dr) and abs(dr) >= 0.03:
        score += 10
        reasons.append(f"Kräftige Tagesbewegung ({dr * 100:+.1f}%)")
        if dr > 0:
            long_sig += 1
        else:
            short_sig += 1

    if long_sig == 0 and short_sig == 0:
        direction = "NEUTRAL"
    elif long_sig >= short_sig:
        direction = "LONG"
    else:
        direction = "SHORT"
    return round(_clamp(score), 1), direction, reasons


def score_longterm(f):
    """Long-term investment / good-entry opportunity.

    Returns (score 0-100, list[str] reasons).
    Rewards intact uptrends bought on healthy pullbacks, not overbought chases.
    """
    score = 0.0
    reasons = []
    price = f["price"]
    sma50 = f.get("sma50")
    sma200 = f.get("sma200")

    if _ok(sma200):
        if price > sma200:
            score += 25
            reasons.append("Kurs über 200-Tage-Linie – langfristiger Aufwärtstrend intakt")
        else:
            score -= 10
            reasons.append("Kurs unter 200-Tage-Linie – Trend angeschlagen")

    if _ok(sma50) and _ok(sma200) and sma50 > sma200:
        score += 15
        reasons.append("Golden-Cross-Struktur (SMA50 über SMA200)")

    rsi = f.get("rsi")
    if _ok(rsi):
        if 40 <= rsi <= 55:
            score += 20
            reasons.append(f"Gesunder Rücksetzer (RSI {rsi:.0f}) – attraktiver Einstieg statt überkauft")
        elif rsi < 40:
            score += 10
            reasons.append(f"RSI niedrig ({rsi:.0f}) – möglicher Boden, aber Trend prüfen")
        elif rsi > 70:
            score -= 10
            reasons.append(f"RSI überkauft ({rsi:.0f}) – schlechter Langfrist-Einstieg")

    if _ok(sma50) and _ok(sma200) and price > sma200:
        dist = (price / sma50 - 1) * 100
        if -3 <= dist <= 3:
            score += 15
            reasons.append("Kurs testet die 50-Tage-Linie im Aufwärtstrend – klassischer Einstiegspunkt")

    pfh = f.get("pct_from_high52")
    if _ok(pfh):
        if -15 <= pfh <= -3:
            score += 10
            reasons.append(f"{abs(pfh):.0f}% unter 52-Wochen-Hoch – Luft nach oben, aber nahe der Stärke")
        elif pfh <= -40:
            score += 5
            reasons.append(f"{abs(pfh):.0f}% unter 52-Wochen-Hoch – tief, spekulativer Turnaround")

    ret60 = f.get("ret_60d")
    if _ok(ret60) and ret60 > 0:
        score += min(10, ret60 / 3)
        reasons.append(f"Positives 3-Monats-Momentum ({ret60:+.0f}%)")

    macd, macd_signal = f.get("macd"), f.get("macd_signal")
    if _ok(macd) and _ok(macd_signal) and macd > macd_signal:
        score += 5
        reasons.append("MACD über Signallinie – Momentum stützt")

    return round(_clamp(score), 1), reasons
