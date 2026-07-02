"""Expert trading/investing frameworks from the literature:
  - Mark Minervini's Trend Template (SEPA) — 8 stage-2 criteria
  - Stan Weinstein's Stage Analysis (1-4) via the 30-week (150d) MA
  - Aggregated Trend / Momentum / Volume-flow sub-scores from the pro indicators
"""


def _has(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def minervini(f, rs_rating=None):
    """Return (score 0-100, list of met criteria, list of failed).
    rs_rating: 0-100 relative-strength percentile (from the universe)."""
    price = f.get("price")
    s50, s150, s200 = f.get("sma50"), f.get("sma150"), f.get("sma200")
    s200_prev = f.get("sma200_1m_ago")
    pfh = f.get("pct_from_high52")          # negative, distance below 52w high
    pal = f.get("pct_above_low52")
    met, failed = [], []

    def check(cond, label):
        (met if cond else failed).append(label)

    if _has(price) and _has(s150) and _has(s200):
        check(price > s150 and price > s200, "Kurs über 150- & 200-Tage-Linie")
    if _has(s150) and _has(s200):
        check(s150 > s200, "150-Tage über 200-Tage-Linie")
    if _has(s200) and _has(s200_prev):
        check(s200 > s200_prev, "200-Tage-Linie steigt")
    if _has(s50) and _has(s150) and _has(s200):
        check(s50 > s150 and s50 > s200, "50-Tage über 150- & 200-Tage")
    if _has(price) and _has(s50):
        check(price > s50, "Kurs über 50-Tage-Linie")
    if _has(pal):
        check(pal >= 30, "≥30 % über 52-Wochen-Tief")
    if _has(pfh):
        check(pfh >= -25, "≤25 % unter 52-Wochen-Hoch")
    if _has(rs_rating):
        check(rs_rating >= 70, f"Relative Stärke stark (RS {rs_rating:.0f})")

    total = len(met) + len(failed)
    score = round(len(met) / total * 100) if total else None
    return score, met, failed


def weinstein_stage(f):
    """Return (stage_number, label) from 30-week (150d) MA slope + price position."""
    price, ma, ma_prev = f.get("price"), f.get("sma150"), f.get("sma150_1m_ago")
    if not (_has(price) and _has(ma)):
        return None, None
    slope = ((ma - ma_prev) / ma) if (_has(ma_prev) and ma) else 0.0
    if price > ma and slope > 0.005:
        return 2, "Phase 2 – Aufwärtstrend (kaufen/halten)"
    if price < ma and slope < -0.005:
        return 4, "Phase 4 – Abwärtstrend (meiden)"
    if price >= ma and slope <= 0.005:
        return 3, "Phase 3 – Topbildung (Gewinne sichern)"
    return 1, "Phase 1 – Bodenbildung (beobachten)"


def _band(x, bands):
    for thr, sc in bands:
        if x <= thr:
            return sc
    return bands[-1][1]


def tech_trend_score(f):
    """0-100: strength & alignment of the trend (ADX/DMI, MAs, Ichimoku, Supertrend, PSAR)."""
    parts = []
    adx, pdi, mdi = f.get("adx"), f.get("plus_di"), f.get("minus_di")
    if _has(adx) and _has(pdi) and _has(mdi):
        strong = _band(adx, [(15, 40), (25, 65), (40, 85), (1e9, 100)])
        parts.append(strong if pdi >= mdi else 100 - strong)
    price, s50, s200 = f.get("price"), f.get("sma50"), f.get("sma200")
    if _has(price) and _has(s50) and _has(s200):
        parts.append(100 if price > s50 > s200 else 70 if price > s200 else 25)
    if f.get("ema9_above_21") is not None:
        parts.append(75 if f["ema9_above_21"] else 25)
    ich = f.get("ichimoku")
    if ich:
        parts.append({"über Wolke": 90, "in Wolke": 50, "unter Wolke": 15}.get(ich, 50))
    if f.get("supertrend_up") is not None:
        parts.append(80 if f["supertrend_up"] else 20)
    if f.get("psar_bull") is not None:
        parts.append(70 if f["psar_bull"] else 30)
    return round(sum(parts) / len(parts)) if parts else None


def tech_momentum_score(f):
    """0-100: momentum oscillators (RSI, Stochastic, Williams %R, CCI, ROC, MACD, Aroon)."""
    parts = []
    rsi = f.get("rsi")
    if _has(rsi):
        parts.append(_band(rsi, [(30, 30), (45, 55), (60, 80), (70, 65), (1e9, 40)]))
    k = f.get("stoch_k")
    if _has(k):
        parts.append(_band(k, [(20, 30), (50, 60), (80, 80), (1e9, 45)]))
    wr = f.get("williams_r")
    if _has(wr):
        parts.append(_band(wr, [(-80, 30), (-50, 60), (-20, 80), (1e9, 45)]))
    cci = f.get("cci")
    if _has(cci):
        parts.append(_band(cci, [(-100, 30), (0, 55), (100, 80), (1e9, 55)]))
    roc = f.get("roc10")
    if _has(roc):
        parts.append(_band(roc, [(-5, 30), (0, 50), (5, 70), (1e9, 85)]))
    mh, mhp = f.get("macd_hist"), f.get("macd_hist_prev")
    if _has(mh) and _has(mhp):
        parts.append(80 if mh > 0 and mh > mhp else 60 if mh > 0 else 30)
    au, ad = f.get("aroon_up"), f.get("aroon_down")
    if _has(au) and _has(ad):
        parts.append(_band(au - ad, [(-50, 25), (0, 50), (50, 75), (1e9, 90)]))
    return round(sum(parts) / len(parts)) if parts else None


def tech_volume_score(f):
    """0-100: volume-flow confirmation (OBV, MFI, CMF, relative volume)."""
    parts = []
    if f.get("obv_rising") is not None:
        parts.append(75 if f["obv_rising"] else 30)
    mfi = f.get("mfi")
    if _has(mfi):
        parts.append(_band(mfi, [(20, 30), (50, 60), (80, 80), (1e9, 45)]))
    cmf = f.get("cmf")
    if _has(cmf):
        parts.append(_band(cmf, [(-0.1, 30), (0, 50), (0.1, 70), (1e9, 85)]))
    rvol = f.get("rvol")
    if _has(rvol):
        parts.append(_band(rvol, [(0.7, 40), (1.2, 60), (2, 80), (1e9, 90)]))
    return round(sum(parts) / len(parts)) if parts else None
