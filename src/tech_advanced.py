"""Advanced technical indicators from the standard literature (Wilder, Lane,
Appel, Donchian, Ichimoku, Chaikin, etc.) computed on daily OHLCV.

All functions return the LATEST value(s) as plain floats so they can be merged
into the per-ticker feature dict. Everything is vectorised with pandas.
"""
import numpy as np
import pandas as pd


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _wilder(series, period):
    """Wilder's smoothing (RMA)."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def advanced_indicators(df):
    """Return dict of latest advanced-indicator values for one ticker."""
    if df is None or len(df) < 40:
        return {}
    o, h, l, c, v = (df["Open"].astype(float), df["High"].astype(float),
                     df["Low"].astype(float), df["Close"].astype(float),
                     df["Volume"].astype(float))
    out = {}
    price = _f(c.iloc[-1])

    # --- EMAs (short-term trend) ---
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    out["ema9"] = _f(ema9.iloc[-1])
    out["ema21"] = _f(ema21.iloc[-1])
    out["ema9_above_21"] = bool(ema9.iloc[-1] > ema21.iloc[-1])

    # --- True Range / ATR (Wilder) ---
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = _wilder(tr, 14)

    # --- ADX / DMI (Wilder): trend strength + direction ---
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_s = _wilder(tr, 14)
    plus_di = 100 * _wilder(pd.Series(plus_dm, index=c.index), 14) / atr_s
    minus_di = 100 * _wilder(pd.Series(minus_dm, index=c.index), 14) / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder(dx, 14)
    out["adx"] = _f(adx.iloc[-1])
    out["plus_di"] = _f(plus_di.iloc[-1])
    out["minus_di"] = _f(minus_di.iloc[-1])

    # --- Stochastic Oscillator (Lane) %K/%D ---
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_k"] = _f(k.iloc[-1])
    out["stoch_d"] = _f(k.rolling(3).mean().iloc[-1])

    # --- Williams %R ---
    out["williams_r"] = _f(-100 * (hh.iloc[-1] - c.iloc[-1]) / (hh.iloc[-1] - ll.iloc[-1])
                           if (hh.iloc[-1] - ll.iloc[-1]) else np.nan)

    # --- CCI (Commodity Channel Index) ---
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = (tp - sma_tp).abs().rolling(20).mean()
    out["cci"] = _f(((tp - sma_tp) / (0.015 * mad)).iloc[-1])

    # --- ROC (Rate of Change, 10) ---
    out["roc10"] = _f((c.iloc[-1] / c.iloc[-11] - 1) * 100) if len(c) > 11 else None

    # --- OBV (On-Balance Volume) + slope ---
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    out["obv_rising"] = bool(obv.iloc[-1] > obv.iloc[-21]) if len(obv) > 21 else None

    # --- MFI (Money Flow Index, 14) ---
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
    out["mfi"] = _f(mfi.iloc[-1])

    # --- Chaikin Money Flow (20) ---
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    cmf = (mfm * v).rolling(20).sum() / v.rolling(20).sum()
    out["cmf"] = _f(cmf.iloc[-1])

    # --- Aroon (25) ---
    n = 25
    if len(c) > n:
        aroon_up = h.rolling(n + 1).apply(lambda x: (n - x.argmax()) / n * 100, raw=True)
        aroon_dn = l.rolling(n + 1).apply(lambda x: (n - x.argmin()) / n * 100, raw=True)
        out["aroon_up"] = _f(aroon_up.iloc[-1])
        out["aroon_down"] = _f(aroon_dn.iloc[-1])

    # --- Supertrend (10, 3) direction ---
    try:
        mult, per = 3.0, 10
        atr_st = _wilder(tr, per)
        hl2 = (h + l) / 2
        upper = hl2 + mult * atr_st
        lower = hl2 - mult * atr_st
        st_dir = pd.Series(index=c.index, dtype=float)
        fu, fl = upper.copy(), lower.copy()
        dir_up = True
        for i in range(1, len(c)):
            fu.iloc[i] = min(upper.iloc[i], fu.iloc[i - 1]) if c.iloc[i - 1] <= fu.iloc[i - 1] else upper.iloc[i]
            fl.iloc[i] = max(lower.iloc[i], fl.iloc[i - 1]) if c.iloc[i - 1] >= fl.iloc[i - 1] else lower.iloc[i]
            if c.iloc[i] > fu.iloc[i - 1]:
                dir_up = True
            elif c.iloc[i] < fl.iloc[i - 1]:
                dir_up = False
            st_dir.iloc[i] = 1 if dir_up else -1
        out["supertrend_up"] = bool(st_dir.iloc[-1] == 1)
    except Exception:  # noqa: BLE001
        pass

    # --- Parabolic SAR direction (Wilder) ---
    try:
        af0, afmax, step = 0.02, 0.2, 0.02
        psar = c.copy().astype(float)
        bull = True
        af = af0
        ep = h.iloc[0]
        psar.iloc[0] = l.iloc[0]
        for i in range(1, len(c)):
            prev = psar.iloc[i - 1]
            psar.iloc[i] = prev + af * (ep - prev)
            if bull:
                if l.iloc[i] < psar.iloc[i]:
                    bull = False
                    psar.iloc[i] = ep
                    ep = l.iloc[i]
                    af = af0
                else:
                    if h.iloc[i] > ep:
                        ep = h.iloc[i]
                        af = min(af + step, afmax)
            else:
                if h.iloc[i] > psar.iloc[i]:
                    bull = True
                    psar.iloc[i] = ep
                    ep = h.iloc[i]
                    af = af0
                else:
                    if l.iloc[i] < ep:
                        ep = l.iloc[i]
                        af = min(af + step, afmax)
        out["psar_bull"] = bool(c.iloc[-1] > psar.iloc[-1])
    except Exception:  # noqa: BLE001
        pass

    # --- Ichimoku Cloud ---
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2)
    span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    a_now, b_now = _f(span_a.iloc[-1]), _f(span_b.iloc[-1])
    if a_now is not None and b_now is not None and price is not None:
        top, bot = max(a_now, b_now), min(a_now, b_now)
        out["ichimoku"] = ("über Wolke" if price > top else "unter Wolke" if price < bot else "in Wolke")
    out["tenkan_above_kijun"] = bool(tenkan.iloc[-1] > kijun.iloc[-1])

    # --- Keltner Channel position (EMA20 ± 2*ATR) ---
    ema20 = c.ewm(span=20, adjust=False).mean()
    kc_u = ema20 + 2 * atr14
    kc_l = ema20 - 2 * atr14
    if _f(kc_u.iloc[-1]) and _f(kc_l.iloc[-1]) and (kc_u.iloc[-1] - kc_l.iloc[-1]):
        out["keltner_pos"] = _f((c.iloc[-1] - kc_l.iloc[-1]) / (kc_u.iloc[-1] - kc_l.iloc[-1]) * 100)

    # --- Annualised volatility & max drawdown (1y) ---
    logret = np.log(c / c.shift(1)).dropna()
    out["vol_annual_pct"] = _f(logret.iloc[-252:].std() * np.sqrt(252) * 100) if len(logret) >= 30 else None
    roll_max = c.cummax()
    out["max_drawdown_pct"] = _f(((c / roll_max - 1).min()) * 100)

    # --- Classic Pivot Points (from last completed bar) ---
    P = (h.iloc[-1] + l.iloc[-1] + c.iloc[-1]) / 3
    out["pivot"] = _f(P)
    out["pivot_r1"] = _f(2 * P - l.iloc[-1])
    out["pivot_s1"] = _f(2 * P - h.iloc[-1])

    # --- Fibonacci retracement position within 52-week range ---
    win = min(len(c), 252)
    hi52, lo52 = _f(h.iloc[-win:].max()), _f(l.iloc[-win:].min())
    if hi52 and lo52 and (hi52 - lo52) and price is not None:
        pos = (price - lo52) / (hi52 - lo52)  # 0 = at low, 1 = at high
        out["range_pos_pct"] = round(pos * 100, 1)
        levels = {"0%": lo52, "23.6%": hi52 - 0.236 * (hi52 - lo52),
                  "38.2%": hi52 - 0.382 * (hi52 - lo52), "50%": (hi52 + lo52) / 2,
                  "61.8%": hi52 - 0.618 * (hi52 - lo52), "100%": hi52}
        nearest = min(levels.items(), key=lambda kv: abs(kv[1] - price))
        out["fib_nearest"] = nearest[0]

    return out
