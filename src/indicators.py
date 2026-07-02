"""Technical indicators. Hand-rolled to avoid fragile external TA dependencies."""
import numpy as np
import pandas as pd
from .config import (
    RSI_PERIOD, ATR_PERIOD, SMA_WINDOWS, BB_WINDOW, BB_STD,
    VOL_AVG_WINDOW, BREAKOUT_WINDOW,
)


def _rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df, period=ATR_PERIOD):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _f(x):
    """Safe float conversion; NaN stays NaN."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def compute_features(df):
    """Return a dict of the latest indicator values for one ticker.

    Returns None if there is not enough data to be meaningful.
    """
    if df is None or len(df) < 30:
        return None
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    if close.isna().all():
        return None

    f = {}
    f["price"] = _f(close.iloc[-1])
    f["prev_close"] = _f(close.iloc[-2])
    f["daily_return"] = f["price"] / f["prev_close"] - 1 if f["prev_close"] else np.nan

    # Moving averages
    for w in SMA_WINDOWS:
        f[f"sma{w}"] = _f(close.rolling(w).mean().iloc[-1]) if len(close) >= w else np.nan

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    f["macd"] = _f(macd.iloc[-1])
    f["macd_signal"] = _f(signal.iloc[-1])
    f["macd_hist"] = _f(hist.iloc[-1])
    f["macd_hist_prev"] = _f(hist.iloc[-2])

    # RSI
    f["rsi"] = _f(_rsi(close).iloc[-1])

    # ATR / volatility
    atr = _atr(df)
    f["atr"] = _f(atr.iloc[-1])
    f["atr_pct"] = f["atr"] / f["price"] * 100 if f["price"] else np.nan

    # Bollinger Bands
    mid = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    width = _f((upper.iloc[-1] - lower.iloc[-1]))
    f["bb_pctb"] = ((f["price"] - _f(lower.iloc[-1])) / width) if width else np.nan
    f["bb_bandwidth"] = (width / _f(mid.iloc[-1]) * 100) if _f(mid.iloc[-1]) else np.nan

    # Volume
    f["vol"] = _f(volume.iloc[-1])
    f["vol_avg20"] = _f(volume.rolling(VOL_AVG_WINDOW).mean().iloc[-1])
    f["rvol"] = f["vol"] / f["vol_avg20"] if f["vol_avg20"] else np.nan

    # Breakout levels (prior N-day high/low, excluding today)
    f["high20"] = _f(close.rolling(BREAKOUT_WINDOW).max().iloc[-2])
    f["low20"] = _f(close.rolling(BREAKOUT_WINDOW).min().iloc[-2])

    # 52-week context
    win52 = min(len(close), 252)
    f["high52"] = _f(close.iloc[-win52:].max())
    f["low52"] = _f(close.iloc[-win52:].min())
    f["pct_from_high52"] = (f["price"] / f["high52"] - 1) * 100 if f["high52"] else np.nan

    # Momentum returns
    f["ret_5d"] = (f["price"] / _f(close.iloc[-6]) - 1) * 100 if len(close) > 6 else np.nan
    f["ret_20d"] = (f["price"] / _f(close.iloc[-21]) - 1) * 100 if len(close) > 21 else np.nan
    f["ret_60d"] = (f["price"] / _f(close.iloc[-61]) - 1) * 100 if len(close) > 61 else np.nan

    # Historical daily volatility (stdev of log returns, ~60d) for range projection
    log_ret = np.log(close / close.shift(1)).dropna()
    f["vol_daily"] = _f(log_ret.iloc[-60:].std()) if len(log_ret) >= 20 else np.nan

    # SMA150 + slopes (for Minervini trend template & Weinstein stage analysis)
    sma150 = close.rolling(150).mean()
    sma200_series = close.rolling(200).mean()
    f["sma150"] = _f(sma150.iloc[-1]) if len(close) >= 150 else np.nan
    f["sma150_1m_ago"] = _f(sma150.iloc[-22]) if len(close) >= 172 else np.nan
    f["sma200_1m_ago"] = _f(sma200_series.iloc[-22]) if len(close) >= 222 else np.nan
    f["low52"] = f.get("low52")
    f["pct_above_low52"] = (f["price"] / f["low52"] - 1) * 100 if f.get("low52") else np.nan

    # Merge advanced literature indicators (ADX, Stochastic, MFI, Ichimoku, …)
    try:
        from .tech_advanced import advanced_indicators
        f.update(advanced_indicators(df))
    except Exception:  # noqa: BLE001
        pass

    return f
