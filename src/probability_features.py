"""Point-in-time price/volume features for the calibrated probability engine.

All price-level relationships are dimensionless except the explicitly requested
log-dollar-liquidity field. ``log_avg_dollar_volume_20`` uses contemporaneous raw
close when available so a later Yahoo back-adjustment cannot change liquidity
observed at the feature date.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

FEATURE_VERSION = "probability-price-spy-ordered-v2"
MIN_HISTORY_BARS = 252

STOCK_FEATURES = (
    "ret_log_5",
    "ret_log_20",
    "ret_log_60",
    "ret_log_126",
    "ret_log_252",
    "price_sma20",
    "price_sma50",
    "price_sma150",
    "price_sma200",
    "price_ema21",
    "sma20_slope_5",
    "sma50_slope_20",
    "sma200_slope_20",
    "rsi14",
    "macd_hist_atr",
    "atr_pct",
    "vol_20",
    "vol_60",
    "vol_252",
    "downside_semivol_60",
    "upside_semivol_60",
    "drawdown_252",
    "dist_high_252",
    "dist_low_252",
    "bb_percent_b",
    "bb_bandwidth",
    "adx14",
    "plus_di14",
    "minus_di14",
    "rvol20",
    "log_avg_dollar_volume_20",
    "prior_pivot_atr_dist",
    "prior_s1_atr_dist",
    "low20_atr_dist",
)

SPY_FEATURES = (
    "spy_ret_log_5",
    "spy_ret_log_20",
    "spy_ret_log_60",
    "spy_ret_log_126",
    "spy_ret_log_252",
    "spy_vol_20",
    "spy_vol_60",
    "spy_drawdown_252",
    "spy_price_sma50",
    "spy_price_sma200",
)

INTERACTION_FEATURES = (
    "interaction_spy_above_sma200_x_ret_60d",
    "interaction_spy_above_sma200_x_price_to_sma200_minus1",
    "interaction_spy_vol60_x_vol60",
    "interaction_spy_vol60_x_trailing_drawdown",
    "interaction_spy_above_sma200_x_downside_semivol",
)

FEATURE_NAMES = STOCK_FEATURES + SPY_FEATURES + INTERACTION_FEATURES

PROBABILITY_CODE_FILES = (
    "src/probability_contract.py",
    "src/probability_features.py",
    "src/probability_dataset.py",
    "src/probability_model.py",
    "src/probability_ordered.py",
    "src/probability_inference.py",
    "src/probability_train.py",
    "src/fetch.py",
    "src/markets.py",
    "src/config.py",
    "src/persistence.py",
    "src/analyze.py",
    "src/assets.py",
    "src/fx.py",
    "requirements-ci.txt",
    "requirements.txt",
)

_SCHEMA_DEFINITION = {
    "version": FEATURE_VERSION,
    "features": FEATURE_NAMES,
    "returns": "log adjusted-close ratios ending at completed session t",
    "volatility": "annualized standard deviation of daily adjusted log returns",
    "atr": "Wilder 14-session true range on adjusted OHLC",
    "adx": "Wilder DMI/ADX(14)",
    "market_alignment": "last completed SPY session at or before stock session t",
    "liquidity": "log mean(raw contemporaneous close * reported volume, 20)",
    "frozen_interactions": {
        "interaction_spy_above_sma200_x_ret_60d": (
            "I(spy_price_sma200>=0) * ret_log_60"
        ),
        "interaction_spy_above_sma200_x_price_to_sma200_minus1": (
            "I(spy_price_sma200>=0) * price_sma200"
        ),
        "interaction_spy_vol60_x_vol60": "spy_vol_60 * vol_60",
        "interaction_spy_vol60_x_trailing_drawdown": (
            "spy_vol_60 * drawdown_252"
        ),
        "interaction_spy_above_sma200_x_downside_semivol": (
            "I(spy_price_sma200>=0) * downside_semivol_60"
        ),
    },
}


def feature_schema_hash() -> str:
    encoded = json.dumps(
        _SCHEMA_DEFINITION, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probability_code_hash(
    root: Path | None = None,
    names: Iterable[str] = PROBABILITY_CODE_FILES,
) -> str:
    """Portable semantic source hash (CRLF/LF and platform path invariant)."""
    source_root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for name in sorted(names):
        logical_name = str(name).replace("\\", "/")
        path = source_root / Path(logical_name)
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            text = path.read_text(encoding="utf-8-sig")
            canonical = text.replace("\r\n", "\n").replace("\r", "\n")
            digest.update(canonical.encode("utf-8"))
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    digest.update(feature_schema_hash().encode("ascii"))
    return digest.hexdigest()


def dependency_versions(root: Path | None = None) -> dict[str, Any]:
    project_root = Path(root) if root is not None else Path(__file__).resolve().parent.parent

    def pinned(path: Path) -> list[str]:
        if not path.exists():
            return ["<missing>"]
        return sorted(
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    return {
        "python": "3.12",
        "training_ci": pinned(project_root / "requirements-ci.txt"),
        "dashboard": pinned(project_root / "requirements.txt"),
        "feature_version": FEATURE_VERSION,
        "feature_schema_hash": feature_schema_hash(),
    }


def _canonical_history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(history.columns):
        missing = sorted(required - set(history.columns))
        raise ValueError(f"OHLCV history is missing columns: {missing}")
    frame = history.copy().sort_index()
    index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    valid = ~index.isna()
    frame = frame.loc[valid].copy()
    index = index[valid].tz_convert(None).normalize()
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="last")]
    numeric = set(required) | {"RawClose"}
    for column in numeric.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _wilder(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame["Close"].shift(1)
    return pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous).abs(),
            (frame["Low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = _wilder(change.clip(lower=0), period)
    loss = _wilder((-change).clip(lower=0), period)
    denominator = gain + loss
    result = 100.0 * gain / denominator.replace(0, np.nan)
    both_flat = (gain == 0) & (loss == 0)
    result = result.where(~both_flat, 50.0)
    result = result.where(~((loss == 0) & (gain > 0)), 100.0)
    result = result.where(~((gain == 0) & (loss > 0)), 0.0)
    return result


def _dmi(
    frame: pd.DataFrame, atr: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    upward = frame["High"].diff()
    downward = -frame["Low"].diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder(minus_dm, period) / atr.replace(0, np.nan)
    total = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / total.replace(0, np.nan)
    adx = _wilder(dx, period)
    return adx, plus_di, minus_di


def _semivol(values: np.ndarray, *, positive: bool) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    selected = np.maximum(finite, 0.0) if positive else np.minimum(finite, 0.0)
    return float(math.sqrt(252.0 * np.mean(np.square(selected))))


def _stock_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = _canonical_history(history)
    if frame.empty:
        return pd.DataFrame(index=frame.index, columns=STOCK_FEATURES, dtype=float)

    close = frame["Close"].where(frame["Close"] > 0)
    high = frame["High"].where(frame["High"] > 0)
    low = frame["Low"].where(frame["Low"] > 0)
    volume = frame["Volume"].where(frame["Volume"] >= 0)
    log_close = np.log(close)
    daily_log_return = log_close.diff()
    output = pd.DataFrame(index=frame.index)

    for window in (5, 20, 60, 126, 252):
        output[f"ret_log_{window}"] = log_close - log_close.shift(window)

    moving = {}
    for window in (20, 50, 150, 200):
        moving[window] = close.rolling(window, min_periods=window).mean()
        output[f"price_sma{window}"] = (
            close / moving[window].replace(0, np.nan) - 1.0
        )
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    output["price_ema21"] = close / ema21.replace(0, np.nan) - 1.0
    output["sma20_slope_5"] = (
        moving[20] / moving[20].shift(5).replace(0, np.nan) - 1.0
    ) / 5.0
    output["sma50_slope_20"] = (
        moving[50] / moving[50].shift(20).replace(0, np.nan) - 1.0
    ) / 20.0
    output["sma200_slope_20"] = (
        moving[200] / moving[200].shift(20).replace(0, np.nan) - 1.0
    ) / 20.0

    output["rsi14"] = _rsi(close)
    true_range = _true_range(frame.assign(Close=close, High=high, Low=low))
    atr = _wilder(true_range, 14)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    output["macd_hist_atr"] = (macd - macd_signal) / atr.replace(0, np.nan)
    output["atr_pct"] = atr / close.replace(0, np.nan)

    for window in (20, 60, 252):
        output[f"vol_{window}"] = (
            daily_log_return.rolling(window, min_periods=window).std(ddof=1)
            * math.sqrt(252.0)
        )
    output["downside_semivol_60"] = daily_log_return.rolling(
        60, min_periods=60
    ).apply(lambda values: _semivol(values, positive=False), raw=True)
    output["upside_semivol_60"] = daily_log_return.rolling(
        60, min_periods=60
    ).apply(lambda values: _semivol(values, positive=True), raw=True)

    high252 = close.rolling(252, min_periods=252).max()
    low252 = close.rolling(252, min_periods=252).min()
    output["drawdown_252"] = close / high252.replace(0, np.nan) - 1.0
    output["dist_high_252"] = close / high252.replace(0, np.nan) - 1.0
    output["dist_low_252"] = close / low252.replace(0, np.nan) - 1.0

    std20 = close.rolling(20, min_periods=20).std(ddof=1)
    upper = moving[20] + 2.0 * std20
    lower = moving[20] - 2.0 * std20
    output["bb_percent_b"] = (close - lower) / (upper - lower).replace(0, np.nan)
    output["bb_bandwidth"] = (upper - lower) / moving[20].replace(0, np.nan)

    adx, plus_di, minus_di = _dmi(
        frame.assign(Close=close, High=high, Low=low), atr
    )
    output["adx14"] = adx
    output["plus_di14"] = plus_di
    output["minus_di14"] = minus_di
    output["rvol20"] = volume / volume.rolling(20, min_periods=20).mean().replace(
        0, np.nan
    )

    raw_close = (
        frame["RawClose"].where(frame["RawClose"] > 0)
        if "RawClose" in frame
        else close
    )
    dollar_volume = raw_close * volume
    average_dollar_volume = dollar_volume.rolling(20, min_periods=20).mean()
    output["log_avg_dollar_volume_20"] = np.log(
        average_dollar_volume.where(average_dollar_volume > 0)
    )

    prior_pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
    prior_s1 = 2.0 * prior_pivot - high.shift(1)
    low20 = low.rolling(20, min_periods=20).min()
    output["prior_pivot_atr_dist"] = (close - prior_pivot) / atr.replace(0, np.nan)
    output["prior_s1_atr_dist"] = (close - prior_s1) / atr.replace(0, np.nan)
    output["low20_atr_dist"] = (close - low20) / atr.replace(0, np.nan)
    return output.loc[:, STOCK_FEATURES].replace([np.inf, -np.inf], np.nan)


def _market_features(spy_history: pd.DataFrame) -> pd.DataFrame:
    frame = _canonical_history(spy_history)
    if frame.empty:
        return pd.DataFrame(index=frame.index, columns=SPY_FEATURES, dtype=float)
    close = frame["Close"].where(frame["Close"] > 0)
    log_close = np.log(close)
    returns = log_close.diff()
    output = pd.DataFrame(index=frame.index)
    for window in (5, 20, 60, 126, 252):
        output[f"spy_ret_log_{window}"] = log_close - log_close.shift(window)
    output["spy_vol_20"] = (
        returns.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(252.0)
    )
    output["spy_vol_60"] = (
        returns.rolling(60, min_periods=60).std(ddof=1) * math.sqrt(252.0)
    )
    high252 = close.rolling(252, min_periods=252).max()
    output["spy_drawdown_252"] = close / high252.replace(0, np.nan) - 1.0
    for window in (50, 200):
        average = close.rolling(window, min_periods=window).mean()
        output[f"spy_price_sma{window}"] = close / average.replace(0, np.nan) - 1.0
    return output.loc[:, SPY_FEATURES].replace([np.inf, -np.inf], np.nan)


def build_probability_features(
    history: pd.DataFrame,
    spy_history: pd.DataFrame,
) -> pd.DataFrame:
    """Return the stable feature schema using information available by each t."""
    stock = _stock_features(history)
    market = _market_features(spy_history)
    if stock.empty:
        return pd.DataFrame(index=stock.index, columns=FEATURE_NAMES, dtype=float)
    left = pd.DataFrame({"stock_timestamp": stock.index}).sort_values(
        "stock_timestamp"
    )
    right = market.sort_index().rename_axis("spy_timestamp").reset_index()
    aligned_market = pd.merge_asof(
        left,
        right,
        left_on="stock_timestamp",
        right_on="spy_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    available = aligned_market["spy_timestamp"].notna()
    if (
        aligned_market.loc[available, "spy_timestamp"]
        > aligned_market.loc[available, "stock_timestamp"]
    ).any():
        raise AssertionError("SPY as-of alignment selected a future session")
    aligned_market = aligned_market.set_index("stock_timestamp")
    aligned_market = aligned_market.drop(columns=["spy_timestamp"])
    aligned_market = aligned_market.reindex(stock.index)
    result = pd.concat([stock, aligned_market], axis=1)
    spy_above_sma200 = (result["spy_price_sma200"] >= 0.0).astype(float)
    result["interaction_spy_above_sma200_x_ret_60d"] = (
        spy_above_sma200 * result["ret_log_60"]
    )
    result["interaction_spy_above_sma200_x_price_to_sma200_minus1"] = (
        spy_above_sma200 * result["price_sma200"]
    )
    result["interaction_spy_vol60_x_vol60"] = (
        result["spy_vol_60"] * result["vol_60"]
    )
    result["interaction_spy_vol60_x_trailing_drawdown"] = (
        result["spy_vol_60"] * result["drawdown_252"]
    )
    result["interaction_spy_above_sma200_x_downside_semivol"] = (
        spy_above_sma200 * result["downside_semivol_60"]
    )
    return result.loc[:, FEATURE_NAMES]


def latest_probability_features(
    history: pd.DataFrame,
    spy_history: pd.DataFrame,
    *,
    as_of: object | None = None,
) -> tuple[pd.Timestamp, dict[str, float]]:
    """Return a complete latest point-in-time vector or raise explicitly."""
    frame = build_probability_features(history, spy_history)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)
        frame = frame.loc[frame.index <= cutoff.normalize()]
    if frame.empty:
        raise ValueError("no feature session is available")
    timestamp = frame.index[-1]
    row = frame.iloc[-1]
    missing = [name for name in FEATURE_NAMES if not np.isfinite(row[name])]
    if missing:
        raise ValueError(f"missing probability features at {timestamp.date()}: {missing}")
    return timestamp, {name: float(row[name]) for name in FEATURE_NAMES}


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "INTERACTION_FEATURES",
    "MIN_HISTORY_BARS",
    "PROBABILITY_CODE_FILES",
    "SPY_FEATURES",
    "STOCK_FEATURES",
    "build_probability_features",
    "dependency_versions",
    "feature_schema_hash",
    "latest_probability_features",
    "probability_code_hash",
]
