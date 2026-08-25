"""Reliable completed-daily-bar ingestion from Yahoo Finance."""
from __future__ import annotations

import math
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import yfinance as yf

from .config import FETCH_CHUNK, FETCH_PAUSE, HISTORY_PERIOD
from .markets import market_profile, session_is_complete, session_metadata


@dataclass
class PriceFetchResult:
    prices: dict[str, pd.DataFrame] = field(default_factory=dict)
    bar_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_symbols: dict[str, str] = field(default_factory=dict)


def _chunks(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def completed_daily_bars(
    df: pd.DataFrame,
    *,
    now: datetime | None = None,
    symbol: str | None = None,
    close_buffer_minutes: int = 90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return adjusted OHLC using only sessions past a conservative close buffer.

    The yfinance index timezone is preserved when present. Naive indexes use a
    conservative suffix profile; unknown/24x7 markets require the UTC day to end.
    """
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(), {}
    now = now or datetime.now(timezone.utc)
    clean = df.sort_index().dropna(how="all").copy()
    profile = market_profile(symbol, getattr(clean.index, "tz", None))
    keep = [
        session_is_complete(
            ts.date(),
            profile,
            now,
            close_buffer_minutes=close_buffer_minutes,
        )
        for ts in clean.index
    ]
    excluded = int(len(clean) - sum(keep))
    clean = clean.loc[keep]
    if clean.empty or "Close" not in clean.columns:
        return pd.DataFrame(), {"excluded_partial_rows": excluded}

    action_source = clean.copy()
    required_ohlc = ("Open", "High", "Low", "Close")
    if any(column not in clean.columns for column in required_ohlc):
        return pd.DataFrame(), {
            "excluded_partial_rows": excluded,
            "excluded_incomplete_price_rows": len(clean),
        }
    complete_prices = clean.loc[:, required_ohlc].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_price_row = complete_prices.notna().all(axis=1)
    valid_price_row &= (complete_prices > 0).all(axis=1)
    excluded_incomplete = int((~valid_price_row).sum())
    clean = clean.loc[valid_price_row].copy()
    if clean.empty:
        return pd.DataFrame(), {
            "excluded_partial_rows": excluded,
            "excluded_incomplete_price_rows": excluded_incomplete,
        }

    raw_open = clean["Open"].astype(float).copy()
    raw_high = clean["High"].astype(float).copy()
    raw_low = clean["Low"].astype(float).copy()
    raw_close = clean["Close"].astype(float).copy()
    if "Adj Close" in clean.columns:
        factor = (clean["Adj Close"].astype(float) / raw_close.replace(0, float("nan"))).fillna(1.0)
    else:
        factor = pd.Series(1.0, index=clean.index)
    clean["RawOpen"] = raw_open
    clean["RawHigh"] = raw_high
    clean["RawLow"] = raw_low
    clean["RawClose"] = raw_close
    clean["Open"] = raw_open * factor
    clean["High"] = raw_high * factor
    clean["Low"] = raw_low * factor
    clean["Close"] = raw_close * factor
    if "Dividends" not in clean.columns:
        clean["Dividends"] = 0.0
    if "Stock Splits" not in clean.columns:
        clean["Stock Splits"] = 0.0

    last = clean.index[-1]
    timestamp = last.isoformat() if hasattr(last, "isoformat") else str(last)
    actions = []
    for ts, row in action_source.iterrows():
        dividend = float(row.get("Dividends") or 0.0)
        split = float(row.get("Stock Splits") or 0.0)
        dividend = dividend if math.isfinite(dividend) else 0.0
        split = split if math.isfinite(split) else 0.0
        if dividend or split:
            actions.append(
                {
                    "bar_date": ts.date().isoformat(),
                    "bar_timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "dividend_local": dividend,
                    "stock_split": split,
                }
            )
    actions = actions[-128:]
    return clean, {
        "bar_date": last.date().isoformat(),
        "bar_timestamp": timestamp,
        "excluded_partial_rows": excluded,
        "excluded_incomplete_price_rows": excluded_incomplete,
        "source_interval": "1d",
        "completed_bars_only": True,
        "corporate_actions": actions,
        **session_metadata(
            last.date(),
            profile,
            close_buffer_minutes=close_buffer_minutes,
        ),
    }


def _symbol_frame(data: pd.DataFrame, symbol: str, chunk_size: int) -> pd.DataFrame:
    if chunk_size == 1:
        if isinstance(data.columns, pd.MultiIndex):
            for level in range(data.columns.nlevels):
                if symbol in data.columns.get_level_values(level):
                    return data.xs(symbol, axis=1, level=level)
        return data
    try:
        return data[symbol]
    except (KeyError, TypeError):
        if isinstance(data.columns, pd.MultiIndex):
            for level in range(data.columns.nlevels):
                if symbol in data.columns.get_level_values(level):
                    return data.xs(symbol, axis=1, level=level)
        raise


def _default_download(symbols: list[str], period: str) -> pd.DataFrame:
    return yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        threads=True,
        progress=False,
        timeout=30,
    )


def _stooq_symbol(symbol: str) -> str | None:
    """Conservative Stooq mapping for ordinary US ticker symbols only."""
    if not re.fullmatch(r"[A-Z]{1,5}", symbol):
        return None
    return f"{symbol.lower()}.us"


def _default_stooq_download(symbol: str, period: str) -> pd.DataFrame:
    mapped = _stooq_symbol(symbol)
    if not mapped:
        return pd.DataFrame()
    years = 3 if period == "2y" else 6
    end = datetime.now(timezone.utc).date()
    start = end.replace(year=end.year - years)
    query = urllib.parse.urlencode(
        {
            "s": mapped,
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
        }
    )
    frame = pd.read_csv(f"https://stooq.com/q/d/l/?{query}")
    if frame.empty or "Date" not in frame:
        return pd.DataFrame()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    if "Close" in frame:
        frame["Adj Close"] = frame["Close"]
    frame["Dividends"] = 0.0
    frame["Stock Splits"] = 0.0
    return frame


def fetch_prices_with_status(
    symbols: list[str],
    period: str = HISTORY_PERIOD,
    *,
    retries: int = 2,
    now: datetime | None = None,
    downloader: Callable[[list[str], str], pd.DataFrame] | None = None,
    fallback_downloader: Callable[[str, str], pd.DataFrame] | None = None,
    verbose: bool = True,
) -> PriceFetchResult:
    """Download with retries, recursive batch splitting and a failure manifest."""
    result = PriceFetchResult()
    download = downloader or _default_download
    now = now or datetime.now(timezone.utc)

    def ingest(chunk: list[str]) -> None:
        last_error: Exception | None = None
        data: pd.DataFrame | None = None
        for attempt in range(retries + 1):
            try:
                data = download(chunk, period)
                last_error = None
                break
            except Exception as exc:  # provider exceptions vary across releases
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 4))
        if last_error is not None:
            if len(chunk) > 1:
                mid = len(chunk) // 2
                ingest(chunk[:mid])
                ingest(chunk[mid:])
            else:
                result.failed_symbols[chunk[0]] = f"download failed: {str(last_error)[:200]}"
            return

        missing: list[str] = []
        for symbol in chunk:
            try:
                frame = _symbol_frame(data, symbol, len(chunk))
                frame, info = completed_daily_bars(frame, now=now, symbol=symbol)
                if frame.empty or len(frame) < 30:
                    missing.append(symbol)
                    continue
                result.prices[symbol] = frame
                result.bar_info[symbol] = {**info, "price_provider": "yfinance"}
                result.failed_symbols.pop(symbol, None)
            except Exception as exc:
                result.failed_symbols[symbol] = f"invalid response: {str(exc)[:200]}"
                missing.append(symbol)

        # A successful multi-ticker request may still omit/rate-limit individual
        # names. Retry those one by one rather than silently losing a whole slice.
        if len(chunk) > 1:
            for symbol in missing:
                ingest([symbol])
        else:
            for symbol in missing:
                result.failed_symbols.setdefault(symbol, "no usable completed daily bars")

    total = len(symbols)
    for index, chunk in enumerate(_chunks(symbols, FETCH_CHUNK), 1):
        ingest(chunk)
        if verbose:
            done = min(index * FETCH_CHUNK, total)
            print(f"  Batch {index}: {done}/{total} | successful {len(result.prices)}")
        if index * FETCH_CHUNK < total:
            time.sleep(FETCH_PAUSE)

    fallback_enabled = (
        os.environ.get("STOCK_RADAR_STOOQ_FALLBACK", "1") == "1"
        and (fallback_downloader is not None or downloader is None)
    )
    if fallback_enabled:
        fallback = fallback_downloader or _default_stooq_download
        for symbol in list(result.failed_symbols):
            if _stooq_symbol(symbol) is None:
                continue
            try:
                frame = fallback(symbol, period)
                frame, info = completed_daily_bars(frame, now=now, symbol=symbol)
                if frame.empty or len(frame) < 30:
                    continue
                result.prices[symbol] = frame
                result.bar_info[symbol] = {
                    **info,
                    "price_provider": "stooq",
                    "fallback_reason": result.failed_symbols[symbol],
                    "adjustment_status": (
                        "Stooq EOD fallback; adjusted-close/corporate-action parity "
                        "with Yahoo is not claimed"
                    ),
                }
                result.failed_symbols.pop(symbol, None)
            except Exception as exc:
                result.failed_symbols[symbol] = (
                    f"{result.failed_symbols[symbol]}; Stooq fallback failed: "
                    f"{str(exc)[:120]}"
                )
            time.sleep(0.15)
    return result


def fetch_prices(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.DataFrame]:
    """Backward-compatible mapping API; new code should use status metadata."""
    return fetch_prices_with_status(symbols, period).prices
