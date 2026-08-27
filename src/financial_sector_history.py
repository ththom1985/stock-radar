"""Cached common-equity history for bank and insurance valuation."""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from numbers import Real
from statistics import median

import yfinance as yf

from .config import DATA
from .persistence import (
    atomic_write_json,
    cache_failure,
    clear_cache_failure,
    load_json,
    schema_meta,
    utc_now,
)

HISTORY_PATH = DATA / "financial_sector_history.json"
MAX_AGE_DAYS = 30
MIN_ANNUAL_POINTS = 4
FETCH_PAUSE_SECONDS = 0.15

STANDARD_FINANCIAL_SYMBOLS = frozenset(
    {
        "BEKE",
        "BLK",
        "CME",
        "COIN",
        "FUTU",
        "GOLD",
        "ICE",
        "MA",
        "MCO",
        "MSCI",
        "NDAQ",
        "PYPL",
        "SPGI",
        "TROW",
    }
)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def financial_model_group(row):
    industry = str(row.get("industry_display") or row.get("industry") or "").casefold()
    if "reit" in industry:
        return "reit"
    if "insurance" in industry:
        return "insurance"
    if "bank" in industry:
        return "bank"
    symbol = str(row.get("symbol") or "").upper()
    sector = str(row.get("sector_display") or row.get("sector") or "").casefold()
    if symbol in STANDARD_FINANCIAL_SYMBOLS:
        return "standard"
    if "financial" in sector or "real estate" in sector:
        return "unsupported_financial"
    return "standard"


def _statement_series(frame, candidates):
    if frame is None or getattr(frame, "empty", True):
        return None
    for candidate in candidates:
        if candidate in frame.index:
            return frame.loc[candidate]
    return None


def parse_financial_statements(balance_sheet, income_statement):
    common_equity = _statement_series(
        balance_sheet,
        ("Common Stock Equity", "Stockholders Equity"),
    )
    shares = _statement_series(
        balance_sheet,
        ("Ordinary Shares Number", "Share Issued"),
    )
    net_income = _statement_series(
        income_statement,
        ("Net Income Common Stockholders", "Net Income"),
    )
    if common_equity is None or shares is None or net_income is None:
        return []
    points = []
    for column in balance_sheet.columns:
        if column not in income_statement.columns:
            continue
        equity = _number(common_equity.get(column))
        share_count = _number(shares.get(column))
        income = _number(net_income.get(column))
        if (
            equity is None
            or equity <= 0
            or share_count is None
            or share_count <= 0
            or income is None
        ):
            continue
        try:
            period_end = column.date().isoformat()
        except AttributeError:
            period_end = str(column)[:10]
        points.append(
            {
                "period_end": period_end,
                "common_equity": round(equity, 4),
                "ordinary_shares": round(share_count, 4),
                "net_income_common": round(income, 4),
            }
        )
    return sorted(points, key=lambda point: point["period_end"])[-MIN_ANNUAL_POINTS:]


def _is_fresh(entry, max_age_days):
    if not isinstance(entry, dict):
        return False
    try:
        fetched = datetime.fromisoformat(str(entry.get("last_success_at")))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return fetched >= datetime.now(timezone.utc) - timedelta(days=max_age_days)


def fetch_financial_sector_history(
    rows,
    *,
    max_age_days=MAX_AGE_DAYS,
    force=False,
    verbose=True,
):
    symbols = sorted(
        {
            row.get("symbol")
            for row in rows
            if financial_model_group(row) in {"bank", "insurance"}
            and row.get("symbol")
        }
    )
    cache = load_json(HISTORY_PATH, expected_type=dict, default={})
    stale = [
        symbol
        for symbol in symbols
        if force or not _is_fresh(cache.get(symbol), max_age_days)
    ]
    if verbose:
        print(
            f"Financial sector history: {len(symbols) - len(stale)} cached, "
            f"refreshing {len(stale)} ..."
        )
    for index, symbol in enumerate(stale, 1):
        try:
            ticker = yf.Ticker(symbol)
            annual = parse_financial_statements(
                ticker.balance_sheet,
                ticker.income_stmt,
            )
            if len(annual) < MIN_ANNUAL_POINTS:
                raise ValueError(
                    f"only {len(annual)} clean common-equity years; "
                    f"{MIN_ANNUAL_POINTS} required"
                )
            cache[symbol] = clear_cache_failure(
                {
                    "annual": annual,
                    "source": "Yahoo Finance annual financial statements",
                    "source_url": f"https://finance.yahoo.com/quote/{symbol}/financials/",
                    "history_years": MIN_ANNUAL_POINTS,
                    "last_success_at": utc_now(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary records the error
            cache[symbol] = cache_failure(cache.get(symbol), exc)
        if verbose and (index % 10 == 0 or index == len(stale)):
            print(f"  Financial statements {index}/{len(stale)}")
        time.sleep(FETCH_PAUSE_SECONDS)
    if stale:
        payload = dict(cache)
        payload["_meta"] = schema_meta(
            "stock-radar-financial-sector-history",
            schema_version=1,
            cadence="monthly",
            history_years=MIN_ANNUAL_POINTS,
            common_equity_only=True,
        )
        atomic_write_json(HISTORY_PATH, payload, indent=1)
    records = {symbol: cache.get(symbol, {}) for symbol in symbols}
    return records, {
        "status": "ok",
        "candidate_count": len(symbols),
        "complete_count": sum(
            len((record or {}).get("annual") or []) >= MIN_ANNUAL_POINTS
            for record in records.values()
        ),
        "history_years_required": MIN_ANNUAL_POINTS,
        "source": "Yahoo Finance annual financial statements",
    }


def _close_on_or_before(frame, raw_date):
    try:
        target = datetime.fromisoformat(str(raw_date)).date()
    except (TypeError, ValueError):
        return None
    if frame is None or getattr(frame, "empty", True) or "Close" not in frame:
        return None
    available = [
        float(value)
        for timestamp, value in frame["Close"].dropna().items()
        if timestamp.date() <= target and _number(value) is not None
    ]
    return available[-1] if available else None


def financial_history_baseline(record, price_frame):
    points = []
    for annual in (record or {}).get("annual") or []:
        equity = _number(annual.get("common_equity"))
        shares = _number(annual.get("ordinary_shares"))
        income = _number(annual.get("net_income_common"))
        price = _close_on_or_before(price_frame, annual.get("period_end"))
        if (
            equity is None
            or equity <= 0
            or shares is None
            or shares <= 0
            or income is None
            or income <= 0
            or price is None
            or price <= 0
        ):
            continue
        book_value_per_share = equity / shares
        price_to_book = price / book_value_per_share
        roe = income / equity
        if price_to_book <= 0 or roe <= 0:
            continue
        points.append(
            {
                "period_end": annual.get("period_end"),
                "price": round(price, 4),
                "book_value_per_share": round(book_value_per_share, 4),
                "price_to_book": round(price_to_book, 4),
                "roe_pct": round(roe * 100.0, 4),
                "pb_to_roe": round(price_to_book / roe, 4),
            }
        )
    ratios = [point["pb_to_roe"] for point in points]
    return {
        "complete": len(points) >= MIN_ANNUAL_POINTS,
        "annual_points": points[-MIN_ANNUAL_POINTS:],
        "annual_points_available": len(points),
        "pb_to_roe_median": round(median(ratios), 4) if ratios else None,
        "history_years": MIN_ANNUAL_POINTS,
        "source": (record or {}).get("source"),
    }


def financial_peer_benchmarks(rows):
    grouped = {"bank": [], "insurance": []}
    for row in rows:
        group = financial_model_group(row)
        if group not in grouped:
            continue
        price_to_book = _number(row.get("pb"))
        roe_pct = _number(row.get("roe_pct"))
        if (
            price_to_book is None
            or price_to_book <= 0
            or roe_pct is None
            or roe_pct <= 0
        ):
            continue
        grouped[group].append(price_to_book / (roe_pct / 100.0))
    return {
        group: {
            "pb_to_roe_median": round(median(values), 4) if values else None,
            "peer_count": len(values),
        }
        for group, values in grouped.items()
    }
