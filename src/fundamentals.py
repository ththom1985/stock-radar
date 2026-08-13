"""Fetch fundamental data via yfinance (.info) with on-disk caching.

Fundamentals change slowly, so we cache per symbol and only refetch entries
older than FUND_MAX_AGE_DAYS. This keeps the daily technical run fast while a
weekly refresh keeps valuations current.
"""
import time
from datetime import datetime, timezone, timedelta

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

FUND_CACHE = DATA / "fundamentals.json"
FUND_MAX_AGE_DAYS = 7
FETCH_PAUSE = 0.3

# yfinance .info key -> our short name
_FIELDS = {
    "trailingPE": "pe",
    "forwardPE": "forward_pe",
    "priceToBook": "pb",
    "priceToSalesTrailing12Months": "ps",
    "enterpriseToEbitda": "ev_ebitda",
    "trailingPegRatio": "peg",
    "returnOnEquity": "roe",
    "returnOnAssets": "roa",
    "profitMargins": "profit_margin",
    "grossMargins": "gross_margin",
    "operatingMargins": "operating_margin",
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "dividendYield": "dividend_yield",
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "ebitda": "ebitda",
    "freeCashflow": "free_cashflow",
    "sector": "sector",
    "industry": "industry",
    # Analyst consensus
    "recommendationMean": "rec_mean",     # 1=Strong Buy … 5=Sell
    "recommendationKey": "rec_key",
    "numberOfAnalystOpinions": "analyst_n",
    "targetMeanPrice": "target_price",
    # For Graham number / FCF yield / Rule of 40 / risk
    "trailingEps": "eps",
    "bookValue": "bvps",
    "beta": "beta",
    "totalRevenue": "revenue",
    "quoteType": "quote_type",
    "currency": "reported_currency",
    "uuid": "issuer_uuid",
    "longName": "provider_long_name",
    "country": "provider_country",
}


def _load_cache():
    return load_json(FUND_CACHE, expected_type=dict, default={})


def _save_cache(cache):
    cache = dict(cache)
    cache["_meta"] = schema_meta("stock-radar-fundamentals-cache")
    atomic_write_json(FUND_CACHE, cache, indent=1)


def _extract(info):
    out = {}
    for src, dst in _FIELDS.items():
        val = info.get(src)
        if isinstance(val, (int, float)) or isinstance(val, str):
            out[dst] = val
    return out


def fetch_fundamentals(symbols, max_age_days=FUND_MAX_AGE_DAYS, force=False, verbose=True):
    """Return dict symbol -> fundamentals. Refetches only stale/missing symbols."""
    cache = _load_cache()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    stale = []
    for s in symbols:
        entry = cache.get(s)
        if force or not entry:
            stale.append(s)
            continue
        try:
            fetched = datetime.fromisoformat(entry.get("fetched_at", "1970-01-01T00:00:00+00:00"))
            if fetched < cutoff:
                stale.append(s)
        except Exception:  # noqa: BLE001
            stale.append(s)

    if verbose:
        fresh = len(symbols) - len(stale)
        print(f"Fundamentals: {fresh} cached, refreshing {len(stale)} ...")

    for i, sym in enumerate(stale, 1):
        try:
            info = yf.Ticker(sym).info
            data = _extract(info)
            if not data:
                raise ValueError("provider returned no usable fundamental fields")
            data["fetched_at"] = utc_now()
            data["last_success_at"] = data["fetched_at"]
            cache[sym] = clear_cache_failure(data)
        except Exception as exc:  # noqa: BLE001
            cache[sym] = cache_failure(cache.get(sym), exc)
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(stale)} ...")
        time.sleep(FETCH_PAUSE)

    if stale:
        _save_cache(cache)

    return {s: cache.get(s, {}) for s in symbols}
