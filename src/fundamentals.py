"""Fetch fundamental data via yfinance (.info) with on-disk caching.

Fundamentals change slowly, so we cache per symbol and only refetch entries
older than FUND_MAX_AGE_DAYS. This keeps the daily technical run fast while a
weekly refresh keeps valuations current.
"""
import json
import time
from datetime import datetime, timezone, timedelta

import yfinance as yf

from .config import DATA

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
}


def _load_cache():
    if FUND_CACHE.exists():
        try:
            return json.loads(FUND_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(cache):
    FUND_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


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
        print(f"Fundamentaldaten: {fresh} aus Cache, {len(stale)} werden neu geladen …")

    for i, sym in enumerate(stale, 1):
        try:
            info = yf.Ticker(sym).info
            data = _extract(info)
            data["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            cache[sym] = data
        except Exception as exc:  # noqa: BLE001
            cache[sym] = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "error": str(exc)[:120]}
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(stale)} …")
        time.sleep(FETCH_PAUSE)

    if stale:
        _save_cache(cache)

    return {s: cache.get(s, {}) for s in symbols}
