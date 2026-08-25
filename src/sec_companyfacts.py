"""Bounded SEC EDGAR Companyfacts enrichment for US-reporting issuers."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import DATA
from .persistence import (
    atomic_write_json,
    cache_failure,
    clear_cache_failure,
    load_json,
    schema_meta,
    utc_now,
)

CACHE_PATH = DATA / "sec_companyfacts.json"
TICKER_MAP_PATH = DATA / "sec_ticker_map.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MAX_AGE_DAYS = 30
TICKER_MAP_MAX_AGE_DAYS = 7
REQUEST_PAUSE_SECONDS = 0.12

TAG_CANDIDATES = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForProceedsFromPropertyPlantAndEquipment",
    ),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "long_term_debt": (
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ),
}


def _is_fresh(entry, max_age_days):
    if not isinstance(entry, dict):
        return False
    timestamp = entry.get("last_success_at") or entry.get("fetched_at")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(days=max_age_days)


def request_sec_json(url, user_agent, timeout=30):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_sec_ticker_map(user_agent, force=False):
    cached = load_json(TICKER_MAP_PATH, expected_type=dict, default={})
    if not force and _is_fresh(cached.get("_meta"), TICKER_MAP_MAX_AGE_DAYS):
        return cached.get("symbols") or {}
    payload = request_sec_json(COMPANY_TICKERS_URL, user_agent)
    symbols = {}
    for item in payload.values():
        ticker = str(item.get("ticker") or "").upper()
        cik = item.get("cik_str")
        if ticker and isinstance(cik, int):
            symbols[ticker] = cik
    atomic_write_json(
        TICKER_MAP_PATH,
        {
            "symbols": symbols,
            "_meta": {
                **schema_meta("stock-radar-sec-ticker-map", schema_version=1),
                "last_success_at": utc_now(),
            },
        },
        indent=1,
    )
    return symbols


def _annual_points(companyfacts, candidates):
    us_gaap = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
    entries = []
    for tag in candidates:
        fact = us_gaap.get(tag) or {}
        units = fact.get("units") or {}
        unit_rows = units.get("USD") or units.get("shares") or []
        for item in unit_rows:
            if item.get("form") not in {"10-K", "20-F", "40-F"}:
                continue
            if item.get("fp") not in {"FY", None}:
                continue
            value = item.get("val")
            if not isinstance(value, (int, float)):
                continue
            entries.append(
                {
                    "tag": tag,
                    "value": float(value),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "filed": item.get("filed"),
                    "form": item.get("form"),
                    "accession": item.get("accn"),
                    "fy": item.get("fy"),
                }
            )
        if entries:
            break
    deduped = {}
    for item in entries:
        end = item.get("end")
        if not end:
            continue
        previous = deduped.get(end)
        if previous is None or str(item.get("filed") or "") > str(previous.get("filed") or ""):
            deduped[end] = item
    return sorted(deduped.values(), key=lambda item: item["end"])[-6:]


def parse_companyfacts(payload):
    """Reduce a raw SEC payload to a bounded, auditable five-year series."""
    series = {
        metric: _annual_points(payload, candidates)
        for metric, candidates in TAG_CANDIDATES.items()
    }
    periods = {}
    for metric, points in series.items():
        for point in points:
            periods.setdefault(point["end"], {})[metric] = point["value"]
            periods[point["end"]].setdefault("filed", point.get("filed"))
            periods[point["end"]].setdefault("form", point.get("form"))
            periods[point["end"]].setdefault("accession", point.get("accession"))
    annual = []
    for end, values in sorted(periods.items()):
        item = {"period_end": end, **values}
        if isinstance(item.get("operating_cash_flow"), (int, float)):
            item["free_cash_flow"] = item["operating_cash_flow"] - max(
                0.0, item.get("capex") or 0.0
            )
        annual.append(item)
    annual = annual[-5:]
    latest = annual[-1] if annual else {}
    prior = annual[-2] if len(annual) >= 2 else {}

    def ratio(numerator, denominator):
        if (
            isinstance(numerator, (int, float))
            and isinstance(denominator, (int, float))
            and denominator
        ):
            return numerator / denominator
        return None

    debt = latest.get("long_term_debt")
    equity = latest.get("equity")
    revenue = latest.get("revenue")
    net_income = latest.get("net_income")
    derived = {
        "profit_margin": ratio(net_income, revenue),
        "return_on_equity": ratio(net_income, equity),
        "return_on_assets": ratio(net_income, latest.get("assets")),
        "debt_to_equity": (
            ratio(debt, equity) * 100.0
            if ratio(debt, equity) is not None
            else None
        ),
        "revenue_growth": (
            ratio(revenue, prior.get("revenue")) - 1.0
            if ratio(revenue, prior.get("revenue")) is not None
            else None
        ),
        "fcf_conversion": ratio(latest.get("free_cash_flow"), net_income),
    }
    return {
        "entity_name": payload.get("entityName"),
        "cik": payload.get("cik"),
        "latest": latest,
        "annual": annual,
        "derived": derived,
        "source": "SEC EDGAR Companyfacts",
        "source_url": (
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(payload.get('cik')):010d}.json"
            if str(payload.get("cik") or "").isdigit()
            else None
        ),
    }


def fetch_sec_companyfacts(symbols, max_new=None, force=False, verbose=True):
    """Refresh a bounded number of SEC-mapped symbols and retain stale-good data."""
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if not user_agent:
        return {
            symbol: cache.get(symbol, {})
            for symbol in symbols
            if isinstance(cache.get(symbol), dict)
        }, {
            "status": "disabled",
            "reason": "SEC_USER_AGENT is required by SEC fair-access policy",
            "refreshed": 0,
        }

    ticker_map = load_sec_ticker_map(user_agent)
    candidates = [
        symbol
        for symbol in symbols
        if symbol.upper() in ticker_map
        and (force or not _is_fresh(cache.get(symbol), MAX_AGE_DAYS))
    ]
    limit = max_new
    if limit is None:
        limit = int(os.environ.get("STOCK_RADAR_SEC_MAX", "25"))
    candidates = candidates[: max(0, limit)]
    failures = {}
    for index, symbol in enumerate(candidates, 1):
        try:
            cik = ticker_map[symbol.upper()]
            payload = request_sec_json(COMPANYFACTS_URL.format(cik=cik), user_agent)
            reduced = parse_companyfacts(payload)
            reduced["fetched_at"] = utc_now()
            reduced["last_success_at"] = reduced["fetched_at"]
            cache[symbol] = clear_cache_failure(reduced)
        except Exception as exc:  # provider errors vary
            cache[symbol] = cache_failure(cache.get(symbol), exc)
            failures[symbol] = str(exc)[:200]
        if verbose and index % 10 == 0:
            print(f"  SEC Companyfacts {index}/{len(candidates)}")
        time.sleep(REQUEST_PAUSE_SECONDS)
    if candidates:
        cache["_meta"] = schema_meta(
            "stock-radar-sec-companyfacts-cache",
            schema_version=1,
            refreshed=len(candidates),
        )
        atomic_write_json(CACHE_PATH, cache, indent=1)
    return {
        symbol: cache.get(symbol, {})
        for symbol in symbols
        if isinstance(cache.get(symbol), dict)
    }, {
        "status": "ok",
        "mapped": sum(symbol.upper() in ticker_map for symbol in symbols),
        "cached": sum(isinstance(cache.get(symbol), dict) for symbol in symbols),
        "refreshed": len(candidates),
        "failures": failures,
        "max_age_days": MAX_AGE_DAYS,
    }


def merge_official_fundamentals(yahoo, sec):
    """Prefer official annual SEC facts for overlapping accounting fields."""
    merged = dict(yahoo or {})
    latest = (sec or {}).get("latest") or {}
    derived = (sec or {}).get("derived") or {}
    source_fields = dict(merged.get("field_sources") or {})
    replacements = {
        "revenue": latest.get("revenue"),
        "free_cashflow": latest.get("free_cash_flow"),
        "profit_margin": derived.get("profit_margin"),
        "roe": derived.get("return_on_equity"),
        "roa": derived.get("return_on_assets"),
        "debt_to_equity": derived.get("debt_to_equity"),
        "revenue_growth": derived.get("revenue_growth"),
    }
    for key, value in replacements.items():
        if isinstance(value, (int, float)):
            merged[key] = value
            source_fields[key] = "sec_companyfacts_latest_annual"
    market_cap = merged.get("market_cap")
    revenue = latest.get("revenue")
    free_cash_flow = latest.get("free_cash_flow")
    if isinstance(market_cap, (int, float)) and isinstance(revenue, (int, float)) and revenue > 0:
        merged["ps"] = market_cap / revenue
        source_fields["ps"] = "yfinance_market_cap/sec_latest_annual_revenue"
    if (
        isinstance(market_cap, (int, float))
        and isinstance(free_cash_flow, (int, float))
        and free_cash_flow > 0
    ):
        merged["price_to_fcf"] = market_cap / free_cash_flow
        source_fields["price_to_fcf"] = "yfinance_market_cap/sec_latest_annual_fcf"
    merged["field_sources"] = source_fields
    merged["sec_companyfacts"] = sec or None
    return merged
