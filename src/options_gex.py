"""Self-computed options gamma exposure from bounded yfinance chains."""
from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone

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

CACHE_PATH = DATA / "options_gex.json"
MAX_AGE_HOURS = 18
MAX_EXPIRATIONS = 4
MAX_DAYS_TO_EXPIRY = 120
RISK_FREE_RATE = 0.04


def normal_pdf(value):
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def black_scholes_gamma(spot, strike, time_years, volatility, rate=RISK_FREE_RATE):
    if min(spot, strike, time_years, volatility) <= 0:
        return None
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * time_years
    ) / (volatility * math.sqrt(time_years))
    return normal_pdf(d1) / (spot * volatility * math.sqrt(time_years))


def calculate_gex(spot, expirations, market_cap=None, now=None):
    """Aggregate call-positive/put-negative gamma under a documented dealer proxy."""
    now = now or datetime.now(timezone.utc)
    by_strike = {}
    contracts = 0
    total = 0.0
    for expiration in expirations:
        expiry = expiration.get("expiration")
        try:
            expiry_date = datetime.fromisoformat(str(expiry)).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        time_years = max(
            (expiry_date - now).total_seconds() / (365.25 * 86400),
            1.0 / 365.25,
        )
        for option_type, sign in (("calls", 1.0), ("puts", -1.0)):
            for contract in expiration.get(option_type) or []:
                strike = contract.get("strike")
                open_interest = contract.get("openInterest")
                volatility = contract.get("impliedVolatility")
                if not all(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                    for value in (strike, open_interest, volatility)
                ):
                    continue
                gamma = black_scholes_gamma(
                    spot, strike, time_years, volatility
                )
                if gamma is None:
                    continue
                exposure = (
                    sign * gamma * open_interest * 100.0 * spot * spot * 0.01
                )
                total += exposure
                by_strike[strike] = by_strike.get(strike, 0.0) + exposure
                contracts += 1
    walls = sorted(
        (
            {"strike": strike, "gex_usd_per_1pct": round(value, 2)}
            for strike, value in by_strike.items()
        ),
        key=lambda item: abs(item["gex_usd_per_1pct"]),
        reverse=True,
    )[:8]
    ratio = (
        total / market_cap
        if isinstance(market_cap, (int, float)) and market_cap > 0
        else None
    )
    score = (
        round(max(0.0, min(100.0, 50.0 + math.tanh(ratio * 100.0) * 40.0)), 1)
        if ratio is not None
        else None
    )
    return {
        "score": score,
        "direction": (
            "dampening" if total > 0 else "amplifying" if total < 0 else "neutral"
        ),
        "net_gex_usd_per_1pct": round(total, 2),
        "gex_to_market_cap": round(ratio, 8) if ratio is not None else None,
        "contract_rows_used": contracts,
        "gamma_walls": walls,
        "method": (
            "Black-Scholes gamma × open interest × 100 shares × spot² × 1%; "
            "calls positive, puts negative as a dealer-position proxy"
        ),
        "limitations": (
            "Yahoo open interest/IV can be delayed; dealer sign is assumed, zero-DTE "
            "intraday changes and customer positioning are not observable"
        ),
    }


def _fresh(entry):
    timestamp = (entry or {}).get("last_success_at")
    try:
        value = datetime.fromisoformat(str(timestamp))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return value >= datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)


def _chain_rows(frame):
    columns = ("strike", "openInterest", "impliedVolatility")
    return [
        {
            column: (
                float(record[column])
                if isinstance(record.get(column), (int, float))
                and math.isfinite(record[column])
                else None
            )
            for column in columns
        }
        for record in frame.loc[:, list(columns)].to_dict("records")
    ]


def fetch_gex_signals(rows, max_symbols=None, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if max_symbols is None:
        max_symbols = int(os.environ.get("STOCK_RADAR_OPTIONS_MAX", "5"))
    eligible = sorted(
        [
            row
            for row in rows
            if row.get("asset_type") == "company_equity"
            and row.get("currency") == "USD"
            and isinstance(row.get("price_local"), (int, float))
        ],
        key=lambda row: (
            -(row.get("entry_timing_score") or 0),
            -(row.get("longterm_score") or 0),
            row.get("symbol") or "",
        ),
    )
    candidates = [
        row for row in eligible if force or not _fresh(cache.get(row["symbol"]))
    ][: max(0, max_symbols)]
    failures = {}
    now = datetime.now(timezone.utc)
    for index, row in enumerate(candidates, 1):
        symbol = row["symbol"]
        try:
            ticker = yf.Ticker(symbol)
            expirations = []
            for expiration in list(ticker.options)[:12]:
                expiry = datetime.fromisoformat(expiration).replace(
                    tzinfo=timezone.utc
                )
                if expiry <= now or expiry > now + timedelta(days=MAX_DAYS_TO_EXPIRY):
                    continue
                chain = ticker.option_chain(expiration)
                expirations.append(
                    {
                        "expiration": expiration,
                        "calls": _chain_rows(chain.calls),
                        "puts": _chain_rows(chain.puts),
                    }
                )
                if len(expirations) >= MAX_EXPIRATIONS:
                    break
                time.sleep(0.15)
            if not expirations:
                raise ValueError("no bounded option expirations available")
            result = calculate_gex(
                row["price_local"],
                expirations,
                market_cap=row.get("market_cap_local"),
                now=now,
            )
            result.update(
                {
                    "source": "Yahoo Finance options via yfinance",
                    "fetched_at": utc_now(),
                    "last_success_at": utc_now(),
                    "expected_delay": "provider dependent; not exchange-certified real time",
                    "expiration_count": len(expirations),
                }
            )
            cache[symbol] = clear_cache_failure(result)
        except Exception as exc:  # provider response shapes vary
            cache[symbol] = cache_failure(cache.get(symbol), exc)
            failures[symbol] = str(exc)[:200]
        if verbose:
            print(f"  Options GEX {index}/{len(candidates)}")
        time.sleep(0.25)
    if candidates:
        cache["_meta"] = schema_meta(
            "stock-radar-options-gex-cache",
            schema_version=1,
            refreshed=len(candidates),
        )
        atomic_write_json(CACHE_PATH, cache, indent=1)
    return {
        row["symbol"]: cache.get(row["symbol"], {})
        for row in rows
        if isinstance(cache.get(row.get("symbol")), dict)
    }, {
        "status": "ok",
        "eligible": len(eligible),
        "cached": sum(isinstance(cache.get(row["symbol"]), dict) for row in eligible),
        "refreshed": len(candidates),
        "failures": failures,
        "max_age_hours": MAX_AGE_HOURS,
    }
