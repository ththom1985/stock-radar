"""Official free CFTC TFF market-positioning context."""
from __future__ import annotations

import json
import math
import urllib.parse
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

CACHE_PATH = DATA / "market_positioning.json"
CFTC_URL = "https://publicreporting.cftc.gov/resource/yw9f-hn96.json"
CONTRACTS = {
    "sp500": {"code": "13874+", "label": "S&P 500 Consolidated", "invert": False},
    "nasdaq100": {"code": "20974+", "label": "NASDAQ-100 Consolidated", "invert": False},
    "vix": {"code": "1170E1", "label": "VIX Futures", "invert": True},
}
MAX_AGE_DAYS = 7
CBOE_STATUS = {
    "status": "official_free_machine_endpoint_unavailable",
    "checked_at": "2026-08-25",
    "legacy_csv": "HTTP 404",
    "current_cdn_api": "HTTP 403 without browser session",
    "official_commercial_channel": "Cboe DataShop",
    "substitute": "CFTC VIX Futures TFF positioning (official and free)",
}


def _request_contract(code):
    query = urllib.parse.urlencode(
        {
            "$where": f"cftc_contract_market_code='{code}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": "8",
            "$select": (
                "report_date_as_yyyy_mm_dd,market_and_exchange_names,"
                "asset_mgr_positions_long,asset_mgr_positions_short,"
                "lev_money_positions_long,lev_money_positions_short,"
                "open_interest_all"
            ),
        }
    )
    request = urllib.request.Request(
        f"{CFTC_URL}?{query}",
        headers={"User-Agent": "Stock-Radar public CFTC research"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def summarize_contract(records, *, invert=False):
    clean = []
    for record in records or []:
        oi = _number(record.get("open_interest_all"))
        asset_long = _number(record.get("asset_mgr_positions_long"))
        asset_short = _number(record.get("asset_mgr_positions_short"))
        lev_long = _number(record.get("lev_money_positions_long"))
        lev_short = _number(record.get("lev_money_positions_short"))
        if not all(
            value is not None for value in (oi, asset_long, asset_short, lev_long, lev_short)
        ) or oi <= 0:
            continue
        clean.append(
            {
                "report_date": str(record.get("report_date_as_yyyy_mm_dd"))[:10],
                "market": record.get("market_and_exchange_names"),
                "open_interest": oi,
                "asset_manager_net_pct_oi": (asset_long - asset_short) / oi * 100.0,
                "leveraged_money_net_pct_oi": (lev_long - lev_short) / oi * 100.0,
            }
        )
    clean.sort(key=lambda item: item["report_date"], reverse=True)
    if not clean:
        return None
    latest = clean[0]
    previous = clean[1] if len(clean) > 1 else None
    raw = (
        latest["asset_manager_net_pct_oi"] * 1.2
        + latest["leveraged_money_net_pct_oi"] * 0.8
    )
    if invert:
        raw *= -1.0
    score = round(max(0.0, min(100.0, 50.0 + raw)), 1)
    return {
        "score": score,
        "direction": "risk_on" if score >= 60 else "risk_off" if score <= 40 else "neutral",
        "report_date": latest["report_date"],
        "asset_manager_net_pct_oi": round(latest["asset_manager_net_pct_oi"], 3),
        "leveraged_money_net_pct_oi": round(
            latest["leveraged_money_net_pct_oi"], 3
        ),
        "asset_manager_weekly_change": (
            round(
                latest["asset_manager_net_pct_oi"]
                - previous["asset_manager_net_pct_oi"],
                3,
            )
            if previous
            else None
        ),
        "leveraged_money_weekly_change": (
            round(
                latest["leveraged_money_net_pct_oi"]
                - previous["leveraged_money_net_pct_oi"],
                3,
            )
            if previous
            else None
        ),
        "periods": clean,
    }


def build_positioning_context(contract_data):
    available = {
        name: value
        for name, value in contract_data.items()
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float))
    }
    score = (
        round(sum(value["score"] for value in available.values()) / len(available), 1)
        if available
        else None
    )
    latest_date = max(
        (value["report_date"] for value in available.values()), default=None
    )
    return {
        "score": score,
        "regime": (
            "risk_on" if score is not None and score >= 60
            else "risk_off" if score is not None and score <= 40
            else "neutral" if score is not None
            else "unavailable"
        ),
        "report_date": latest_date,
        "contracts": contract_data,
        "source": "CFTC Traders in Financial Futures Combined",
        "expected_delay": "Tuesday positions published Friday (~3 calendar days)",
        "model_status": "heuristic_context_only",
        "actionable": False,
        "cboe_put_call": CBOE_STATUS,
    }


def _fresh(cache):
    timestamp = cache.get("last_success_at")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)


def fetch_market_positioning(force=False):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if not force and _fresh(cache):
        return cache, {"status": "cached", "max_age_days": MAX_AGE_DAYS}
    failures = {}
    contracts = {}
    for name, config in CONTRACTS.items():
        try:
            contracts[name] = summarize_contract(
                _request_contract(config["code"]), invert=config["invert"]
            )
            if contracts[name] is None:
                raise ValueError("CFTC returned no usable TFF rows")
            contracts[name]["label"] = config["label"]
        except Exception as exc:
            failures[name] = str(exc)[:200]
            contracts[name] = None
    context = build_positioning_context(contracts)
    now = utc_now()
    context.update({"fetched_at": now, "last_success_at": now})
    if failures and not any(contracts.values()):
        failed = cache_failure(cache, RuntimeError(str(failures)))
        return failed, {"status": "error", "failures": failures}
    context = clear_cache_failure(context)
    context["_meta"] = schema_meta(
        "stock-radar-market-positioning-cache", schema_version=1
    )
    atomic_write_json(CACHE_PATH, context, indent=1)
    return context, {
        "status": "ok" if not failures else "partial",
        "contract_count": sum(value is not None for value in contracts.values()),
        "failures": failures,
        "report_date": context.get("report_date"),
        "cboe_put_call": CBOE_STATUS,
        "max_age_days": MAX_AGE_DAYS,
    }
