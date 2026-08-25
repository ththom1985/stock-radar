"""Bounded FINRA consolidated short-interest enrichment."""
from __future__ import annotations

import base64
import json
import math
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

CACHE_PATH = DATA / "finra_short_interest.json"
TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    "?grant_type=client_credentials"
)
DATA_URL = (
    "https://api.finra.org/data/group/otcmarket/name/"
    "consolidatedShortInterest"
)
MAX_AGE_DAYS = 10
REQUEST_PAUSE_SECONDS = 3.1
_EDGE_NOISE = "\ufeff\u200b\u2060 \t\r\n"
_TOKEN_CACHE = {"token": None, "expires_at": 0.0}


def _credential(name):
    value = os.environ.get(name, "").strip(_EDGE_NOISE)
    if value and any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError(f"{name} must contain printable non-whitespace ASCII only")
    return value


def credential_status():
    missing = [
        name
        for name in ("FINRA_CLIENT_ID", "FINRA_CLIENT_SECRET")
        if not _credential(name)
    ]
    return {
        "configured": not missing,
        "missing": missing,
        "registration": "https://gateway.finra.org/app/dfo-console",
    }


def _json_request(url, *, headers=None, payload=None, method=None, timeout=45):
    data = (
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers or {},
        method=method or ("POST" if data is not None else "GET"),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_token():
    now = time.time()
    if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"] - 60:
        return _TOKEN_CACHE["token"]
    status = credential_status()
    if not status["configured"]:
        raise ValueError(
            "Missing FINRA Public API credentials: " + ", ".join(status["missing"])
        )
    raw = f"{_credential('FINRA_CLIENT_ID')}:{_credential('FINRA_CLIENT_SECRET')}"
    authorization = base64.b64encode(raw.encode("ascii")).decode("ascii")
    payload = _json_request(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {authorization}",
            "Accept": "application/json",
        },
        method="POST",
    )
    token = payload.get("access_token")
    if not token:
        raise ValueError("FINRA token response did not contain access_token")
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires_at"] = now + int(payload.get("expires_in") or 3600)
    return token


def fetch_symbol_records(symbol, *, start_date, end_date, token=None):
    token = token or get_token()
    return _json_request(
        DATA_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        payload={
            "limit": 100,
            "compareFilters": [
                {
                    "compareType": "equal",
                    "fieldName": "symbolCode",
                    "fieldValue": symbol,
                }
            ],
            "dateRangeFilters": [
                {
                    "fieldName": "settlementDate",
                    "startDate": start_date,
                    "endDate": end_date,
                }
            ],
        },
    )


def _number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def summarize_records(records):
    clean = []
    for record in records or []:
        settlement_date = str(record.get("settlementDate") or "")[:10]
        current = _number(record.get("currentShortPositionQuantity"))
        if not settlement_date or current is None:
            continue
        clean.append(
            {
                "settlement_date": settlement_date,
                "short_position": current,
                "previous_short_position": _number(
                    record.get("previousShortPositionQuantity")
                ),
                "average_daily_volume": _number(
                    record.get("averageDailyVolumeQuantity")
                ),
                "days_to_cover": _number(record.get("daysToCoverQuantity")),
                "change_percent": _number(record.get("changePercent")),
                "revision": record.get("revisionFlag") == "R",
            }
        )
    clean.sort(key=lambda item: item["settlement_date"])
    clean = clean[-8:]
    if not clean:
        return None
    latest = clean[-1]
    first = clean[0]
    period_trend_pct = (
        (latest["short_position"] / first["short_position"] - 1.0) * 100.0
        if first["short_position"] > 0 and len(clean) > 1
        else None
    )
    change = latest.get("change_percent")
    days_to_cover = latest.get("days_to_cover")
    bearish_pressure = 0.0
    if change is not None:
        bearish_pressure += max(-20.0, min(20.0, change * 0.35))
    if period_trend_pct is not None:
        bearish_pressure += max(-15.0, min(15.0, period_trend_pct * 0.2))
    if days_to_cover is not None:
        bearish_pressure += max(0.0, min(15.0, (days_to_cover - 2.0) * 2.0))
    score = round(max(0.0, min(100.0, 50.0 - bearish_pressure)), 1)
    return {
        "score": score,
        "direction": "positive" if score > 55 else "negative" if score < 45 else "neutral",
        "settlement_date": latest["settlement_date"],
        "short_position": latest["short_position"],
        "days_to_cover": latest.get("days_to_cover"),
        "change_percent": latest.get("change_percent"),
        "period_trend_pct": (
            round(period_trend_pct, 2) if period_trend_pct is not None else None
        ),
        "period_count": len(clean),
        "periods": clean,
        "source": "FINRA consolidatedShortInterest",
        "expected_delay": "twice monthly; publication follows FINRA schedule",
        "limitations": (
            "Short interest is reported twice monthly and is not real-time. "
            "Days-to-cover is FINRA-reported and capped by source conventions."
        ),
    }


def _fresh(entry):
    timestamp = (entry or {}).get("last_success_at")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)


def fetch_finra_signals(rows, max_new=None, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    credentials = credential_status()
    if not credentials["configured"]:
        return {
            row["symbol"]: cache.get(row["symbol"], {})
            for row in rows
            if isinstance(cache.get(row.get("symbol")), dict)
        }, {
            "status": "disabled",
            "reason": "Missing " + ", ".join(credentials["missing"]),
            "registration": credentials["registration"],
            "refreshed": 0,
        }
    eligible = [
        row
        for row in rows
        if row.get("asset_type") == "company_equity"
        and row.get("currency") == "USD"
        and str(row.get("symbol") or "").replace(".", "").isalnum()
    ]
    eligible.sort(
        key=lambda row: (
            -(row.get("entry_timing_score") or 0),
            -(row.get("longterm_score") or 0),
            row.get("symbol") or "",
        )
    )
    if max_new is None:
        max_new = int(os.environ.get("STOCK_RADAR_FINRA_MAX", "15"))
    candidates = [
        row for row in eligible if force or not _fresh(cache.get(row["symbol"]))
    ][: max(0, max_new)]
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=370)
    failures = {}
    token = get_token()
    for index, row in enumerate(candidates, 1):
        symbol = row["symbol"]
        try:
            records = fetch_symbol_records(
                symbol,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                token=token,
            )
            summary = summarize_records(records)
            if summary is None:
                raise ValueError("FINRA returned no consolidated short-interest records")
            summary["fetched_at"] = utc_now()
            summary["last_success_at"] = summary["fetched_at"]
            cache[symbol] = clear_cache_failure(summary)
        except Exception as exc:
            cache[symbol] = cache_failure(cache.get(symbol), exc)
            failures[symbol] = str(exc)[:200]
        if verbose:
            print(f"  FINRA short interest {index}/{len(candidates)}")
        if index < len(candidates):
            time.sleep(REQUEST_PAUSE_SECONDS)
    if candidates:
        cache["_meta"] = schema_meta(
            "stock-radar-finra-short-interest-cache",
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
        "max_age_days": MAX_AGE_DAYS,
        "request_limit_note": "bounded below FINRA 20 requests/minute per dataset",
    }
