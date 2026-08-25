"""Free FRED macro regime with explicit credential gating and daily cache."""
from __future__ import annotations

import json
import os
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

CACHE_PATH = DATA / "fred_regime.json"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {
    "dgs10": "DGS10",
    "dgs2": "DGS2",
    "high_yield_spread": "BAMLH0A0HYM2",
    "financial_conditions": "NFCI",
}
MAX_AGE_HOURS = 24


def _fresh(payload):
    timestamp = (payload or {}).get("last_success_at")
    try:
        value = datetime.fromisoformat(str(timestamp))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return value >= datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)


def _latest_observation(series_id, api_key):
    query = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        }
    )
    request = urllib.request.Request(
        f"{FRED_URL}?{query}",
        headers={"User-Agent": "Stock-Radar free macro research"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("observations") or []:
        try:
            return {
                "value": float(item["value"]),
                "observation_date": item["date"],
            }
        except (KeyError, TypeError, ValueError):
            continue
    raise ValueError(f"FRED {series_id} returned no numeric observation")


def build_regime(observations):
    dgs10 = (observations.get("dgs10") or {}).get("value")
    dgs2 = (observations.get("dgs2") or {}).get("value")
    high_yield = (observations.get("high_yield_spread") or {}).get("value")
    nfci = (observations.get("financial_conditions") or {}).get("value")
    curve = dgs10 - dgs2 if all(isinstance(v, (int, float)) for v in (dgs10, dgs2)) else None
    points = []
    reasons = []
    if curve is not None:
        points.append(65 if curve >= 0 else 35)
        reasons.append(f"10Y–2Y Zinskurve {curve:.2f} Prozentpunkte")
    if isinstance(high_yield, (int, float)):
        points.append(75 if high_yield < 3.5 else 50 if high_yield < 5 else 25)
        reasons.append(f"US High-Yield-Spread {high_yield:.2f}%")
    if isinstance(nfci, (int, float)):
        points.append(70 if nfci < -0.25 else 50 if nfci < 0.25 else 25)
        reasons.append(f"NFCI {nfci:.2f}")
    score = round(sum(points) / len(points), 1) if points else None
    regime = (
        "risk_on" if score is not None and score >= 60
        else "risk_off" if score is not None and score <= 40
        else "neutral" if score is not None
        else "unavailable"
    )
    return {
        "score": score,
        "regime": regime,
        "yield_curve_10y_2y": round(curve, 4) if curve is not None else None,
        "reasons": reasons,
        "observations": observations,
        "model_status": "heuristic_context_only",
        "actionable": False,
    }


def fetch_fred_regime(force=False):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if not force and _fresh(cache):
        return cache, {"status": "cached", "max_age_hours": MAX_AGE_HOURS}
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        return cache, {
            "status": "disabled",
            "reason": "FRED_API_KEY is not configured",
            "max_age_hours": MAX_AGE_HOURS,
        }
    try:
        observations = {
            name: _latest_observation(series_id, api_key)
            for name, series_id in SERIES.items()
        }
        result = build_regime(observations)
        result.update(
            {
                "source": "FRED",
                "fetched_at": utc_now(),
                "last_success_at": utc_now(),
            }
        )
        result = clear_cache_failure(result)
        result["_meta"] = schema_meta(
            "stock-radar-fred-regime-cache", schema_version=1
        )
        atomic_write_json(CACHE_PATH, result, indent=1)
        return result, {"status": "ok", "max_age_hours": MAX_AGE_HOURS}
    except Exception as exc:
        failed = cache_failure(cache, exc)
        if failed:
            atomic_write_json(CACHE_PATH, failed, indent=1)
        return failed, {
            "status": "error",
            "reason": str(exc)[:200],
            "max_age_hours": MAX_AGE_HOURS,
        }
