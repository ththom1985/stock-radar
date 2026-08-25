"""Direct configured career-page job-count trends."""
from __future__ import annotations

import json
import math
import urllib.request
from datetime import date, datetime, timedelta, timezone

from .config import DATA
from .persistence import (
    atomic_write_json,
    cache_failure,
    clear_cache_failure,
    load_json,
    schema_meta,
    utc_now,
)

CACHE_PATH = DATA / "job_postings.json"
CONFIG_PATH = DATA / "career_pages.json"
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
LEVER_URL = "https://api.lever.co/v0/postings/{token}?mode=json"


def load_career_pages():
    payload = load_json(CONFIG_PATH, required=True, expected_type=dict)
    companies = payload.get("companies")
    if not isinstance(companies, dict):
        raise ValueError("career_pages.json must contain companies")
    return companies


def _request_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Stock-Radar direct public career-page research",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_job_count(config):
    provider = config.get("provider")
    token = config.get("token")
    if provider == "greenhouse":
        payload = _request_json(GREENHOUSE_URL.format(token=token))
        jobs = payload.get("jobs")
    elif provider == "lever":
        jobs = _request_json(LEVER_URL.format(token=token))
    else:
        raise ValueError(f"unsupported career provider {provider!r}")
    if not isinstance(jobs, list):
        raise ValueError("career endpoint did not return a jobs list")
    return len(jobs)


def summarize_job_history(snapshots, today=None):
    today = today or date.today()
    clean = sorted(
        [
            {"date": item["date"], "count": int(item["count"])}
            for item in snapshots
            if item.get("date") and isinstance(item.get("count"), (int, float))
        ],
        key=lambda item: item["date"],
    )
    if not clean:
        return None
    latest = clean[-1]
    baseline_candidates = [
        item
        for item in clean[:-1]
        if date.fromisoformat(item["date"]) <= today - timedelta(days=7)
    ]
    if not baseline_candidates:
        return {
            "status": "collecting_history",
            "open_jobs": latest["count"],
            "history_days": (
                (today - date.fromisoformat(clean[0]["date"])).days
                if clean else 0
            ),
            "snapshots": clean[-60:],
        }
    baseline = baseline_candidates[-1]
    delta = latest["count"] - baseline["count"]
    change_pct = (
        delta / baseline["count"] * 100.0 if baseline["count"] > 0 else None
    )
    if change_pct is None or not math.isfinite(change_pct):
        score = None
    else:
        score = round(max(0.0, min(100.0, 50.0 + change_pct * 0.5)), 1)
    return {
        "status": "ok" if score is not None else "collecting_history",
        "score": score,
        "direction": (
            "positive" if score is not None and score > 55
            else "negative" if score is not None and score < 45
            else "neutral" if score is not None
            else None
        ),
        "open_jobs": latest["count"],
        "baseline_jobs": baseline["count"],
        "baseline_date": baseline["date"],
        "change_count": delta,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "snapshots": clean[-60:],
    }


def fetch_job_signals(rows, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    configured = load_career_pages()
    watchlist = {row.get("symbol") for row in rows}
    active = {
        symbol: config
        for symbol, config in configured.items()
        if symbol in watchlist
    }
    today = datetime.now(timezone.utc).date()
    failures = {}
    for index, (symbol, config) in enumerate(sorted(active.items()), 1):
        entry = cache.get(symbol) or {}
        snapshots = list(entry.get("snapshots") or [])
        already_today = snapshots and snapshots[-1].get("date") == today.isoformat()
        if force or not already_today:
            try:
                count = fetch_job_count(config)
                snapshot = {"date": today.isoformat(), "count": count}
                if already_today:
                    snapshots[-1] = snapshot
                else:
                    snapshots.append(snapshot)
            except Exception as exc:
                cache[symbol] = cache_failure(entry, exc)
                failures[symbol] = str(exc)[:200]
                continue
        summary = summarize_job_history(snapshots, today=today)
        summary.update(
            {
                "provider": config["provider"],
                "source": "Direct company career ATS endpoint",
                "source_url": config["source_url"],
                "expected_delay": "current public job-board inventory; daily snapshot",
                "fetched_at": utc_now(),
                "last_success_at": utc_now(),
            }
        )
        cache[symbol] = clear_cache_failure(summary)
        if verbose:
            print(f"  Job postings {index}/{len(active)}")
    cache["_meta"] = schema_meta(
        "stock-radar-job-postings-cache",
        schema_version=1,
        configured_count=len(configured),
    )
    atomic_write_json(CACHE_PATH, cache, indent=1)
    signals = {
        symbol: cache.get(symbol)
        for symbol in active
        if (cache.get(symbol) or {}).get("status") == "ok"
    }
    coverage = {
        row["symbol"]: (
            cache.get(row["symbol"])
            if row["symbol"] in active
            else {
                "status": "without_jobs_signal",
                "reason": "no reliably configured direct career endpoint",
            }
        )
        for row in rows
    }
    return signals, coverage, {
        "status": "ok" if not failures else "partial",
        "watchlist_count": len(rows),
        "configured_count": len(active),
        "signal_count": len(signals),
        "collecting_history_count": sum(
            (cache.get(symbol) or {}).get("status") == "collecting_history"
            for symbol in active
        ),
        "without_signal_count": len(rows) - len(active),
        "failures": failures,
    }
