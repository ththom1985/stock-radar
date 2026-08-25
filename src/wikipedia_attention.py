"""Official Wikimedia pageview trend signals with bounded title resolution."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
import urllib.error
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

CACHE_PATH = DATA / "wikipedia_attention.json"
SEARCH_URL = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)
MAX_AGE_HOURS = 24
REQUEST_PAUSE_SECONDS = 1.0


def _request_json(url, retries=3):
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Stock-Radar/1.0 "
                    "(https://github.com/ththom1985/stock-radar)"
                ),
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.replace(".", "", 1).isdigit()
                else 2**attempt
            )
            time.sleep(max(1.0, min(delay, 30.0)))
    raise RuntimeError("unreachable Wikimedia retry state")


def resolve_article(company_name):
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": f'"{company_name}"',
            "srnamespace": "0",
            "srlimit": "5",
            "format": "json",
            "utf8": "1",
        }
    )
    results = ((_request_json(f"{SEARCH_URL}?{query}").get("query") or {}).get("search") or [])
    return select_article(company_name, results)


def select_article(company_name, results):
    if not results:
        return None
    normalized = _normalize(company_name)
    company_tokens = set(normalized.split())
    ranked = []
    for result in results:
        title = result.get("title")
        if not title:
            continue
        title_normalized = _normalize(title)
        overlap = len(company_tokens & set(title_normalized.split()))
        exact = normalized == title_normalized
        minimum_overlap = max(1, math.ceil(len(company_tokens) / 2))
        if not exact and overlap < minimum_overlap:
            continue
        ranked.append(((exact, overlap, -len(title)), title))
    return max(ranked)[1] if ranked else None


def _normalize(value):
    return " ".join(
        token
        for token in "".join(
            character if character.isalnum() else " "
            for character in str(value).casefold()
        ).split()
        if token not in {"inc", "corp", "corporation", "plc", "ltd", "limited", "company"}
    )


def summarize_pageviews(items):
    views = [
        int(item["views"])
        for item in sorted(items or [], key=lambda item: item.get("timestamp") or "")
        if isinstance(item.get("views"), (int, float))
    ]
    if len(views) < 35:
        return None
    recent = views[-7:]
    baseline = views[-35:-7]
    recent_average = sum(recent) / len(recent)
    baseline_average = sum(baseline) / len(baseline)
    change_pct = (
        (recent_average / baseline_average - 1.0) * 100.0
        if baseline_average > 0
        else None
    )
    if change_pct is None or not math.isfinite(change_pct):
        return None
    score = round(max(0.0, min(100.0, 50.0 + change_pct * 0.6)), 1)
    return {
        "score": score,
        "direction": "positive" if score > 55 else "negative" if score < 45 else "neutral",
        "trend_change_pct": round(change_pct, 2),
        "recent_7d_average": round(recent_average, 2),
        "prior_28d_average": round(baseline_average, 2),
        "observation_count": len(views),
    }


def _fetch_pageviews(article):
    now = datetime.now(timezone.utc).date()
    start = (now - timedelta(days=50)).strftime("%Y%m%d")
    end = (now - timedelta(days=1)).strftime("%Y%m%d")
    encoded = urllib.parse.quote(article.replace(" ", "_"), safe="")
    return _request_json(
        PAGEVIEWS_URL.format(article=encoded, start=start, end=end)
    ).get("items") or []


def _fresh(entry):
    timestamp = (entry or {}).get("last_success_at")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)


def fetch_wikipedia_signals(rows, max_new=None, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    eligible = [
        row for row in rows if row.get("asset_type") == "company_equity"
    ]
    eligible.sort(
        key=lambda row: (
            -(row.get("longterm_score") or 0),
            row.get("symbol") or "",
        )
    )
    if max_new is None:
        max_new = int(os.environ.get("STOCK_RADAR_WIKIPEDIA_MAX", "20"))
    candidates = [
        row for row in eligible if force or not _fresh(cache.get(row["symbol"]))
    ][: max(0, max_new)]
    failures = {}
    unsupported = []
    for index, row in enumerate(candidates, 1):
        symbol = row["symbol"]
        try:
            name = (
                row.get("display_name_full")
                or row.get("provider_long_name")
                or row.get("name")
            )
            article = (cache.get(symbol) or {}).get("article") or resolve_article(name)
            if not article:
                unsupported.append(symbol)
                cache[symbol] = {
                    "status": "without_wikipedia_signal",
                    "reason": "no reliable English Wikipedia article match",
                    "fetched_at": utc_now(),
                    "last_success_at": utc_now(),
                }
                continue
            summary = summarize_pageviews(_fetch_pageviews(article))
            if summary is None:
                raise ValueError("fewer than 35 usable daily pageview observations")
            summary.update(
                {
                    "status": "ok",
                    "article": article,
                    "source": "Wikimedia Pageviews API",
                    "source_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(article.replace(' ', '_'))}",
                    "expected_delay": "daily aggregates; latest complete UTC day",
                    "fetched_at": utc_now(),
                    "last_success_at": utc_now(),
                }
            )
            cache[symbol] = clear_cache_failure(summary)
        except Exception as exc:
            cache[symbol] = cache_failure(cache.get(symbol), exc)
            failures[symbol] = str(exc)[:200]
        if verbose:
            print(f"  Wikipedia attention {index}/{len(candidates)}")
        if index < len(candidates):
            time.sleep(REQUEST_PAUSE_SECONDS)
    if candidates:
        cache["_meta"] = schema_meta(
            "stock-radar-wikipedia-attention-cache",
            schema_version=1,
            refreshed=len(candidates),
        )
        atomic_write_json(CACHE_PATH, cache, indent=1)
    return {
        row["symbol"]: cache.get(row["symbol"], {})
        for row in rows
        if isinstance(cache.get(row.get("symbol")), dict)
        and cache.get(row["symbol"], {}).get("status") == "ok"
    }, {
        "status": "ok" if not failures else "partial",
        "eligible": len(eligible),
        "cached": sum(_fresh(cache.get(row["symbol"])) for row in eligible),
        "refreshed": len(candidates),
        "signal_count": sum(
            (cache.get(row["symbol"]) or {}).get("status") == "ok"
            for row in eligible
        ),
        "without_signal": unsupported,
        "failures": failures,
        "max_age_hours": MAX_AGE_HOURS,
    }
