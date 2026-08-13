"""Bounded, age-filtered news context with stale-good caching."""
from __future__ import annotations

import concurrent.futures as cf
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser

from .config import DATA
from .persistence import atomic_write_json, cache_failure, load_json, schema_meta, utc_now

YF_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
MARKET_FEEDS = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.investing.com/rss/news.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
]
NEWS_CACHE = DATA / "news_cache.json"
MAX_NEWS_AGE_HOURS = 168
HTTP_TIMEOUT_SECONDS = 15

POS = {
    "beat", "beats", "surge", "surges", "jump", "jumps", "soar", "soars", "rally",
    "rallies", "upgrade", "upgraded", "raise", "raised", "record", "strong", "growth",
    "profit", "profits", "wins", "win", "deal", "approval", "approved", "breakthrough",
    "outperform", "bullish", "gain", "gains", "rebound", "expand", "expansion",
    "buyback", "tops", "boost", "boosts", "optimistic", "steigt", "steigen", "gewinnt",
    "rekord", "stark", "wächst", "wachstum", "gewinn", "hochgestuft", "erholung",
}
NEG = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls", "slump",
    "cut", "cuts", "downgrade", "downgraded", "warning", "warns", "lawsuit", "probe",
    "investigation", "recall", "layoff", "layoffs", "loss", "losses", "weak", "bankruptcy",
    "fraud", "decline", "declines", "slash", "halt", "delay", "delays", "bearish", "sinks",
    "tumble", "tumbles", "plummet", "concern", "concerns", "fear", "fears", "fällt",
    "fallen", "sinkt", "verlust", "warnung", "klage", "ermittlung", "insolvenz", "betrug",
}


def _load_cache() -> dict[str, Any]:
    raw = load_json(NEWS_CACHE, expected_type=dict, default={})
    return raw.get("entries", raw) if isinstance(raw, dict) else {}


def _save_cache(entries: dict[str, Any]) -> None:
    atomic_write_json(
        NEWS_CACHE,
        {
            "_meta": schema_meta(
                "stock-radar-news-cache",
                max_age_hours=MAX_NEWS_AGE_HOURS,
            ),
            "entries": entries,
        },
        indent=1,
    )


def _entry_datetime(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            value = parsedate_to_datetime(raw)
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _safe_link(link: str) -> str:
    parsed = urllib.parse.urlparse(link or "")
    return link if parsed.scheme in {"http", "https"} else ""


def _filter_entries(feed, *, limit: int, now: datetime, source: str = "") -> list[dict[str, Any]]:
    cutoff = now - timedelta(hours=MAX_NEWS_AGE_HOURS)
    items = []
    for entry in feed.entries:
        published = _entry_datetime(entry)
        if published is None or not (cutoff <= published <= now + timedelta(hours=1)):
            continue
        title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": _safe_link(entry.get("link", "")),
                "published_at": published.isoformat(timespec="seconds"),
                "source": source,
                "relevance_method": "provider_ticker_feed" if not source else "market_feed",
            }
        )
        if len(items) >= limit:
            break
    return items


def _download_feed(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "StockRadar/2.0"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read(2_000_000)
    return feedparser.parse(payload)


def _fetch_one(symbol: str, limit: int, now: datetime):
    try:
        feed = _download_feed(YF_RSS.format(sym=urllib.parse.quote(symbol)))
        return symbol, _filter_entries(feed, limit=limit, now=now), None
    except Exception as exc:
        return symbol, [], str(exc)[:300]


def fetch_all_ticker_news(
    symbols,
    limit=6,
    workers=12,
    *,
    now: datetime | None = None,
    return_status: bool = False,
):
    now = now or datetime.now(timezone.utc)
    cache = _load_cache()
    out: dict[str, list[dict[str, Any]]] = {}
    status: dict[str, dict[str, Any]] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, symbol, limit, now): symbol for symbol in symbols}
        for future in cf.as_completed(futures):
            symbol, items, error = future.result()
            previous = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
            if error:
                record = cache_failure(previous, error)
                cached_items = record.get("items") or []
                cutoff = now - timedelta(hours=MAX_NEWS_AGE_HOURS)
                items = [
                    item
                    for item in cached_items
                    if item.get("published_at")
                    and datetime.fromisoformat(item["published_at"]) >= cutoff
                ]
                record["status"] = "stale_good" if items else "unavailable"
            else:
                record = {
                    "items": items,
                    "fetched_at": utc_now(),
                    "last_success_at": utc_now(),
                    "status": "fresh",
                }
            cache[symbol] = record
            out[symbol] = items
            status[symbol] = {
                key: value for key, value in record.items() if key != "items"
            }
    _save_cache(cache)
    return (out, status) if return_status else out


def _score_titles(titles):
    positive = negative = 0
    for title in titles:
        words = set(re.findall(r"[\wäöüß]+", title.lower()))
        positive += len(words & POS)
        negative += len(words & NEG)
    score = max(5, min(95, 50 + (positive - negative) * 6))
    label = "positiv" if score >= 60 else "negativ" if score <= 40 else "neutral"
    return score, label, len(titles)


def news_signal(items):
    titles = [item["title"] for item in items if item.get("title")]
    score, label, count = _score_titles(titles)
    return {
        "news_score": score,
        "news_sentiment": label,
        "news_n": count,
        "news": items[:3],
        "news_model_status": "unvalidated_context_only",
    }


def fetch_market_news(limit=10, *, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    cache = _load_cache()
    previous = cache.get("__market__") if isinstance(cache.get("__market__"), dict) else {}
    headlines = []
    failures = {}
    for url in MARKET_FEEDS:
        try:
            feed = _download_feed(url)
            source = str(feed.feed.get("title", ""))
            headlines.extend(_filter_entries(feed, limit=limit, now=now, source=source))
        except Exception as exc:
            failures[url] = str(exc)[:300]
    if failures and not headlines:
        cutoff = now - timedelta(hours=MAX_NEWS_AGE_HOURS)
        for item in previous.get("items") or []:
            try:
                published = datetime.fromisoformat(item["published_at"])
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                if published >= cutoff:
                    headlines.append(item)
            except (KeyError, TypeError, ValueError):
                continue
        record = cache_failure(previous, "; ".join(f"{url}: {error}" for url, error in failures.items()))
        record["items"] = headlines
        record["status"] = "stale_good" if headlines else "unavailable"
    else:
        record = {
            "items": headlines[:24],
            "fetched_at": utc_now(),
            "last_success_at": utc_now(),
            "status": "fresh" if not failures else "fresh_partial",
            "failures": failures,
        }
    cache["__market__"] = record
    _save_cache(cache)
    score, label, _ = _score_titles([item["title"] for item in headlines])
    return {
        "market_sentiment": score,
        "market_label": label,
        "headlines": headlines[:24],
        "failures": failures,
        "model_status": "unvalidated_context_only",
    }
