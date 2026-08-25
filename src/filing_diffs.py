"""SEC 10-K/10-Q Risk Factors change signals."""
from __future__ import annotations

import difflib
import html
import os
import re
import time
from datetime import datetime, timedelta, timezone

from .config import DATA
from .persistence import atomic_write_json, cache_failure, clear_cache_failure, load_json, schema_meta, utc_now
from .sec_companyfacts import load_sec_ticker_map, normalize_sec_user_agent, request_sec_json
from .sec_insiders import _request_text

CACHE_PATH = DATA / "filing_diffs.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MAX_AGE_DAYS = 14
RISK_TERMS = ("material advers", "substantial doubt", "uncertain", "headwind", "litigation", "regulatory", "cybersecurity", "liquidity", "impairment", "default")


def _filings(submissions):
    recent = (submissions.get("filings") or {}).get("recent") or {}
    values = []
    for index, form in enumerate(recent.get("form") or []):
        if form not in {"10-K", "10-Q"}:
            continue
        values.append({"form": form, "filing_date": recent["filingDate"][index], "report_date": recent["reportDate"][index], "accession": recent["accessionNumber"][index], "document": recent["primaryDocument"][index].split("/")[-1]})
        if len(values) == 2:
            break
    return values


def _plain_text(document):
    document = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", document)
    return " ".join(html.unescape(re.sub(r"(?s)<[^>]+>", " ", document)).split())


def extract_risk_factors(document):
    text = _plain_text(document)
    starts = list(re.finditer(r"(?i)\bitem\s+1a[\.\s:;-]+risk\s+factors\b", text))
    candidates = []
    for start in starts:
        tail = text[start.end():]
        end = re.search(r"(?i)\bitem\s+(?:1b|2)[\.\s:;-]+", tail)
        section = tail[: end.start() if end else min(len(tail), 180000)]
        if len(section) > 500:
            candidates.append(section)
    return max(candidates, key=len) if candidates else None


def compare_risk_sections(current, previous):
    if not current or not previous:
        return None
    split = lambda value: [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if len(part.strip()) >= 70]
    current_parts, previous_parts = split(current), split(previous)
    new_parts = []
    for paragraph in current_parts:
        similarity = max((difflib.SequenceMatcher(None, paragraph, old).ratio() for old in previous_parts), default=0)
        if similarity < 0.62:
            new_parts.append(paragraph)
    intensified = [part for part in new_parts if any(term in part.casefold() for term in RISK_TERMS)]
    ratio = len(new_parts) / max(1, len(current_parts))
    score = round(max(0.0, min(100.0, 50.0 - ratio * 80.0 - min(20, len(intensified) * 3))), 1)
    return {"score": score, "direction": "negative" if score < 45 else "neutral", "new_paragraph_count": len(new_parts), "intensified_count": len(intensified), "new_risk_excerpts": new_parts[:5], "intensified_excerpts": intensified[:5], "current_paragraph_count": len(current_parts)}


def _fresh(entry):
    try:
        parsed = datetime.fromisoformat(str((entry or {}).get("last_success_at")))
        if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)


def fetch_filing_diff_signals(rows, max_new=None, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    ua = normalize_sec_user_agent()
    if not ua: return {}, {"status": "disabled", "reason": "SEC_USER_AGENT required"}
    ticker_map = load_sec_ticker_map(ua)
    eligible = [row for row in rows if row.get("symbol","").upper() in ticker_map]
    eligible.sort(key=lambda row: (-(row.get("longterm_score") or 0), row["symbol"]))
    max_new = max_new if max_new is not None else int(os.environ.get("STOCK_RADAR_FILING_DIFF_MAX", "5"))
    candidates = [row for row in eligible if force or not _fresh(cache.get(row["symbol"]))][:max_new]
    failures = {}
    for index, row in enumerate(candidates, 1):
        symbol, cik = row["symbol"], ticker_map[row["symbol"].upper()]
        try:
            filings = _filings(request_sec_json(SUBMISSIONS_URL.format(cik=cik), ua))
            if len(filings) < 2: raise ValueError("fewer than two recent 10-K/10-Q filings")
            sections, filing_urls = [], []
            for filing in filings:
                url = ARCHIVE_URL.format(cik=cik, accession=filing["accession"].replace("-",""), document=filing["document"])
                filing_urls.append(url)
                sections.append(extract_risk_factors(_request_text(url, ua)))
                time.sleep(0.15)
            summary = compare_risk_sections(*sections)
            if summary is None: raise ValueError("Risk Factors section unavailable")
            summary.update({"current_form": filings[0]["form"], "current_filing_date": filings[0]["filing_date"], "previous_form": filings[1]["form"], "previous_filing_date": filings[1]["filing_date"], "source": "SEC EDGAR 10-K/10-Q Risk Factors", "source_url": filing_urls[0], "previous_source_url": filing_urls[1], "expected_delay": "filing publication time", "fetched_at": utc_now(), "last_success_at": utc_now()})
            cache[symbol] = clear_cache_failure(summary)
        except Exception as exc:
            cache[symbol] = cache_failure(cache.get(symbol), exc); failures[symbol] = str(exc)[:200]
        if verbose: print(f"  Filing risk diffs {index}/{len(candidates)}")
    if candidates:
        cache["_meta"] = schema_meta("stock-radar-filing-diffs-cache", schema_version=1, refreshed=len(candidates))
        atomic_write_json(CACHE_PATH, cache, indent=1)
    return {row["symbol"]:cache.get(row["symbol"]) for row in rows if isinstance(cache.get(row.get("symbol")),dict) and cache.get(row["symbol"],{}).get("score") is not None}, {"status":"ok" if not failures else "partial","eligible":len(eligible),"cached":sum(_fresh(cache.get(row["symbol"])) for row in eligible),"refreshed":len(candidates),"signal_count":sum((cache.get(row["symbol"]) or {}).get("score") is not None for row in eligible),"failures":failures}
