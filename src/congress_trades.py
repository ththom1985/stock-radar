"""Official House PTR trades plus consent-gated Senate EFDS status."""
from __future__ import annotations

import io
import html
import json
import math
import os
import re
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone

from pypdf import PdfReader

from .config import DATA
from .persistence import (
    atomic_write_json,
    cache_failure,
    clear_cache_failure,
    load_json,
    schema_meta,
    utc_now,
)

CACHE_PATH = DATA / "congress_trades.json"
SENATE_PRIVATE_CACHE_PATH = DATA / "senate_efds_private.json"
SOURCE_POLICIES_PATH = DATA / "source_policies.json"
HOUSE_INDEX_URL = (
    "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
)
HOUSE_PDF_URL = (
    "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
)
SENATE_HOME_URL = "https://efdsearch.senate.gov/search/home/"
SENATE_REPORTS_URL = "https://efdsearch.senate.gov/search/report/data/"
MAX_AGE_HOURS = 24
LOOKBACK_DAYS = 180
EXPECTED_DELAY = "STOCK Act filing can lag transaction date by up to 45 days"
_AMOUNT_RE = re.compile(r"\$([\d,]+)\s*-\s*\$([\d,]+)")
_TRANSACTION_RE = re.compile(
    r"(?P<asset>[A-Za-z0-9&.,'’/ \-\n]{2,160}?)\s*"
    r"\((?P<ticker>[A-Z][A-Z0-9.\-]{0,9})\)\s*"
    r"\[(?P<asset_type>[A-Z]{2})\]\s*"
    r"(?P<transaction>P|S|E)(?:\s*\([^)]+\))?\s*"
    r"(?P<transaction_date>\d{2}/\d{2}/\d{4})\s*"
    r"(?P<notification_date>\d{2}/\d{2}/\d{4})\s*"
    r"(?P<amount>\$[\d,]+\s*-\s*\$[\d,]+)",
    re.MULTILINE,
)


def _request_bytes(url, timeout=60):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Stock-Radar public congressional-disclosure research",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_house_index(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        xml_name = next(
            name for name in archive.namelist() if name.lower().endswith(".xml")
        )
        xml_text = archive.read(xml_name).decode("utf-8-sig")
    members = []
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    for node in root.findall(".//Member"):
        item = {child.tag: (child.text or "").strip() for child in node}
        if item.get("FilingType") != "P" or not item.get("DocID"):
            continue
        members.append(
            {
                "first": item.get("First"),
                "last": item.get("Last"),
                "state_district": item.get("StateDst"),
                "filing_date": _iso_date(item.get("FilingDate")),
                "year": int(item.get("Year")),
                "doc_id": item.get("DocID"),
            }
        )
    members.sort(
        key=lambda item: (item.get("filing_date") or "", item["doc_id"]),
        reverse=True,
    )
    return members


def _iso_date(value):
    try:
        return datetime.strptime(str(value), "%m/%d/%Y").date().isoformat()
    except ValueError:
        return None


def _clean_pdf_text(value):
    value = value.replace("\x00", "").replace("\ufffd", " ")
    return re.sub(r"[ \t]+", " ", value)


def parse_house_ptr_pdf(pdf_bytes, filing):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = _clean_pdf_text(
        "\n".join((page.extract_text() or "") for page in reader.pages)
    )
    return parse_house_ptr_text(text, filing)


def parse_house_ptr_text(text, filing):
    trades = []
    seen = set()
    for match in _TRANSACTION_RE.finditer(text):
        amount_match = _AMOUNT_RE.search(match.group("amount"))
        if not amount_match:
            continue
        ticker = match.group("ticker").replace(".", "-")
        transaction_date = _iso_date(match.group("transaction_date"))
        key = (ticker, match.group("transaction"), transaction_date, match.group("amount"))
        if key in seen:
            continue
        seen.add(key)
        trades.append(
            {
                "chamber": "House",
                "member": " ".join(
                    filter(None, [filing.get("first"), filing.get("last")])
                ),
                "ticker": ticker,
                "asset": " ".join(match.group("asset").split())[-160:],
                "transaction_type": match.group("transaction"),
                "transaction_date": transaction_date,
                "notification_date": _iso_date(match.group("notification_date")),
                "filing_date": filing.get("filing_date"),
                "amount_range": match.group("amount"),
                "amount_low": float(amount_match.group(1).replace(",", "")),
                "amount_high": float(amount_match.group(2).replace(",", "")),
                "doc_id": filing.get("doc_id"),
                "source_url": HOUSE_PDF_URL.format(
                    year=filing["year"], doc_id=filing["doc_id"]
                ),
            }
        )
    return trades


def senate_source_status():
    policy = load_json(
        SOURCE_POLICIES_PATH, required=True, expected_type=dict
    ).get("senate_efds") or {}
    acknowledged = bool(policy.get("statutory_use_acknowledged"))
    local_only = not bool(policy.get("public_detail"))
    cloud_blocked = os.environ.get("GITHUB_ACTIONS") == "true" and local_only
    return {
        "status": (
            "pending_legal_acknowledgement"
            if not acknowledged
            else "local_only_not_run"
            if cloud_blocked
            else "enabled_local_only"
            if local_only
            else "enabled"
        ),
        "acknowledged": acknowledged,
        "local_only": local_only,
        "public_detail": bool(policy.get("public_detail")),
        "public_aggregate_score": bool(policy.get("public_aggregate_score")),
        "terms_url": SENATE_HOME_URL,
        "reason": (
            "The official Senate portal requires explicit acceptance of statutory "
            "use prohibitions before report access."
        ),
    }


def _csrf(text):
    match = re.search(
        r'name=["\']csrfmiddlewaretoken["\'][^>]*value=["\']([^"\']+)',
        text,
    )
    if not match:
        raise ValueError("Senate EFDS page did not contain a CSRF token")
    return match.group(1)


def _strip_html(value):
    value = re.sub(r"<[^>]+>", " ", str(value))
    return " ".join(html.unescape(value).split())


def parse_senate_report_html(report_html, report):
    tbody_match = re.search(r"<tbody[^>]*>(.*?)</tbody>", report_html, re.S | re.I)
    if not tbody_match:
        return []
    trades = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody_match.group(1), re.S | re.I):
        columns = [
            _strip_html(value)
            for value in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S | re.I)
        ]
        if len(columns) < 8:
            continue
        ticker = columns[3].replace(".", "-")
        if not re.fullmatch(r"[A-Z][A-Z0-9\-]{0,9}", ticker):
            continue
        order = columns[6].casefold()
        transaction_type = (
            "P" if order.startswith("purchase")
            else "S" if order.startswith("sale")
            else "E" if order.startswith("exchange")
            else None
        )
        amount_match = _AMOUNT_RE.search(columns[7])
        transaction_date = _iso_date(columns[1])
        if not transaction_type or not amount_match or not transaction_date:
            continue
        trades.append(
            {
                "chamber": "Senate",
                "member": report["member"],
                "ticker": ticker,
                "asset": columns[4],
                "transaction_type": transaction_type,
                "transaction_date": transaction_date,
                "notification_date": None,
                "filing_date": report.get("filing_date"),
                "amount_range": columns[7],
                "amount_low": float(amount_match.group(1).replace(",", "")),
                "amount_high": float(amount_match.group(2).replace(",", "")),
                "doc_id": report["report_path"].rstrip("/").split("/")[-1],
                "source_url": f"https://efdsearch.senate.gov{report['report_path']}",
            }
        )
    return trades


def fetch_senate_trades(force=False, verbose=True):
    status = senate_source_status()
    if status["status"] not in {"enabled_local_only", "enabled"}:
        return [], status
    private_cache = load_json(
        SENATE_PRIVATE_CACHE_PATH, expected_type=dict, default={}
    )
    if not force and _fresh(private_cache):
        trades = [
            trade
            for document in (private_cache.get("documents") or {}).values()
            for trade in document.get("trades") or []
        ]
        return trades, {
            **status,
            "status": "cached_local_only",
            "cached_document_count": len(private_cache.get("documents") or {}),
            "trade_count": len(trades),
        }
    from curl_cffi import requests

    session = requests.Session(impersonate="chrome")
    home = session.get(SENATE_HOME_URL, timeout=45)
    home.raise_for_status()
    agreement = session.post(
        SENATE_HOME_URL,
        data={
            "csrfmiddlewaretoken": _csrf(home.text),
            "prohibition_agreement": "1",
        },
        headers={"Referer": SENATE_HOME_URL},
        timeout=45,
    )
    agreement.raise_for_status()
    search_csrf = _csrf(agreement.text)
    start_date = (date.today() - timedelta(days=LOOKBACK_DAYS + 60)).strftime(
        "%m/%d/%Y 00:00:00"
    )
    response = session.post(
        SENATE_REPORTS_URL,
        data={
            "start": "0",
            "length": "100",
            "report_types": "[11]",
            "filer_types": "[]",
            "submitted_start_date": start_date,
            "submitted_end_date": "",
            "candidate_state": "",
            "senator_state": "",
            "office_id": "",
            "first_name": "",
            "last_name": "",
            "csrfmiddlewaretoken": search_csrf,
        },
        headers={
            "Referer": "https://efdsearch.senate.gov/search/",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=45,
    )
    response.raise_for_status()
    reports = []
    for item in response.json().get("data") or []:
        link_match = re.search(r'href="([^"]+)"', item[3])
        if not link_match or link_match.group(1).startswith("/search/view/paper/"):
            continue
        reports.append(
            {
                "member": " ".join(filter(None, [item[0], item[1]])),
                "report_path": link_match.group(1),
                "filing_date": _iso_date(item[4]),
            }
        )
    documents = dict(private_cache.get("documents") or {})
    max_new = int(os.environ.get("STOCK_RADAR_SENATE_MAX", "20"))
    unseen = [
        report
        for report in reports
        if report["report_path"] not in documents
    ][:max_new]
    failures = {}
    for index, report in enumerate(unseen, 1):
        try:
            page = session.get(
                f"https://efdsearch.senate.gov{report['report_path']}",
                headers={"Referer": "https://efdsearch.senate.gov/search/"},
                timeout=45,
            )
            page.raise_for_status()
            documents[report["report_path"]] = {
                "report": report,
                "trades": parse_senate_report_html(page.text, report),
                "fetched_at": utc_now(),
            }
        except Exception as exc:
            failures[report["report_path"]] = str(exc)[:200]
        if verbose:
            print(f"  Senate PTR reports {index}/{len(unseen)}")
        time.sleep(2.0)
    now = utc_now()
    private_payload = clear_cache_failure(
        {
            "documents": documents,
            "last_success_at": now,
            "_meta": {
                **schema_meta(
                    "stock-radar-senate-efds-private-cache",
                    schema_version=1,
                    policy="local_only",
                ),
                "last_success_at": now,
            },
        }
    )
    atomic_write_json(SENATE_PRIVATE_CACHE_PATH, private_payload, indent=1)
    trades = [
        trade
        for document in documents.values()
        for trade in document.get("trades") or []
    ]
    return trades, {
        **status,
        "status": "ok_local_only" if not failures else "partial_local_only",
        "report_count": len(reports),
        "cached_document_count": len(documents),
        "refreshed_document_count": len(unseen),
        "trade_count": len(trades),
        "failures": failures,
    }


def summarize_congress_trades(trades, today=None):
    today = today or date.today()
    cutoff = today - timedelta(days=LOOKBACK_DAYS)
    grouped = {}
    for trade in trades:
        try:
            trade_date = date.fromisoformat(trade["transaction_date"])
        except (KeyError, TypeError, ValueError):
            continue
        if trade_date < cutoff:
            continue
        grouped.setdefault(trade["ticker"], []).append(trade)
    signals = {}
    for ticker, items in grouped.items():
        directional = []
        for item in items:
            direction = 1 if item["transaction_type"] == "P" else -1
            size_weight = min(2.0, max(1.0, math.log10(item["amount_low"]) - 2.0))
            directional.append(direction * size_weight)
        average = sum(directional) / max(1, len(directional))
        score = round(max(0.0, min(100.0, 50.0 + average * 12.0)), 1)
        most_recent = max(item["transaction_date"] for item in items)
        chambers = sorted({item.get("chamber") for item in items if item.get("chamber")})
        signals[ticker] = {
            "score": score,
            "direction": "positive" if score > 55 else "negative" if score < 45 else "neutral",
            "latest_transaction_date": most_recent,
            "trade_count": len(items),
            "purchase_count": sum(item["transaction_type"] == "P" for item in items),
            "sale_count": sum(item["transaction_type"] == "S" for item in items),
            "trades": sorted(
                items, key=lambda item: item["transaction_date"], reverse=True
            )[:20],
            "source": (
                "Official U.S. congressional PTR disclosures: "
                + " + ".join(chambers)
            ),
            "expected_delay": EXPECTED_DELAY,
            "limitations": (
                "Reported amount ranges are not exact values. House and Senate "
                "coverage are incrementally bounded; Senate details remain local-only."
            ),
        }
    return signals


def _fresh(cache):
    timestamp = (cache.get("_meta") or {}).get("last_success_at")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)


def fetch_congress_signals(rows, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if not force and _fresh(cache):
        house_trades = [
            trade
            for document in (cache.get("house_documents") or {}).values()
            for trade in document.get("trades") or []
        ]
        senate_trades, senate_status = fetch_senate_trades(
            force=False, verbose=verbose
        )
        watchlist_symbols = {str(row.get("symbol") or "") for row in rows}
        signals = summarize_congress_trades(
            [
                trade
                for trade in (*house_trades, *senate_trades)
                if trade.get("ticker") in watchlist_symbols
            ]
        )
        return signals, {
            "status": "cached",
            "house": cache.get("house_status"),
            "senate": senate_status,
            "signal_count": len(signals),
        }
    current_year = datetime.now(timezone.utc).year
    docs = dict(cache.get("house_documents") or {})
    failures = {}
    try:
        filings = parse_house_index(
            _request_bytes(HOUSE_INDEX_URL.format(year=current_year))
        )
    except Exception as exc:
        failed = cache_failure(cache, exc)
        return cache.get("signals") or {}, {
            "status": "error",
            "house": {"status": "error", "reason": str(exc)[:200]},
            "senate": senate_source_status(),
        }
    max_new = int(os.environ.get("STOCK_RADAR_HOUSE_MAX", "20"))
    unseen = [filing for filing in filings if filing["doc_id"] not in docs][:max_new]
    for index, filing in enumerate(unseen, 1):
        try:
            pdf = _request_bytes(
                HOUSE_PDF_URL.format(
                    year=filing["year"], doc_id=filing["doc_id"]
                )
            )
            docs[filing["doc_id"]] = {
                "filing": filing,
                "trades": parse_house_ptr_pdf(pdf, filing),
                "fetched_at": utc_now(),
            }
        except Exception as exc:
            failures[filing["doc_id"]] = str(exc)[:200]
        if verbose:
            print(f"  House PTR documents {index}/{len(unseen)}")
    watchlist_symbols = {str(row.get("symbol") or "") for row in rows}
    house_trades = [
        trade
        for document in docs.values()
        for trade in document.get("trades") or []
        if trade.get("ticker") in watchlist_symbols
    ]
    senate_trades, senate_status = fetch_senate_trades(force=force, verbose=verbose)
    signals = summarize_congress_trades([*house_trades, *senate_trades])
    now = utc_now()
    for signal in signals.values():
        signal["fetched_at"] = now
        signal["last_success_at"] = now
    house_status = {
        "status": "ok" if not failures else "partial",
        "index_count": len(filings),
        "cached_document_count": len(docs),
        "refreshed_document_count": len(unseen),
        "parsed_trade_count": sum(
            len(document.get("trades") or []) for document in docs.values()
        ),
        "failures": failures,
        "index_url": HOUSE_INDEX_URL.format(year=current_year),
    }
    payload = clear_cache_failure(
        {
            "signals": signals,
            "house_documents": docs,
            "house_status": house_status,
            "senate_status": senate_status,
            "fetched_at": now,
            "last_success_at": now,
            "_meta": {
                **schema_meta(
                    "stock-radar-congress-trades-cache",
                    schema_version=1,
                    expected_delay=EXPECTED_DELAY,
                ),
                "last_success_at": now,
            },
        }
    )
    atomic_write_json(CACHE_PATH, payload, indent=1)
    return signals, {
        "status": "ok" if not failures else "partial",
        "house": house_status,
        "senate": senate_status,
        "signal_count": len(signals),
    }
