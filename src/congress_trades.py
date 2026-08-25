"""Official House PTR trades plus consent-gated Senate EFDS status."""
from __future__ import annotations

import io
import json
import math
import os
import re
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
HOUSE_INDEX_URL = (
    "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
)
HOUSE_PDF_URL = (
    "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
)
SENATE_HOME_URL = "https://efdsearch.senate.gov/search/home/"
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
    acknowledged = os.environ.get("SENATE_EFDS_AGREEMENT") == "1"
    return {
        "status": "pending_legal_acknowledgement" if not acknowledged else "enabled",
        "acknowledged": acknowledged,
        "required_setting": "SENATE_EFDS_AGREEMENT=1",
        "terms_url": SENATE_HOME_URL,
        "reason": (
            "The official Senate portal requires explicit acceptance of statutory "
            "use prohibitions before report access."
        ),
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
            "source": "U.S. House Clerk official PTR disclosures",
            "expected_delay": EXPECTED_DELAY,
            "limitations": (
                "Reported amount ranges are not exact values. House coverage is "
                "incrementally bounded; Senate is separate and consent-gated."
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
        return cache.get("signals") or {}, {
            "status": "cached",
            "house": cache.get("house_status"),
            "senate": senate_source_status(),
            "signal_count": len(cache.get("signals") or {}),
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
    trades = [
        trade
        for document in docs.values()
        for trade in document.get("trades") or []
        if trade.get("ticker") in watchlist_symbols
    ]
    signals = summarize_congress_trades(trades)
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
            "senate_status": senate_source_status(),
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
        "senate": senate_source_status(),
        "signal_count": len(signals),
    }
