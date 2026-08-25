"""Direct SEC Form 4 purchase/cluster signals with bounded refresh."""
from __future__ import annotations

import os
import time
import urllib.request
import xml.etree.ElementTree as ET
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
from .sec_companyfacts import (
    REQUEST_PAUSE_SECONDS,
    load_sec_ticker_map,
    normalize_sec_user_agent,
    request_sec_json,
)

CACHE_PATH = DATA / "sec_insider_signals.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MAX_AGE_DAYS = 2
LOOKBACK_DAYS = 90
CLUSTER_DAYS = 21
MAX_FILINGS_PER_SYMBOL = 8


def _fresh(entry):
    timestamp = (entry or {}).get("last_success_at")
    try:
        value = datetime.fromisoformat(str(timestamp))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return value >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)


def _request_text(url, user_agent, timeout=30):
    user_agent = normalize_sec_user_agent(user_agent)
    if not user_agent:
        raise ValueError("SEC_USER_AGENT is required by SEC fair-access policy")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/xml,text/xml,*/*",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _text(node, path):
    found = node.find(path)
    return (found.text or "").strip() if found is not None else ""


def parse_form4(xml_text):
    """Extract open-market non-derivative purchases and sales from one Form 4."""
    root = ET.fromstring(xml_text)
    owner = _text(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
    is_director = _text(root, ".//reportingOwner/reportingOwnerRelationship/isDirector") == "1"
    is_officer = _text(root, ".//reportingOwner/reportingOwnerRelationship/isOfficer") == "1"
    officer_title = _text(
        root, ".//reportingOwner/reportingOwnerRelationship/officerTitle"
    )
    transactions = []
    for transaction in root.findall(".//nonDerivativeTransaction"):
        code = _text(
            transaction,
            "./transactionCoding/transactionCode",
        )
        if code not in {"P", "S"}:
            continue
        acquired_disposed = _text(
            transaction,
            "./transactionAmounts/transactionAcquiredDisposedCode/value",
        )
        if (code == "P" and acquired_disposed != "A") or (
            code == "S" and acquired_disposed != "D"
        ):
            continue
        try:
            shares = float(
                _text(
                    transaction,
                    "./transactionAmounts/transactionShares/value",
                )
            )
        except ValueError:
            shares = None
        try:
            price = float(
                _text(
                    transaction,
                    "./transactionAmounts/transactionPricePerShare/value",
                )
            )
        except ValueError:
            price = None
        transactions.append(
            {
                "transaction_date": _text(
                    transaction, "./transactionDate/value"
                ),
                "code": code,
                "owner": owner or None,
                "is_director": is_director,
                "is_officer": is_officer,
                "officer_title": officer_title or None,
                "shares": shares,
                "price": price,
                "value": (
                    shares * price
                    if isinstance(shares, (int, float))
                    and isinstance(price, (int, float))
                    else None
                ),
            }
        )
    return transactions


def summarize_transactions(transactions, today=None):
    today = today or date.today()
    cutoff = today - timedelta(days=LOOKBACK_DAYS)
    recent = []
    for transaction in transactions:
        try:
            transaction_date = date.fromisoformat(transaction["transaction_date"])
        except (KeyError, TypeError, ValueError):
            continue
        if transaction_date >= cutoff:
            recent.append(transaction)
    purchases = [transaction for transaction in recent if transaction["code"] == "P"]
    sales = [transaction for transaction in recent if transaction["code"] == "S"]
    cluster_cutoff = today - timedelta(days=CLUSTER_DAYS)
    cluster_owners = {
        transaction.get("owner")
        for transaction in purchases
        if transaction.get("owner")
        and date.fromisoformat(transaction["transaction_date"]) >= cluster_cutoff
    }
    purchase_value = sum(
        transaction.get("value") or 0.0 for transaction in purchases
    )
    sale_value = sum(transaction.get("value") or 0.0 for transaction in sales)
    cluster = len(cluster_owners) >= 2
    score = 50.0
    if cluster:
        score += min(35.0, 15.0 + len(cluster_owners) * 5.0)
    elif purchases:
        score += 10.0
    if purchase_value > 0 and sale_value > 0:
        score += max(-15.0, min(15.0, (purchase_value - sale_value) / max(
            purchase_value, sale_value
        ) * 15.0))
    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "direction": (
            "positive" if score > 55 else "negative" if score < 45 else "neutral"
        ),
        "cluster_purchase": cluster,
        "cluster_owner_count": len(cluster_owners),
        "purchase_count_90d": len(purchases),
        "sale_count_90d": len(sales),
        "purchase_value_90d": round(purchase_value, 2),
        "sale_value_90d": round(sale_value, 2),
        "lookback_days": LOOKBACK_DAYS,
        "transactions": sorted(
            recent, key=lambda item: item.get("transaction_date") or "", reverse=True
        )[:24],
    }


def _recent_form4_filings(submissions, today=None):
    today = today or date.today()
    cutoff = today - timedelta(days=LOOKBACK_DAYS)
    recent = (submissions.get("filings") or {}).get("recent") or {}
    filings = []
    count = len(recent.get("form") or [])
    for index in range(count):
        if recent["form"][index] != "4":
            continue
        try:
            filed = date.fromisoformat(recent["filingDate"][index])
        except (IndexError, ValueError):
            continue
        if filed < cutoff:
            continue
        filings.append(
            {
                "filing_date": filed.isoformat(),
                "accession": recent["accessionNumber"][index],
                "document": recent["primaryDocument"][index],
            }
        )
    return filings[:MAX_FILINGS_PER_SYMBOL]


def fetch_insider_signals(symbols, max_new=None, force=False, verbose=True):
    user_agent = normalize_sec_user_agent()
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if not user_agent:
        return {
            symbol: cache.get(symbol, {})
            for symbol in symbols
            if isinstance(cache.get(symbol), dict)
        }, {
            "status": "disabled",
            "reason": "SEC_USER_AGENT is required by SEC fair-access policy",
            "refreshed": 0,
        }
    ticker_map = load_sec_ticker_map(user_agent)
    candidates = [
        symbol
        for symbol in symbols
        if symbol.upper() in ticker_map
        and (force or not _fresh(cache.get(symbol)))
    ]
    if max_new is None:
        max_new = int(os.environ.get("STOCK_RADAR_INSIDER_MAX", "15"))
    candidates = candidates[: max(0, max_new)]
    failures = {}
    for index, symbol in enumerate(candidates, 1):
        cik = ticker_map[symbol.upper()]
        try:
            submissions = request_sec_json(
                SUBMISSIONS_URL.format(cik=cik), user_agent
            )
            transactions = []
            for filing in _recent_form4_filings(submissions):
                url = ARCHIVES_URL.format(
                    cik=cik,
                    accession=filing["accession"].replace("-", ""),
                    document=filing["document"],
                )
                transactions.extend(parse_form4(_request_text(url, user_agent)))
                time.sleep(REQUEST_PAUSE_SECONDS)
            summary = summarize_transactions(transactions)
            summary.update(
                {
                    "source": "SEC EDGAR Form 4",
                    "source_url": (
                        f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=only"
                    ),
                    "fetched_at": utc_now(),
                    "last_success_at": utc_now(),
                    "expected_delay": "normally up to 2 business days",
                }
            )
            cache[symbol] = clear_cache_failure(summary)
        except Exception as exc:  # network and issuer payload errors vary
            cache[symbol] = cache_failure(cache.get(symbol), exc)
            failures[symbol] = str(exc)[:200]
        if verbose and index % 5 == 0:
            print(f"  SEC insiders {index}/{len(candidates)}")
        time.sleep(REQUEST_PAUSE_SECONDS)
    if candidates:
        cache["_meta"] = schema_meta(
            "stock-radar-sec-insider-cache",
            schema_version=1,
            refreshed=len(candidates),
        )
        atomic_write_json(CACHE_PATH, cache, indent=1)
    return {
        symbol: cache.get(symbol, {})
        for symbol in symbols
        if isinstance(cache.get(symbol), dict)
    }, {
        "status": "ok",
        "mapped": sum(symbol.upper() in ticker_map for symbol in symbols),
        "cached": sum(isinstance(cache.get(symbol), dict) for symbol in symbols),
        "refreshed": len(candidates),
        "failures": failures,
        "max_age_days": MAX_AGE_DAYS,
    }
