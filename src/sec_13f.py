"""Official SEC 13F quarter-over-quarter changes for configured large managers."""
from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
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
from .sec_companyfacts import (
    REQUEST_PAUSE_SECONDS,
    normalize_sec_user_agent,
    request_sec_json,
)
from .sec_insiders import _request_text

CACHE_PATH = DATA / "sec_13f_changes.json"
MANAGERS_PATH = DATA / "institutional_managers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MAX_AGE_DAYS = 7
EXPECTED_DELAY = "up to 45 days after calendar quarter end"


def load_managers():
    payload = load_json(MANAGERS_PATH, required=True, expected_type=dict)
    managers = payload.get("managers")
    if not isinstance(managers, list) or not managers:
        raise ValueError("institutional_managers.json must contain managers")
    return [
        {"name": str(item["name"]), "cik": int(item["cik"])}
        for item in managers
    ]


def _recent_filings(submissions):
    recent = (submissions.get("filings") or {}).get("recent") or {}
    filings = []
    for index, form in enumerate(recent.get("form") or []):
        if form not in {"13F-HR", "13F-HR/A"}:
            continue
        filings.append(
            {
                "form": form,
                "filing_date": recent["filingDate"][index],
                "report_date": recent["reportDate"][index],
                "accession": recent["accessionNumber"][index],
                "primary_document": recent["primaryDocument"][index],
            }
        )
    filings.sort(
        key=lambda item: (item["report_date"], item["filing_date"], item["accession"]),
        reverse=True,
    )
    selected = []
    seen_periods = set()
    for filing in filings:
        if filing["report_date"] in seen_periods:
            continue
        selected.append(filing)
        seen_periods.add(filing["report_date"])
        if len(selected) == 2:
            break
    return selected


def _information_table_document(index_payload, primary_document):
    items = ((index_payload.get("directory") or {}).get("item") or [])
    primary_name = str(primary_document).split("/")[-1].casefold()
    candidates = []
    for item in items:
        name = str(item.get("name") or "")
        lower = name.casefold()
        if not lower.endswith(".xml") or lower == primary_name:
            continue
        score = 0
        if "info" in lower:
            score += 4
        if "table" in lower:
            score += 3
        score += min(int(item.get("size") or 0), 10_000_000) / 10_000_000
        candidates.append((score, name))
    if not candidates:
        raise ValueError("13F filing index contains no information-table XML")
    return max(candidates)[1]


def _local_text(node, name):
    for child in node.iter():
        if child.tag.rsplit("}", 1)[-1] == name:
            return (child.text or "").strip()
    return ""


def parse_information_table(xml_text):
    root = ET.fromstring(xml_text)
    holdings = {}
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "infoTable":
            continue
        issuer = _local_text(node, "nameOfIssuer")
        cusip = _local_text(node, "cusip")
        put_call = _local_text(node, "putCall").upper() or None
        try:
            shares = float(_local_text(node, "sshPrnamt"))
        except ValueError:
            shares = None
        try:
            value = float(_local_text(node, "value"))
        except ValueError:
            value = None
        if not issuer or not cusip or shares is None:
            continue
        key = (cusip, put_call)
        current = holdings.setdefault(
            key,
            {
                "issuer": issuer,
                "cusip": cusip,
                "put_call": put_call,
                "shares": 0.0,
                "reported_value": 0.0,
            },
        )
        current["shares"] += shares
        if value is not None:
            current["reported_value"] += value
    return list(holdings.values())


def compare_holdings(current, previous):
    current_by_key = {
        (item["cusip"], item.get("put_call")): item for item in current
    }
    previous_by_key = {
        (item["cusip"], item.get("put_call")): item for item in previous
    }
    changes = []
    for key in sorted(
        set(current_by_key) | set(previous_by_key),
        key=lambda item: (str(item[0]), str(item[1] or "")),
    ):
        now = current_by_key.get(key)
        before = previous_by_key.get(key)
        now_shares = (now or {}).get("shares") or 0.0
        before_shares = (before or {}).get("shares") or 0.0
        if before_shares == 0 and now_shares > 0:
            action = "new"
        elif now_shares == 0 and before_shares > 0:
            action = "exited"
        elif now_shares > before_shares:
            action = "increased"
        elif now_shares < before_shares:
            action = "reduced"
        else:
            action = "unchanged"
        change_pct = (
            (now_shares / before_shares - 1.0) * 100.0
            if before_shares > 0
            else None
        )
        base = now or before
        changes.append(
            {
                "issuer": base["issuer"],
                "cusip": base["cusip"],
                "put_call": base.get("put_call"),
                "action": action,
                "shares": now_shares,
                "previous_shares": before_shares,
                "change_pct": (
                    round(change_pct, 2) if change_pct is not None else None
                ),
            }
        )
    return changes


def _normalize_name(value):
    value = re.sub(r"[^A-Z0-9 ]+", " ", str(value).upper())
    tokens = [
        token
        for token in value.split()
        if token
        not in {
            "INC",
            "CORP",
            "CORPORATION",
            "PLC",
            "LTD",
            "LIMITED",
            "CO",
            "COMPANY",
            "SA",
            "SE",
            "NV",
            "AG",
            "THE",
            "CLASS",
            "CL",
            "COM",
        }
    ]
    return " ".join(tokens)


def map_changes_to_rows(changes, rows):
    aliases = {}
    for row in rows:
        symbol = row.get("symbol")
        if not symbol:
            continue
        names = {
            _normalize_name(row.get("display_name_full")),
            _normalize_name(row.get("provider_long_name")),
            _normalize_name(row.get("name")),
        }
        for name in names:
            if len(name) >= 4:
                aliases.setdefault(name, set()).add(symbol)
    mapped = {}
    for change in changes:
        issuer = _normalize_name(change["issuer"])
        exact = aliases.get(issuer, set())
        candidates = set(exact)
        if not candidates and len(issuer) >= 6:
            for alias, symbols in aliases.items():
                if issuer in alias or alias in issuer:
                    candidates.update(symbols)
        if len(candidates) == 1:
            symbol = next(iter(candidates))
            mapped.setdefault(symbol, []).append(change)
    return mapped


def summarize_symbol_changes(symbol_changes, report_date):
    directional = []
    managers = []
    for item in symbol_changes:
        action = item["action"]
        put_call = item.get("put_call")
        base = {"new": 18, "increased": 9, "reduced": -8, "exited": -18}.get(
            action, 0
        )
        if put_call == "PUT":
            base *= -1
        elif put_call == "CALL":
            base *= 1
        directional.append(base)
        managers.append(
            {
                "manager": item["manager"],
                "action": action,
                "change_pct": item.get("change_pct"),
                "shares": item.get("shares"),
                "previous_shares": item.get("previous_shares"),
                "put_call": put_call,
                "filing_date": item.get("filing_date"),
            }
        )
    score = round(
        max(0.0, min(100.0, 50.0 + sum(directional) / max(1, len(directional)))),
        1,
    )
    return {
        "score": score,
        "direction": "positive" if score > 55 else "negative" if score < 45 else "neutral",
        "report_period": report_date,
        "manager_count": len({item["manager"] for item in managers}),
        "changes": managers,
        "source": "SEC EDGAR 13F-HR",
        "expected_delay": EXPECTED_DELAY,
        "limitations": (
            "Configured-manager subset only; 13F is delayed, long holdings only, "
            "and issuer-name mapping is conservative."
        ),
    }


def _fresh(cache):
    timestamp = (cache.get("_meta") or {}).get("last_success_at")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return parsed >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)


def fetch_13f_signals(rows, force=False, verbose=True):
    cache = load_json(CACHE_PATH, expected_type=dict, default={})
    if not force and _fresh(cache):
        return cache.get("signals") or {}, {
            "status": "cached",
            "manager_count": len(cache.get("managers") or []),
            "signal_count": len(cache.get("signals") or {}),
            "max_age_days": MAX_AGE_DAYS,
        }
    user_agent = normalize_sec_user_agent()
    if not user_agent:
        return cache.get("signals") or {}, {
            "status": "disabled",
            "reason": "SEC_USER_AGENT is required by SEC fair-access policy",
        }
    managers = load_managers()
    all_changes = []
    manager_status = []
    failures = {}
    latest_report_date = None
    for index, manager in enumerate(managers, 1):
        try:
            cik = manager["cik"]
            submissions = request_sec_json(SUBMISSIONS_URL.format(cik=cik), user_agent)
            filings = _recent_filings(submissions)
            if len(filings) < 2:
                raise ValueError("fewer than two recent 13F-HR report periods")
            parsed_periods = []
            for filing in filings:
                accession = filing["accession"].replace("-", "")
                index_payload = request_sec_json(
                    INDEX_URL.format(cik=cik, accession=accession), user_agent
                )
                document = _information_table_document(
                    index_payload, filing["primary_document"]
                )
                xml_text = _request_text(
                    ARCHIVE_URL.format(
                        cik=cik,
                        accession=accession,
                        document=document,
                    ),
                    user_agent,
                )
                parsed_periods.append(parse_information_table(xml_text))
                time.sleep(REQUEST_PAUSE_SECONDS)
            changes = compare_holdings(parsed_periods[0], parsed_periods[1])
            for change in changes:
                change.update(
                    {
                        "manager": manager["name"],
                        "filing_date": filings[0]["filing_date"],
                        "report_date": filings[0]["report_date"],
                    }
                )
            all_changes.extend(changes)
            latest_report_date = max(
                latest_report_date or filings[0]["report_date"],
                filings[0]["report_date"],
            )
            manager_status.append(
                {
                    **manager,
                    "status": "ok",
                    "current_report_date": filings[0]["report_date"],
                    "previous_report_date": filings[1]["report_date"],
                    "holding_count": len(parsed_periods[0]),
                }
            )
        except Exception as exc:
            failures[manager["name"]] = str(exc)[:200]
            manager_status.append({**manager, "status": "error"})
        if verbose:
            print(f"  SEC 13F managers {index}/{len(managers)}")
        time.sleep(REQUEST_PAUSE_SECONDS)
    current_changes = [
        change
        for change in all_changes
        if change.get("report_date") == latest_report_date
    ]
    stale_managers = [
        item["name"]
        for item in manager_status
        if item.get("status") == "ok"
        and item.get("current_report_date") != latest_report_date
    ]
    mapped = map_changes_to_rows(current_changes, rows)
    signals = {
        symbol: summarize_symbol_changes(changes, latest_report_date)
        for symbol, changes in mapped.items()
    }
    now = utc_now()
    for signal in signals.values():
        signal["fetched_at"] = now
        signal["last_success_at"] = now
    payload = clear_cache_failure(
        {
            "signals": signals,
            "managers": manager_status,
            "fetched_at": now,
            "last_success_at": now,
            "_meta": {
                **schema_meta(
                    "stock-radar-sec-13f-cache",
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
        "manager_count": len(managers),
        "manager_success_count": len(managers) - len(failures),
        "signal_count": len(signals),
        "report_period": latest_report_date,
        "stale_managers_excluded": stale_managers,
        "failures": failures,
        "max_age_days": MAX_AGE_DAYS,
    }
