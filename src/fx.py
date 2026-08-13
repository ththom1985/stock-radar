"""Fail-closed currency conversion with durable stale-good rates."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import yfinance as yf

from .config import DATA
from .persistence import atomic_write_json, load_json, schema_meta, utc_now

FX_CACHE = DATA / "fx_usd.json"
MAX_AGE_HOURS = 24

_SUFFIX_CCY = {
    ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".AS": "EUR", ".BR": "EUR", ".MI": "EUR",
    ".MC": "EUR", ".VI": "EUR", ".HE": "EUR", ".LS": "EUR", ".IR": "EUR", ".AT": "EUR",
    ".L": "GBp", ".SW": "CHF", ".VX": "CHF", ".ST": "SEK", ".OL": "NOK", ".CO": "DKK",
    ".WA": "PLN", ".HK": "HKD", ".T": "JPY", ".KS": "KRW", ".KQ": "KRW", ".TW": "TWD",
    ".TWO": "TWD", ".NS": "INR", ".BO": "INR", ".SA": "BRL", ".MX": "MXN", ".JK": "IDR",
    ".KL": "MYR", ".BK": "THB", ".SI": "SGD", ".SR": "SAR", ".JO": "ZAc", ".AX": "AUD",
    ".NZ": "NZD", ".TO": "CAD", ".V": "CAD", ".NE": "CAD", ".CN": "CAD",
}
_SUBUNIT = {"GBp": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}


class FXUnavailableError(RuntimeError):
    pass


@dataclass
class FXResult:
    rates: dict[str, float]
    status: dict[str, dict[str, Any]]
    missing: dict[str, str]


def _safe_log(message: str, enabled: bool) -> None:
    """Best-effort ASCII CLI logging that can never affect provider state."""
    if not enabled:
        return
    try:
        safe = str(message).encode("ascii", "backslashreplace").decode("ascii")
        print(safe)
    except Exception:
        # Console encoding/stream failures are observability failures, never
        # market-data failures.
        pass


def currency_for(symbol):
    symbol = (symbol or "").upper()
    if "." in symbol:
        return _SUFFIX_CCY.get(symbol[symbol.rindex(".") :], "UNKNOWN")
    return "USD"


def _load_entries() -> dict[str, dict[str, Any]]:
    raw = load_json(FX_CACHE, expected_type=dict, default={})
    if isinstance(raw.get("rates"), dict):
        return raw["rates"]
    # Legacy migration: numeric top-level rates shared one timestamp.
    timestamp = raw.get("_fetched_at")
    return {
        key: {
            "rate": value,
            "fetched_at": timestamp,
            "status": "legacy_stale",
        }
        for key, value in raw.items()
        if not key.startswith("_") and isinstance(value, (int, float)) and value > 0
    }


def _save_entries(entries: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(
        FX_CACHE,
        {
            "_meta": schema_meta("stock-radar-fx-cache", quote="USD per unit"),
            "rates": entries,
        },
        indent=1,
    )


def _pair_rate(currency: str) -> float:
    if currency == "USD":
        return 1.0
    errors = []
    for ticker, inverse in ((f"{currency}USD=X", False), (f"USD{currency}=X", True)):
        try:
            history = yf.Ticker(ticker).history(period="5d", timeout=20)
            if history is not None and not history.empty:
                price = float(history["Close"].dropna().iloc[-1])
                if price > 0:
                    return 1.0 / price if inverse else price
            errors.append(f"{ticker}: no data")
        except Exception as exc:
            errors.append(f"{ticker}: {str(exc)[:120]}")
        time.sleep(0.1)
    raise FXUnavailableError("; ".join(errors))


def _is_fresh(entry: dict[str, Any], now: datetime) -> bool:
    try:
        timestamp = datetime.fromisoformat(str(entry.get("fetched_at")))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp > now - timedelta(hours=MAX_AGE_HOURS)
    except (TypeError, ValueError):
        return False


def get_fx_rates_with_status(currencies, *, now: datetime | None = None, verbose=True) -> FXResult:
    now = now or datetime.now(timezone.utc)
    entries = _load_entries()
    # USD is definitional, never a market quote. Canonicalize legacy/stale cache
    # entries unconditionally so an old USD timestamp cannot block the pipeline.
    entries["USD"] = {
        "rate": 1.0,
        "fetched_at": utc_now(),
        "status": "fixed",
        "source": "definition",
    }
    requested = sorted({currency for currency in currencies if currency} | {"USD"})

    for currency in requested:
        entry = entries.get(currency)
        if currency == "USD":
            continue
        if entry and _is_fresh(entry, now):
            entry["status"] = "fresh"
            continue
        base, multiplier = _SUBUNIT.get(currency, (currency, 1.0))
        _safe_log(f"Refresh FX rate {currency}->USD ...", verbose)
        try:
            rate = _pair_rate(base) * multiplier
            entries[currency] = {
                "rate": rate,
                "fetched_at": utc_now(),
                "status": "fresh",
            }
        except Exception as exc:
            if entry and isinstance(entry.get("rate"), (int, float)) and entry["rate"] > 0:
                entry = dict(entry)
                entry["status"] = "stale"
                entry["last_failure"] = {"at": utc_now(), "error": str(exc)[:300]}
                entries[currency] = entry
            else:
                entries[currency] = {
                    "status": "missing",
                    "last_failure": {"at": utc_now(), "error": str(exc)[:300]},
                }

    _save_entries(entries)
    rates: dict[str, float] = {}
    status: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}
    for currency in requested:
        entry = entries.get(currency) or {}
        status[currency] = dict(entry)
        rate = entry.get("rate")
        if isinstance(rate, (int, float)) and rate > 0:
            rates[currency] = float(rate)
        else:
            missing[currency] = str((entry.get("last_failure") or {}).get("error") or "no valid rate")
    return FXResult(rates=rates, status=status, missing=missing)


def get_fx_rates(currencies, verbose=True):
    """Compatibility API that fails rather than inventing a 1:1 conversion."""
    result = get_fx_rates_with_status(currencies, verbose=verbose)
    if result.missing:
        raise FXUnavailableError(f"Missing FX rates: {result.missing}")
    return result.rates
