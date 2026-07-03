"""Currency of a ticker (from its Yahoo suffix) + USD conversion rates.

Prices come in the exchange's trading currency (EUR on Xetra, JPY in Tokyo,
GBp = pence in London, …). We convert every price level to USD for display.
Rates are fetched via yfinance FX pairs and cached daily.
"""
import json
import time
from datetime import datetime, timezone, timedelta

import yfinance as yf

from .config import DATA

FX_CACHE = DATA / "fx_usd.json"
MAX_AGE_HOURS = 20

# Yahoo suffix -> trading currency
_SUFFIX_CCY = {
    ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".AS": "EUR", ".BR": "EUR", ".MI": "EUR",
    ".MC": "EUR", ".VI": "EUR", ".HE": "EUR", ".LS": "EUR", ".IR": "EUR", ".AT": "EUR",
    ".L": "GBp", ".SW": "CHF", ".VX": "CHF", ".ST": "SEK", ".OL": "NOK", ".CO": "DKK",
    ".WA": "PLN", ".HK": "HKD", ".T": "JPY", ".KS": "KRW", ".KQ": "KRW", ".TW": "TWD",
    ".TWO": "TWD", ".NS": "INR", ".BO": "INR", ".SA": "BRL", ".MX": "MXN", ".JK": "IDR",
    ".KL": "MYR", ".BK": "THB", ".SI": "SGD", ".SR": "SAR", ".JO": "ZAc", ".AX": "AUD",
    ".NZ": "NZD", ".TO": "CAD", ".V": "CAD", ".NE": "CAD", ".CN": "CAD",
}

# minor unit -> (major currency, factor)
_SUBUNIT = {"GBp": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}


def currency_for(symbol):
    """Trading currency of a ticker. US-listed (incl. ADRs) trade in USD."""
    s = (symbol or "").upper()
    if "." in s:
        return _SUFFIX_CCY.get(s[s.rindex("."):], "USD")
    return "USD"


def _load():
    if FX_CACHE.exists():
        try:
            return json.loads(FX_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _pair_rate(ccy):
    """USD per 1 unit of ccy via yfinance, with inverse fallback."""
    if ccy == "USD":
        return 1.0
    for tkr, inv in ((f"{ccy}USD=X", False), (f"USD{ccy}=X", True)):
        try:
            h = yf.Ticker(tkr).history(period="5d")
            if h is not None and not h.empty:
                px = float(h["Close"].iloc[-1])
                if px > 0:
                    return (1.0 / px) if inv else px
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.1)
    return None


def get_fx_rates(currencies, verbose=True):
    """Return {currency: usd_per_unit}. Cached daily; USD always 1.0."""
    cache = _load()
    ts = cache.get("_fetched_at")
    fresh = False
    if ts:
        try:
            fresh = datetime.fromisoformat(ts) > datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
        except Exception:  # noqa: BLE001
            fresh = False

    rates = {k: v for k, v in cache.items() if not k.startswith("_")}
    need = [c for c in currencies if c and (c not in rates or not fresh)]
    if need:
        if verbose:
            print(f"Wechselkurse nach USD laden: {len(need)} Waehrungen ...")
        for ccy in need:
            base, mult = _SUBUNIT.get(ccy, (ccy, 1.0))
            r = _pair_rate(base)
            if r:
                rates[ccy] = round(r * mult, 8)
        rates["USD"] = 1.0
        cache = dict(rates)
        cache["_fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            FX_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    rates.setdefault("USD", 1.0)
    return rates
