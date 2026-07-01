"""Next earnings (results) date per ticker via yfinance, with weekly caching."""
import json
from datetime import datetime, timezone, timedelta, date

import yfinance as yf

from .config import DATA

EARN_CACHE = DATA / "earnings.json"
EARN_MAX_AGE_DAYS = 7
FETCH_PAUSE = 0.15


def _load():
    if EARN_CACHE.exists():
        try:
            return json.loads(EARN_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save(cache):
    EARN_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _next_date(cal):
    try:
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return None
        today = date.today()
        future = sorted(d for d in dates if isinstance(d, date))
        upcoming = [d for d in future if d >= today]
        pick = upcoming[0] if upcoming else (future[-1] if future else None)
        return pick.isoformat() if pick else None
    except Exception:  # noqa: BLE001
        return None


def fetch_earnings(symbols, max_age_days=EARN_MAX_AGE_DAYS, verbose=True):
    """Return {symbol: {next_earnings: 'YYYY-MM-DD'|None}}. Caches per symbol."""
    import time
    cache = _load()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    stale = []
    for s in symbols:
        e = cache.get(s)
        if not e:
            stale.append(s)
            continue
        try:
            if datetime.fromisoformat(e.get("fetched_at", "1970-01-01T00:00:00+00:00")) < cutoff:
                stale.append(s)
        except Exception:  # noqa: BLE001
            stale.append(s)

    if verbose:
        print(f"Earnings-Termine: {len(symbols) - len(stale)} aus Cache, {len(stale)} neu …")

    for sym in stale:
        try:
            nd = _next_date(yf.Ticker(sym).calendar)
        except Exception:  # noqa: BLE001
            nd = None
        cache[sym] = {"next_earnings": nd,
                      "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        time.sleep(FETCH_PAUSE)

    if stale:
        _save(cache)
    return {s: cache.get(s, {}) for s in symbols}


def days_until(iso_date):
    if not iso_date:
        return None
    try:
        d = date.fromisoformat(iso_date)
        return (d - date.today()).days
    except Exception:  # noqa: BLE001
        return None
