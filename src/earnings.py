"""Next earnings (results) date per ticker via yfinance, with weekly caching."""
from datetime import datetime, timezone, timedelta, date

import yfinance as yf

from .config import DATA
from .persistence import atomic_write_json, cache_failure, load_json, schema_meta, utc_now

EARN_CACHE = DATA / "earnings.json"
EARN_MAX_AGE_DAYS = 7
FETCH_PAUSE = 0.15


def _load():
    return load_json(EARN_CACHE, expected_type=dict, default={})


def _save(cache):
    cache = dict(cache)
    cache["_meta"] = schema_meta("stock-radar-earnings-cache")
    atomic_write_json(EARN_CACHE, cache, indent=1)


def _earnings_dates(cal):
    try:
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return None, None
        today = date.today()
        valid = sorted(d.date() if isinstance(d, datetime) else d for d in dates if isinstance(d, date))
        upcoming = [d for d in valid if d >= today]
        previous = [d for d in valid if d < today]
        return (
            upcoming[0].isoformat() if upcoming else None,
            previous[-1].isoformat() if previous else None,
        )
    except Exception:  # noqa: BLE001
        return None, None


def _next_date(cal):
    """Compatibility helper: future-only by contract."""
    return _earnings_dates(cal)[0]


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
        print(f"Earnings dates: {len(symbols) - len(stale)} cached, refreshing {len(stale)} ...")

    for sym in stale:
        try:
            nd, previous = _earnings_dates(yf.Ticker(sym).calendar)
            cache[sym] = {
                "next_earnings": nd,
                "previous_earnings": previous,
                "fetched_at": utc_now(),
                "last_success_at": utc_now(),
            }
        except Exception as exc:  # noqa: BLE001
            cache[sym] = cache_failure(cache.get(sym), exc)
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
