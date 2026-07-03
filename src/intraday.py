"""Intraday time-of-day pattern per stock: how it typically behaves right after
the open, into the close, and overnight — to hint at a better entry time.

Uses ~1 month of 30-minute bars (free via yfinance). Heavy to fetch, so it is
bounded per run and cached for weeks (only recomputed occasionally). Patterns
are HISTORICAL tendencies, noisy and not guarantees — labelled as such.
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta

import yfinance as yf

from .config import DATA

PAT_CACHE = DATA / "intraday_patterns.json"
MAX_AGE_DAYS = 14
MAX_NEW_PER_RUN = int(os.environ.get("STOCK_RADAR_INTRADAY_PAT_MAX", "40"))
FETCH_PAUSE = 0.3
MIN_DAYS = 8


def _load():
    if PAT_CACHE.exists():
        try:
            return json.loads(PAT_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save(cache):
    PAT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _analyse(df):
    """From 30-min bars compute avg first-hour / last-hour / overnight moves."""
    if df is None or getattr(df, "empty", True) or len(df) < 4 * MIN_DAYS:
        return None
    opens, closes, gaps = [], [], []
    prev_close = None
    by_day = {}
    for ts, row in df.iterrows():
        by_day.setdefault(ts.date(), []).append((float(row["Open"]), float(row["Close"])))
    for _day, bars in by_day.items():
        if len(bars) < 4:
            prev_close = bars[-1][1]
            continue
        d_open = bars[0][0]
        first_hour = bars[1][1]                 # close of the 2nd 30-min bar (~1h in)
        d_close = bars[-1][1]
        pre_close = bars[-3][1]                 # ~1h before the close
        if d_open:
            opens.append((first_hour / d_open - 1) * 100)
        if pre_close:
            closes.append((d_close / pre_close - 1) * 100)
        if prev_close:
            gaps.append((d_open / prev_close - 1) * 100)
        prev_close = d_close
    if len(opens) < MIN_DAYS:
        return None

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None
    return {"open_ret_pct": _avg(opens), "close_ret_pct": _avg(closes),
            "gap_pct": _avg(gaps), "n_days": len(opens)}


def fetch_intraday_patterns(symbols, max_new=MAX_NEW_PER_RUN, verbose=True):
    """Return {symbol: {open_ret_pct, close_ret_pct, gap_pct, n_days}}. Bounded & cached."""
    cache = _load()
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
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

    todo = stale[:max_new]
    if verbose:
        print(f"Tageszeit-Muster (Intraday): {len(symbols) - len(stale)} aus Cache, "
              f"{len(todo)} neu (von {len(stale)} offen) …")
    for sym in todo:
        rec = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            df = yf.Ticker(sym).history(period="1mo", interval="30m")
            rec.update(_analyse(df) or {})
        except Exception:  # noqa: BLE001
            pass
        cache[sym] = rec
        time.sleep(FETCH_PAUSE)
    if todo:
        _save(cache)
    return {s: cache.get(s, {}) for s in symbols}


def pattern_note(rec):
    """A plain-German entry-timing hint from the measured tendencies, or None."""
    if not rec or rec.get("n_days", 0) < MIN_DAYS:
        return None
    o = rec.get("open_ret_pct")
    c = rec.get("close_ret_pct")
    g = rec.get("gap_pct")
    bits = []
    if isinstance(o, (int, float)):
        if o <= -0.25:
            bits.append(f"neigt nach Handelsstart zum Abverkauf (Ø {o:.1f}% 1. Std.) – oft besser 1–2 h nach Open")
        elif o >= 0.25:
            bits.append(f"läuft nach Handelsstart meist an (Ø +{o:.1f}% 1. Std.)")
    if isinstance(c, (int, float)):
        if c >= 0.25:
            bits.append(f"zieht oft zum Schluss an (Ø +{c:.1f}%)")
        elif c <= -0.25:
            bits.append(f"schwächelt oft zum Schluss (Ø {c:.1f}%)")
    if isinstance(g, (int, float)):
        if g >= 0.4:
            bits.append(f"öffnet meist mit Gap up (Ø +{g:.1f}%) – vorbörslich oft schon teuer")
        elif g <= -0.4:
            bits.append(f"öffnet oft schwächer (Gap Ø {g:.1f}%)")
    if not bits:
        return None
    return " · ".join(bits[:2]) + f" (Muster aus {rec['n_days']} Tagen, keine Garantie)"
