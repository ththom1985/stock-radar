"""Statement-based fundamental scores from the literature:
  - Piotroski F-Score (0-9): fundamental strength/quality
  - Altman Z-Score: bankruptcy risk (safe / grey / distress)

Financial statements are slow to fetch, so results are cached with a long
staleness and only a bounded number of new tickers is fetched per run — the
cache fills up over a few runs and daily runs stay fast. Failures never raise.
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta

import yfinance as yf

from .config import DATA

DEEP_CACHE = DATA / "deep_fundamentals.json"
MAX_AGE_DAYS = 30
# Bounded per run so CI stays fast; raise via env for the one-time full build.
MAX_NEW_PER_RUN = int(os.environ.get("STOCK_RADAR_DEEP_MAX", "45"))
FETCH_PAUSE = 0.4


def _load():
    if DEEP_CACHE.exists():
        try:
            return json.loads(DEEP_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save(cache):
    DEEP_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _row(df, names, col=0):
    """Look up a statement line item by trying several possible labels."""
    if df is None or getattr(df, "empty", True) or df.shape[1] <= col:
        return None
    for n in names:
        if n in df.index:
            try:
                val = df.iloc[df.index.get_loc(n), col]
                val = float(val)
                return val if val == val else None
            except Exception:  # noqa: BLE001
                continue
    return None


def _piotroski(inc, bs, cf):
    """Return (score 0-9, computable_count)."""
    pts, n = 0, 0

    def add(cond):
        nonlocal pts, n
        if cond is None:
            return
        n += 1
        if cond:
            pts += 1

    ni = _row(inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"], 0)
    ni_p = _row(inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"], 1)
    ta = _row(bs, ["Total Assets"], 0)
    ta_p = _row(bs, ["Total Assets"], 1)
    cfo = _row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities",
                    "Cash Flow From Continuing Operating Activities"], 0)
    rev = _row(inc, ["Total Revenue", "Operating Revenue"], 0)
    rev_p = _row(inc, ["Total Revenue", "Operating Revenue"], 1)
    gp = _row(inc, ["Gross Profit"], 0)
    gp_p = _row(inc, ["Gross Profit"], 1)
    ca = _row(bs, ["Current Assets", "Total Current Assets"], 0)
    ca_p = _row(bs, ["Current Assets", "Total Current Assets"], 1)
    cl = _row(bs, ["Current Liabilities", "Total Current Liabilities"], 0)
    cl_p = _row(bs, ["Current Liabilities", "Total Current Liabilities"], 1)
    ltd = _row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
    ltd_p = _row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
    sh = _row(bs, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 0)
    sh_p = _row(bs, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], 1)

    roa = (ni / ta) if (ni is not None and ta) else None
    roa_p = (ni_p / ta_p) if (ni_p is not None and ta_p) else None

    add((ni > 0) if ni is not None else None)                       # 1 profitable
    add((cfo > 0) if cfo is not None else None)                     # 2 positive CFO
    add((roa > roa_p) if (roa is not None and roa_p is not None) else None)  # 3 ROA rising
    add((cfo > ni) if (cfo is not None and ni is not None) else None)       # 4 accruals
    add(((ltd / ta) < (ltd_p / ta_p)) if (ltd is not None and ta and ltd_p is not None and ta_p) else None)  # 5 lev down
    add(((ca / cl) > (ca_p / cl_p)) if (ca and cl and ca_p and cl_p) else None)  # 6 current ratio up
    add((sh <= sh_p * 1.01) if (sh is not None and sh_p) else None)          # 7 no dilution
    add(((gp / rev) > (gp_p / rev_p)) if (gp and rev and gp_p and rev_p) else None)  # 8 gross margin up
    add(((rev / ta) > (rev_p / ta_p)) if (rev and ta and rev_p and ta_p) else None)  # 9 asset turnover up
    return (pts, n)


def _altman_z(inc, bs, market_cap):
    ta = _row(bs, ["Total Assets"], 0)
    ca = _row(bs, ["Current Assets", "Total Current Assets"], 0)
    cl = _row(bs, ["Current Liabilities", "Total Current Liabilities"], 0)
    re = _row(bs, ["Retained Earnings"], 0)
    tl = _row(bs, ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"], 0)
    ebit = _row(inc, ["EBIT", "Operating Income", "Total Operating Income As Reported"], 0)
    sales = _row(inc, ["Total Revenue", "Operating Revenue"], 0)
    if not ta or not tl or ca is None or cl is None or not market_cap:
        return None
    try:
        A = (ca - cl) / ta
        B = (re / ta) if re is not None else 0
        C = (ebit / ta) if ebit is not None else 0
        D = market_cap / tl
        E = (sales / ta) if sales else 0
        return round(1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E, 2)
    except Exception:  # noqa: BLE001
        return None


def fetch_deep(symbols, market_caps, max_new=MAX_NEW_PER_RUN, verbose=True):
    """Return {symbol: {piotroski, piotroski_n, altman_z}}. Bounded & cached."""
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
        print(f"Bilanz-Kennzahlen (Piotroski/Altman): {len(symbols) - len(stale)} aus Cache, "
              f"{len(todo)} neu (von {len(stale)} offen) …")

    for sym in todo:
        rec = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            t = yf.Ticker(sym)
            inc, bs, cf = t.income_stmt, t.balance_sheet, t.cashflow
            pio, pion = _piotroski(inc, bs, cf)
            rec["piotroski"] = pio if pion >= 5 else None
            rec["piotroski_n"] = pion
            rec["altman_z"] = _altman_z(inc, bs, market_caps.get(sym))
        except Exception:  # noqa: BLE001
            rec["piotroski"] = None
            rec["altman_z"] = None
        cache[sym] = rec
        time.sleep(FETCH_PAUSE)

    if todo:
        _save(cache)
    return {s: cache.get(s, {}) for s in symbols}
