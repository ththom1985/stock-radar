"""Honest validation backtest: did a high (technical) Radar score historically
predict higher forward returns?

Method (point-in-time, no look-ahead):
  - Download ~4 years of daily prices for a liquid subset.
  - At each month-end, compute score_longterm() from data UP TO that date only.
  - Rank stocks into quintiles; measure the next-month forward return per quintile.
  - Report the rank correlation (Information Coefficient) between score and forward
    return, and the top-minus-bottom-quintile spread.

Honest limitations (stated in the output):
  - Validates the TECHNICAL score only — free point-in-time fundamentals don't exist.
  - Survivorship bias: uses today's names (delisted losers are absent) → results
    are optimistic.
  - No costs/slippage; monthly rebalance.
"""
import json
from datetime import datetime, timezone

import yfinance as yf

from .config import DATA
from .indicators import compute_features
from .score import score_longterm

BACKTEST_CACHE = DATA / "backtest.json"


def _spearman(a, b):
    """Rank correlation of two equal-length lists."""
    n = len(a)
    if n < 5:
        return None

    def _ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0] * n
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return round(num / (da * db), 4) if da and db else None


def run_backtest(symbols, start="2021-01-01", fwd=21, buckets=5, min_hist=220, verbose=True):
    if verbose:
        print(f"Backtest: lade Historie für {len(symbols)} Titel …")
    hist = {}
    for i, s in enumerate(symbols, 1):
        try:
            df = yf.Ticker(s).history(start=start, auto_adjust=True)
            if df is not None and len(df) > min_hist + fwd:
                hist[s] = df
        except Exception:  # noqa: BLE001
            pass
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(symbols)} …")
    if len(hist) < buckets * 3:
        return {"error": "zu wenig Daten"}

    # monthly rebalance dates = last trading day per (year, month)
    all_dates = sorted(set().union(*[set(df.index) for df in hist.values()]))
    last_of_month = {}
    for d in all_dates:
        last_of_month[(d.year, d.month)] = d
    rebal = sorted(last_of_month.values())

    ic_list, spreads = [], []
    bucket_rets = {b: [] for b in range(buckets)}
    top_beats = 0
    n_used = 0
    for t in rebal:
        scored = []
        for s, df in hist.items():
            sub = df[df.index <= t]
            if len(sub) < min_hist:
                continue
            f = compute_features(sub)
            if not f:
                continue
            sc, _ = score_longterm(f)
            fut = df[df.index > t]
            if len(fut) < fwd:
                continue
            p0 = float(sub["Close"].iloc[-1])
            p1 = float(fut["Close"].iloc[fwd - 1])
            if p0:
                scored.append((s, sc, (p1 / p0 - 1) * 100))
        if len(scored) < buckets * 3:
            continue
        n_used += 1
        scored.sort(key=lambda x: x[1])          # ascending score
        n = len(scored)
        this = {b: [] for b in range(buckets)}
        for rank, (_s, _sc, fr) in enumerate(scored):
            b = min(buckets - 1, rank * buckets // n)
            bucket_rets[b].append(fr)
            this[b].append(fr)
        ic = _spearman([x[1] for x in scored], [x[2] for x in scored])
        if ic is not None:
            ic_list.append(ic)
        if this[buckets - 1] and this[0]:
            sp = sum(this[buckets - 1]) / len(this[buckets - 1]) - sum(this[0]) / len(this[0])
            spreads.append(sp)
            if sp > 0:
                top_beats += 1

    def _avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_names": len(hist), "n_months": n_used, "fwd_days": fwd, "buckets": buckets,
        "avg_ic": _avg(ic_list),
        "bucket_avg_fwd_ret_pct": {b: _avg(bucket_rets[b]) for b in range(buckets)},
        "top_minus_bottom_pct": _avg(spreads),
        "hit_rate_top_beats_bottom_pct": round(top_beats / len(spreads) * 100, 1) if spreads else None,
    }
    try:
        BACKTEST_CACHE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return result
