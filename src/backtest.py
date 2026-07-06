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


def _avg(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def run_backtest(symbols, start="2020-01-01", horizons=(21, 63, 126, 252),
                 buckets=5, min_hist=220, verbose=True):
    """Score is horizon-independent, so compute it once per (name, month) and
    measure forward returns over several horizons. Returns results per horizon."""
    if verbose:
        print(f"Backtest: lade Historie für {len(symbols)} Titel …")
    hist = {}
    for i, s in enumerate(symbols, 1):
        try:
            df = yf.Ticker(s).history(start=start, auto_adjust=True)
            if df is not None and len(df) > min_hist + max(horizons):
                hist[s] = df
        except Exception:  # noqa: BLE001
            pass
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(symbols)} …")
    if len(hist) < buckets * 3:
        return {"error": "zu wenig Daten"}

    all_dates = sorted(set().union(*[set(df.index) for df in hist.values()]))
    last_of_month = {}
    for d in all_dates:
        last_of_month[(d.year, d.month)] = d
    rebal = sorted(last_of_month.values())

    # per horizon accumulators
    acc = {h: {"ic": [], "spread": [], "top_beats": 0, "n_spread": 0,
               "buckets": {b: [] for b in range(buckets)}} for h in horizons}
    for t in rebal:
        scored = []   # (score, {h: fwd_ret})
        for s, df in hist.items():
            sub = df[df.index <= t]
            if len(sub) < min_hist:
                continue
            f = compute_features(sub)
            if not f:
                continue
            sc, _ = score_longterm(f)
            p0 = float(sub["Close"].iloc[-1])
            fut = df[df.index > t]
            if not p0 or len(fut) < min(horizons):
                continue
            frs = {}
            for h in horizons:
                if len(fut) >= h:
                    frs[h] = (float(fut["Close"].iloc[h - 1]) / p0 - 1) * 100
            if frs:
                scored.append((sc, frs))
        if len(scored) < buckets * 3:
            continue
        for h in horizons:
            sub_scored = [(sc, frs[h]) for sc, frs in scored if h in frs]
            if len(sub_scored) < buckets * 3:
                continue
            sub_scored.sort(key=lambda x: x[0])
            n = len(sub_scored)
            this = {b: [] for b in range(buckets)}
            for rank, (_sc, fr) in enumerate(sub_scored):
                b = min(buckets - 1, rank * buckets // n)
                acc[h]["buckets"][b].append(fr)
                this[b].append(fr)
            ic = _spearman([x[0] for x in sub_scored], [x[1] for x in sub_scored])
            if ic is not None:
                acc[h]["ic"].append(ic)
            if this[buckets - 1] and this[0]:
                sp = _avg(this[buckets - 1]) - _avg(this[0])
                acc[h]["spread"].append(sp)
                acc[h]["n_spread"] += 1
                if sp > 0:
                    acc[h]["top_beats"] += 1

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_names": len(hist), "n_months": len(rebal), "buckets": buckets, "by_horizon": {}}
    for h in horizons:
        a = acc[h]
        out["by_horizon"][f"{h}d"] = {
            "avg_ic": _avg(a["ic"]),
            "bucket_avg_fwd_ret_pct": {b: _avg(a["buckets"][b]) for b in range(buckets)},
            "top_minus_bottom_pct": _avg(a["spread"]),
            "hit_rate_top_beats_bottom_pct": round(a["top_beats"] / a["n_spread"] * 100, 1) if a["n_spread"] else None,
        }
    try:
        BACKTEST_CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return out
