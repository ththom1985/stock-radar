"""Leakage-controlled validation of the technical score only.

The deployed composite remains UNVALIDATED: point-in-time fundamentals, news
and analyst histories are unavailable. This module makes no alpha claim.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from .config import DATA, ROOT
from .fetch import completed_daily_bars
from .indicators import compute_features
from .persistence import SCHEMA_VERSION, atomic_write_json, schema_meta
from .score import score_longterm

BACKTEST_CACHE = DATA / "backtest.json"


def _average_ranks(values):
    """Average ranks for ties, zero-based; suitable for Spearman correlation."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(left, right):
    if len(left) != len(right) or len(left) < 5:
        return None
    rank_left, rank_right = _average_ranks(left), _average_ranks(right)
    n = len(left)
    mean_left = sum(rank_left) / n
    mean_right = sum(rank_right) / n
    numerator = sum(
        (rank_left[i] - mean_left) * (rank_right[i] - mean_right) for i in range(n)
    )
    denominator_left = sum((value - mean_left) ** 2 for value in rank_left) ** 0.5
    denominator_right = sum((value - mean_right) ** 2 for value in rank_right) ** 0.5
    return numerator / (denominator_left * denominator_right) if denominator_left and denominator_right else None


def _tie_aware_bucket_ids(scores, buckets):
    """Assign equal scores to one bucket; never split a tie at a boundary."""
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    assigned = [0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        midpoint_rank = (start + end - 1) / 2
        bucket = min(buckets - 1, int(midpoint_rank * buckets / len(scores)))
        for position in range(start, end):
            assigned[order[position]] = bucket
        start = end
    return assigned


def _average(values):
    return sum(values) / len(values) if values else None


def _code_hash() -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/backtest.py",
        "src/indicators.py",
        "src/score.py",
        "src/config.py",
    ):
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_backtest(
    symbols,
    start="2020-01-01",
    horizons=(21, 63, 126, 252),
    buckets=5,
    min_hist=220,
    round_trip_cost_bps=20.0,
    verbose=True,
):
    """Signal at completed close t; enter strictly later at t+1 adjusted open."""
    history = {}
    failures = {}
    for index, symbol in enumerate(symbols, 1):
        try:
            raw = yf.Ticker(symbol).history(start=start, auto_adjust=True, actions=True, timeout=30)
            frame, _ = completed_daily_bars(raw, symbol=symbol)
            if len(frame) > min_hist + max(horizons):
                history[symbol] = frame
            else:
                failures[symbol] = "insufficient completed history"
        except Exception as exc:
            failures[symbol] = str(exc)[:200]
        if verbose and index % 25 == 0:
            print(f"  {index}/{len(symbols)}")
    if len(history) < buckets * 3:
        raise RuntimeError(f"Backtest has only {len(history)} usable symbols")

    all_dates = sorted(set().union(*(set(frame.index) for frame in history.values())))
    last_of_month = {}
    for timestamp in all_dates:
        last_of_month[(timestamp.year, timestamp.month)] = timestamp
    rebalance_dates = sorted(last_of_month.values())
    accumulators = {
        horizon: {
            "ic": [],
            "spread": [],
            "top_beats": 0,
            "n_spread": 0,
            "samples": 0,
            "months": 0,
            "buckets": {bucket: [] for bucket in range(buckets)},
        }
        for horizon in horizons
    }

    for signal_date in rebalance_dates:
        scored = []
        for symbol, frame in history.items():
            past = frame[frame.index <= signal_date]
            future = frame[frame.index > signal_date]
            if len(past) < min_hist or len(future) < min(horizons):
                continue
            features = compute_features(past)
            if not features:
                continue
            score, _ = score_longterm(features)
            next_open = float(future["Open"].iloc[0])
            if next_open <= 0:
                continue
            forward = {}
            for horizon in horizons:
                if len(future) >= horizon:
                    exit_close = float(future["Close"].iloc[horizon - 1])
                    gross_pct = (exit_close / next_open - 1) * 100
                    forward[horizon] = gross_pct - round_trip_cost_bps / 100
            if forward:
                scored.append((symbol, score, forward))

        for horizon in horizons:
            horizon_rows = [(symbol, score, returns[horizon]) for symbol, score, returns in scored if horizon in returns]
            if len(horizon_rows) < buckets * 3:
                continue
            horizon_rows.sort(key=lambda item: item[1])
            n = len(horizon_rows)
            monthly = {bucket: [] for bucket in range(buckets)}
            bucket_ids = _tie_aware_bucket_ids(
                [row[1] for row in horizon_rows],
                buckets,
            )
            for bucket, (_symbol, _score, result) in zip(bucket_ids, horizon_rows):
                accumulators[horizon]["buckets"][bucket].append(result)
                monthly[bucket].append(result)
            ic = _spearman(
                [row[1] for row in horizon_rows],
                [row[2] for row in horizon_rows],
            )
            if ic is not None:
                accumulators[horizon]["ic"].append(ic)
            top, bottom = monthly[buckets - 1], monthly[0]
            if top and bottom:
                spread = _average(top) - _average(bottom)
                accumulators[horizon]["spread"].append(spread)
                accumulators[horizon]["n_spread"] += 1
                accumulators[horizon]["top_beats"] += spread > 0
            accumulators[horizon]["samples"] += n
            accumulators[horizon]["months"] += 1

    output = {
        "schema": "stock-radar-backtest",
        "schema_version": SCHEMA_VERSION,
        "model_status": "technical_score_validation_only",
        "deployed_composite_status": "unvalidated",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": {
            "symbols_requested": list(symbols),
            "symbols_used": sorted(history),
            "failed_symbols": failures,
            "start": min(frame.index[0] for frame in history.values()).isoformat(),
            "end": max(frame.index[-1] for frame in history.values()).isoformat(),
            "signal_timing": "completed close t",
            "entry_timing": "next available adjusted open t+1",
            "round_trip_cost_bps": round_trip_cost_bps,
            "code_config_sha256": _code_hash(),
            "survivorship_bias": True,
        },
        "buckets": buckets,
        "by_horizon": {},
        "_meta": schema_meta("stock-radar-backtest"),
    }
    for horizon, accumulator in accumulators.items():
        output["by_horizon"][f"{horizon}d"] = {
            "avg_spearman_ic": _average(accumulator["ic"]),
            "bucket_avg_net_return_pct": {
                str(bucket): _average(accumulator["buckets"][bucket]) for bucket in range(buckets)
            },
            "top_minus_bottom_net_pct": _average(accumulator["spread"]),
            "hit_rate_top_beats_bottom_pct": (
                accumulator["top_beats"] / accumulator["n_spread"] * 100
                if accumulator["n_spread"]
                else None
            ),
            "sample_count": accumulator["samples"],
            "rebalance_count": accumulator["months"],
        }
    atomic_write_json(BACKTEST_CACHE, output, indent=1)
    return output
