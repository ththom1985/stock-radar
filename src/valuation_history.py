"""Auditable own-history valuation baselines from filings and market prices."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import DATA
from .persistence import atomic_write_json, load_json, schema_meta

HISTORY_PATH = DATA / "valuation_history.json"
METRICS = (
    "pe",
    "forward_pe",
    "peg",
    "ev_ebitda",
    "price_to_fcf",
    "price_to_sales",
)
MAX_MONTHS = 60
MIN_FIVE_YEAR_MONTHS = 48
MIN_ANNUAL_POINTS = 4
MIN_HISTORY_METRICS = 2


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _close_on_or_before(frame, raw_date):
    try:
        target = datetime.fromisoformat(str(raw_date)).date()
    except (TypeError, ValueError):
        return None
    if frame is None or getattr(frame, "empty", True) or "Close" not in frame:
        return None
    available = [
        float(value)
        for timestamp, value in frame["Close"].dropna().items()
        if timestamp.date() <= target and _number(value) is not None
    ]
    return available[-1] if available else None


def _sec_annual_backfill(sec, frame):
    points = []
    for annual in (sec or {}).get("annual") or []:
        price = _close_on_or_before(frame, annual.get("period_end"))
        if price is None:
            continue
        eps = _number(annual.get("diluted_eps"))
        shares = _number(annual.get("diluted_shares"))
        revenue = _number(annual.get("revenue"))
        free_cash_flow = _number(annual.get("free_cash_flow"))
        market_cap = price * shares if shares is not None else None
        metrics = {}
        if eps is not None:
            metrics["pe"] = price / eps
        if market_cap is not None and revenue is not None:
            metrics["price_to_sales"] = market_cap / revenue
        if market_cap is not None and free_cash_flow is not None:
            metrics["price_to_fcf"] = market_cap / free_cash_flow
        metrics = {
            key: round(value, 6)
            for key, value in metrics.items()
            if _number(value) is not None
        }
        if metrics:
            points.append(
                {
                    "kind": "sec_annual_backfill",
                    "period_end": annual.get("period_end"),
                    "filed": annual.get("filed"),
                    "metrics": metrics,
                    "source_families": [
                        "SEC EDGAR Companyfacts",
                        "completed adjusted daily prices",
                    ],
                }
            )
    return points[-5:]


def update_valuation_history(
    rows,
    observed_at=None,
    *,
    sec_by_symbol=None,
    price_histories=None,
):
    observed_at = observed_at or datetime.now(timezone.utc)
    month = observed_at.strftime("%Y-%m")
    payload = load_json(HISTORY_PATH, expected_type=dict, default={})
    symbols = payload.get("symbols") if isinstance(payload.get("symbols"), dict) else {}
    changed = False
    for row in rows:
        symbol = row.get("symbol")
        if not symbol:
            continue
        metrics = {
            metric: _number(row.get(metric))
            for metric in METRICS
            if _number(row.get(metric)) is not None
        }
        if not metrics:
            continue
        history = list(symbols.get(symbol) or [])
        if not any(point.get("month") == month for point in history):
            history.append({"kind": "monthly_snapshot", "month": month, "metrics": metrics})
            changed = True
        backfilled = _sec_annual_backfill(
            (sec_by_symbol or {}).get(symbol),
            (price_histories or {}).get(symbol),
        )
        if backfilled:
            retained = [
                point
                for point in history
                if point.get("kind") != "sec_annual_backfill"
            ]
            if retained + backfilled != history:
                history = retained + backfilled
                changed = True
        monthly = [point for point in history if point.get("kind") != "sec_annual_backfill"]
        annual = [point for point in history if point.get("kind") == "sec_annual_backfill"]
        symbols[symbol] = monthly[-MAX_MONTHS:] + annual[-5:]
    if changed:
        atomic_write_json(
            HISTORY_PATH,
            {
                "schema": "stock-radar-valuation-history",
                "schema_version": 1,
                "symbols": symbols,
                "_meta": schema_meta(
                    "stock-radar-valuation-history",
                    schema_version=1,
                    cadence="monthly",
                    retention_months=MAX_MONTHS,
                    backfill="SEC annual facts plus completed adjusted daily prices",
                ),
            },
            indent=1,
        )
    return payload_with_symbols(symbols)


def payload_with_symbols(symbols):
    return {
        "symbols": symbols,
        "status": {
            "cadence": "monthly",
            "retention_months": MAX_MONTHS,
            "minimum_months_for_five_year_baseline": MIN_FIVE_YEAR_MONTHS,
        },
    }


def five_year_averages(history, symbol):
    points = ((history or {}).get("symbols") or {}).get(symbol) or []
    annual = [point for point in points if point.get("kind") == "sec_annual_backfill"]
    monthly = [point for point in points if point.get("kind") != "sec_annual_backfill"]
    annual_counts = {}
    annual_result = {}
    for metric in METRICS:
        values = [
            _number((point.get("metrics") or {}).get(metric))
            for point in annual[-5:]
        ]
        values = [value for value in values if value is not None]
        annual_counts[metric] = len(values)
        annual_result[metric] = (
            round(sum(values) / len(values), 4)
            if len(values) >= MIN_ANNUAL_POINTS
            else None
        )
    supported_annual_metrics = [
        metric for metric, value in annual_result.items() if value is not None
    ]
    annual_complete = (
        len(annual) >= MIN_ANNUAL_POINTS
        and len(supported_annual_metrics) >= MIN_HISTORY_METRICS
    )
    monthly_result = {}
    for metric in METRICS:
        values = [
            _number((point.get("metrics") or {}).get(metric))
            for point in monthly[-MAX_MONTHS:]
        ]
        values = [value for value in values if value is not None]
        monthly_result[metric] = (
            round(sum(values) / len(values), 4)
            if len(values) >= MIN_FIVE_YEAR_MONTHS
            else None
        )
    monthly_complete = (
        len(monthly[-MAX_MONTHS:]) >= MIN_FIVE_YEAR_MONTHS
        and sum(value is not None for value in monthly_result.values())
        >= MIN_HISTORY_METRICS
    )
    result = annual_result if annual_complete else monthly_result
    return {
        "metrics": result,
        "months_available": len(monthly[-MAX_MONTHS:]),
        "annual_points_available": len(annual[-5:]),
        "metric_observation_counts": annual_counts,
        "supported_metrics": supported_annual_metrics if annual_complete else [
            metric for metric, value in monthly_result.items() if value is not None
        ],
        "complete": annual_complete or monthly_complete,
        "history_type": (
            "sec_annual_backfill"
            if annual_complete
            else "monthly_snapshots"
            if monthly_complete
            else "insufficient"
        ),
    }
