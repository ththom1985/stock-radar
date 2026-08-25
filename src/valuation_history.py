"""Monthly point-in-time valuation snapshots for an eventual five-year baseline."""
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


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def update_valuation_history(rows, observed_at=None):
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
        if history and history[-1].get("month") == month:
            continue
        history.append({"month": month, "metrics": metrics})
        symbols[symbol] = history[-MAX_MONTHS:]
        changed = True
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
    result = {}
    for metric in METRICS:
        values = [
            _number((point.get("metrics") or {}).get(metric))
            for point in points[-MAX_MONTHS:]
        ]
        values = [value for value in values if value is not None]
        result[metric] = (
            round(sum(values) / len(values), 4)
            if len(values) >= MIN_FIVE_YEAR_MONTHS
            else None
        )
    return {
        "metrics": result,
        "months_available": len(points[-MAX_MONTHS:]),
        "complete": len(points[-MAX_MONTHS:]) >= MIN_FIVE_YEAR_MONTHS,
    }
