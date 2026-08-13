"""Unvalidated paper simulation with next-completed-bar execution.

Signals only create pending long orders. A fill requires a strictly later
completed daily bar and uses that bar's raw open plus explicit costs/slippage.
No borrowing or shorting is supported.
"""
from __future__ import annotations

import copy
import os
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from .assets import COMPANY_EQUITY
from .config import DATA
from .persistence import SCHEMA_VERSION, atomic_write_json, load_json, schema_meta, utc_now

PORTFOLIO_FILE = DATA / "portfolio.json"

START_CAPITAL = 10_000.0
TARGET_POSITIONS = 10
BUY_SCORE = 60
SELL_SCORE = 42
MIN_AVG_DOLLAR_VOLUME = float(os.environ.get("STOCK_RADAR_MIN_DOLLAR_VOLUME", "20000000"))
MAX_ATR_PCT = float(os.environ.get("STOCK_RADAR_MAX_PAPER_ATR_PCT", "5"))
MAX_ANNUAL_VOL_PCT = float(os.environ.get("STOCK_RADAR_MAX_PAPER_ANNUAL_VOL_PCT", "60"))
MAX_PER_SECTOR = int(os.environ.get("STOCK_RADAR_MAX_PAPER_PER_SECTOR", "3"))
MAX_PER_COUNTRY = int(os.environ.get("STOCK_RADAR_MAX_PAPER_PER_COUNTRY", "4"))
ORDER_MAX_AGE_DAYS = int(os.environ.get("STOCK_RADAR_PAPER_ORDER_MAX_AGE_DAYS", "7"))
SLIPPAGE_BPS = float(os.environ.get("STOCK_RADAR_PAPER_SLIPPAGE_BPS", "10"))
COMMISSION_BPS = float(os.environ.get("STOCK_RADAR_PAPER_COMMISSION_BPS", "5"))


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _initial(today: str) -> dict[str, Any]:
    return {
        "schema": "stock-radar-paper-portfolio",
        "schema_version": SCHEMA_VERSION,
        "simulation_status": "unvalidated",
        "performance_actionable": False,
        "created": today,
        "base_currency": "USD",
        "starting_cash": START_CAPITAL,
        "cash": START_CAPITAL,
        "positions": {},
        "pending_orders": [],
        "ledger": [],
        "equity_curve": [],
        "assumptions": {
            "long_only": True,
            "execution": "strictly later completed daily bar open",
            "slippage_bps": SLIPPAGE_BPS,
            "commission_bps": COMMISSION_BPS,
            "min_avg_dollar_volume": MIN_AVG_DOLLAR_VOLUME,
            "max_atr_pct": MAX_ATR_PCT,
            "max_annual_vol_pct": MAX_ANNUAL_VOL_PCT,
            "max_per_sector": MAX_PER_SECTOR,
            "max_per_country": MAX_PER_COUNTRY,
            "currency": "USD only until point-in-time FX exists",
            "diversification": "issuer uniqueness plus sector/country caps; no correlation claim",
            "order_not_before": "UTC creation date + two calendar dates",
            "corporate_actions": (
                "best effort from Yahoo daily action columns; missed-run gaps are not guaranteed"
            ),
        },
        "_meta": schema_meta("stock-radar-paper-portfolio"),
    }


def _migrate_legacy(legacy: dict[str, Any], today: str) -> dict[str, Any]:
    """Preserve every legacy field and convert positions without inventing fills."""
    migrated = _initial(today)
    migrated["created"] = legacy.get("created") or today
    migrated["cash"] = float(legacy.get("cash") or 0.0)
    migrated["legacy_migrated"] = True
    migrated["migration_requires_review"] = True
    migrated["simulation_status"] = "legacy_frozen_unvalidated"
    migrated["legacy_archive"] = copy.deepcopy(legacy)
    migrated["assumptions"]["legacy_accounting"] = (
        "pre-v2 positions lacked next-bar fills, quantities and cost records; "
        "all performance remains non-actionable"
    )
    for symbol, old in (legacy.get("positions") or {}).items():
        entry = old.get("entry_price")
        stake = old.get("stake_eur")
        if not isinstance(entry, (int, float)) or entry <= 0:
            continue
        if not isinstance(stake, (int, float)) or stake <= 0:
            continue
        quantity = stake / entry
        migrated["positions"][symbol] = {
            "symbol": symbol,
            "name": old.get("name"),
            "quantity": quantity,
            "entry_price": entry,
            "cost_basis": stake,
            "entry_bar_date": old.get("entry_date"),
            "last_price": old.get("last_price") or entry,
            "last_mark_bar_date": old.get("entry_date"),
            "last_action_bar_date": today,
            "legacy": True,
        }
        migrated["ledger"].append(
            {
                "transaction_id": str(uuid.uuid4()),
                "type": "LEGACY_MIGRATION",
                "symbol": symbol,
                "quantity": quantity,
                "price": entry,
                "gross_value": stake,
                "timestamp": utc_now(),
                "note": "Imported without claiming executable historical fills",
            }
        )
    return migrated


def _start_clean_v2_after_review(portfolio: dict[str, Any]) -> dict[str, Any]:
    portfolio = copy.deepcopy(portfolio)
    portfolio["legacy_frozen_positions"] = portfolio.get("positions") or {}
    portfolio["positions"] = {}
    portfolio["pending_orders"] = []
    portfolio["equity_curve"] = []
    portfolio["cash"] = START_CAPITAL
    portfolio["starting_cash"] = START_CAPITAL
    portfolio["migration_requires_review"] = False
    portfolio["simulation_status"] = "unvalidated"
    portfolio["clean_v2_started_at"] = utc_now()
    portfolio["ledger"].append(
        {
            "transaction_id": str(uuid.uuid4()),
            "type": "CLEAN_V2_SIMULATION_STARTED",
            "timestamp": utc_now(),
            "note": "Explicit STOCK_RADAR_START_NEW_PAPER=1 decision; legacy data preserved",
        }
    )
    return portfolio


def load_portfolio(today: str | None = None) -> dict[str, Any]:
    today = today or _today()
    raw = load_json(PORTFOLIO_FILE, expected_type=dict, default=None)
    if raw is None:
        return _initial(today)
    if raw.get("schema") == "stock-radar-paper-portfolio" and raw.get("schema_version") == SCHEMA_VERSION:
        if (
            raw.get("migration_requires_review")
            and os.environ.get("STOCK_RADAR_START_NEW_PAPER") == "1"
        ):
            return _start_clean_v2_after_review(raw)
        return raw
    migrated = _migrate_legacy(raw, today)
    if os.environ.get("STOCK_RADAR_START_NEW_PAPER") == "1":
        return _start_clean_v2_after_review(migrated)
    return migrated


def _save(portfolio: dict[str, Any]) -> None:
    portfolio["_meta"] = schema_meta("stock-radar-paper-portfolio")
    atomic_write_json(PORTFOLIO_FILE, portfolio, indent=1)


def _order(
    action: str,
    row: dict[str, Any],
    reason: str,
    observed_at: datetime,
    **extra: Any,
) -> dict[str, Any]:
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    created = observed_at.astimezone(timezone.utc)
    return {
        "order_id": str(uuid.uuid4()),
        "action": action,
        "symbol": row["symbol"],
        "name": row.get("name"),
        "signal_bar_date": row["bar_date"],
        "signal_timestamp": row.get("bar_timestamp"),
        "created_at": created.isoformat(timespec="seconds"),
        "observed_at": created.isoformat(timespec="seconds"),
        "not_before_bar_date": (created.date() + timedelta(days=2)).isoformat(),
        "status": "pending",
        "reason": reason,
        **extra,
    }


def _row_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    actions = list(row.get("corporate_actions") or [])
    if not actions and (
        (isinstance(row.get("stock_split"), (int, float)) and row.get("stock_split"))
        or (isinstance(row.get("dividend_usd"), (int, float)) and row.get("dividend_usd"))
    ):
        actions = [
            {
                "bar_date": row.get("bar_date"),
                "stock_split": row.get("stock_split") or 0.0,
                "dividend_usd": row.get("dividend_usd") or 0.0,
            }
        ]
    return sorted(
        (action for action in actions if action.get("bar_date")),
        key=lambda action: action["bar_date"],
    )


def _apply_corporate_actions(portfolio: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    for symbol, position in list(portfolio["positions"].items()):
        if position.get("legacy"):
            continue
        row = rows.get(symbol)
        if not row:
            continue
        latest_bar_date = row.get("bar_date")
        if not latest_bar_date:
            continue
        processed = position.setdefault("processed_actions", {})
        entry_bar_date = str(position.get("entry_bar_date") or "")
        for action in _row_actions(row):
            bar_date = action["bar_date"]
            if bar_date <= entry_bar_date or bar_date > latest_bar_date:
                continue
            split = action.get("stock_split")
            if isinstance(split, (int, float)) and split not in (0, 1) and split > 0:
                base_key = f"{symbol}|SPLIT|{bar_date}"
                previous = processed.get(base_key)
                previous_ratio = (
                    float(previous["value"]) if isinstance(previous, dict) else 1.0
                )
                if previous and previous_ratio == float(split):
                    split = None
                else:
                    correction_ratio = float(split) / previous_ratio
            else:
                base_key = None
                correction_ratio = None
            if split is not None and correction_ratio is not None:
                old_quantity = position["quantity"]
                position["quantity"] *= correction_ratio
                position["entry_price"] /= correction_ratio
                if isinstance(position.get("last_price"), (int, float)):
                    position["last_price"] /= correction_ratio
                for pending in portfolio["pending_orders"]:
                    if pending.get("symbol") == symbol and pending.get("action") == "SELL":
                        pending["quantity"] = (
                            float(pending.get("quantity") or old_quantity)
                            * correction_ratio
                        )
                version_key = f"{base_key}|{float(split):.12g}"
                processed[base_key] = {
                    "value": float(split),
                    "version_key": version_key,
                    "processed_at": utc_now(),
                }
                portfolio["ledger"].append(
                    {
                        "transaction_id": str(uuid.uuid4()),
                        "type": "SPLIT" if previous is None else "SPLIT_CORRECTION",
                        "symbol": symbol,
                        "bar_date": bar_date,
                        "ratio": float(split),
                        "correction_ratio": correction_ratio,
                        "action_key": version_key,
                        "quantity_before": old_quantity,
                        "quantity_after": position["quantity"],
                        "timestamp": utc_now(),
                    }
                )
            dividend = action.get("dividend_usd")
            if isinstance(dividend, (int, float)) and dividend > 0:
                base_key = f"{symbol}|DIVIDEND|{bar_date}"
                previous = processed.get(base_key)
                previous_value = (
                    float(previous["value"]) if isinstance(previous, dict) else 0.0
                )
                delta = float(dividend) - previous_value
                if delta == 0:
                    continue
                quantity_basis = (
                    float(previous["quantity_basis"])
                    if isinstance(previous, dict)
                    and isinstance(previous.get("quantity_basis"), (int, float))
                    else position["quantity"]
                )
                amount = quantity_basis * delta
                portfolio["cash"] += amount
                version_key = f"{base_key}|{float(dividend):.12g}"
                processed[base_key] = {
                    "value": float(dividend),
                    "version_key": version_key,
                    "processed_at": utc_now(),
                    "quantity_basis": quantity_basis,
                }
                portfolio["ledger"].append(
                    {
                        "transaction_id": str(uuid.uuid4()),
                        "type": (
                            "DIVIDEND" if previous is None else "DIVIDEND_CORRECTION"
                        ),
                        "symbol": symbol,
                        "bar_date": bar_date,
                        "quantity": quantity_basis,
                        "dividend_per_share": float(dividend),
                        "previous_dividend_per_share": previous_value,
                        "cash_delta_per_share": delta,
                        "cash_amount": amount,
                        "action_key": version_key,
                        "timestamp": utc_now(),
                    }
                )
        if len(processed) > 256:
            keep = sorted(
                processed,
                key=lambda key: key.split("|")[2],
                reverse=True,
            )[:256]
            position["processed_actions"] = {
                key: processed[key] for key in keep
            }
        if processed:
            position["last_confirmed_action_date"] = max(
                key.split("|")[2] for key in processed
            )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _execute_pending(
    portfolio: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    observed_at: datetime,
    *,
    allow_fills: bool = True,
) -> None:
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_at = observed_at.astimezone(timezone.utc)
    remaining = []
    for order in portfolio["pending_orders"]:
        try:
            expired = observed_at > _parse_utc(order["created_at"]) + timedelta(
                days=ORDER_MAX_AGE_DAYS
            )
        except (KeyError, TypeError, ValueError):
            expired = True
        if expired:
            order["status"] = "cancelled"
            order["cancelled_at"] = observed_at.isoformat(timespec="seconds")
            order["cancel_reason"] = "pending order expired without an eligible later session"
            portfolio["ledger"].append({**order, "type": "ORDER_CANCELLED"})
            continue
        if not allow_fills:
            remaining.append(order)
            continue
        row = rows.get(order["symbol"])
        bar_date = row.get("bar_date") if row else None
        open_price = row.get("raw_open_usd") if row else None
        if (
            not row
            or not bar_date
            or bar_date <= order["signal_bar_date"]
            or bar_date < order.get("not_before_bar_date", "9999-12-31")
        ):
            remaining.append(order)
            continue
        if row.get("currency") != "USD":
            order["status"] = "cancelled"
            order["cancelled_at"] = observed_at.isoformat(timespec="seconds")
            order["cancel_reason"] = (
                "non-USD paper fill disabled until point-in-time FX exists"
            )
            portfolio["ledger"].append({**order, "type": "ORDER_CANCELLED"})
            continue
        session_open = row.get("session_open_timestamp")
        if session_open:
            try:
                if _parse_utc(session_open) <= _parse_utc(order["created_at"]):
                    remaining.append(order)
                    continue
            except (TypeError, ValueError):
                remaining.append(order)
                continue
        if not isinstance(open_price, (int, float)) or open_price <= 0:
            order["last_error"] = "later bar has no valid raw USD open"
            remaining.append(order)
            continue
        liquidity = row.get("avg_dollar_volume_20_usd")
        if not isinstance(liquidity, (int, float)) or liquidity < MIN_AVG_DOLLAR_VOLUME:
            order["status"] = "cancelled"
            order["cancelled_at"] = utc_now()
            order["cancel_reason"] = "liquidity below configured minimum"
            portfolio["ledger"].append({**order, "type": "ORDER_CANCELLED"})
            continue

        action = order["action"]
        if action == "BUY" and order["symbol"] in portfolio["positions"]:
            order["status"] = "cancelled"
            order["cancel_reason"] = "position already exists"
            portfolio["ledger"].append({**order, "type": "ORDER_CANCELLED"})
            continue
        execution_price = open_price * (
            1 + SLIPPAGE_BPS / 10_000 if action == "BUY" else 1 - SLIPPAGE_BPS / 10_000
        )
        if action == "BUY":
            budget = min(float(order["target_notional"]), portfolio["cash"])
            gross_budget = budget / (1 + COMMISSION_BPS / 10_000)
            quantity = gross_budget / execution_price
            gross = quantity * execution_price
            commission = gross * COMMISSION_BPS / 10_000
            if quantity <= 0 or gross + commission > portfolio["cash"] + 1e-9:
                order["status"] = "cancelled"
                order["cancel_reason"] = "insufficient cash"
                portfolio["ledger"].append({**order, "type": "ORDER_CANCELLED"})
                continue
            portfolio["cash"] -= gross + commission
            portfolio["positions"][order["symbol"]] = {
                "symbol": order["symbol"],
                "name": order.get("name"),
                "quantity": quantity,
                "entry_price": execution_price,
                "cost_basis": gross + commission,
                "entry_bar_date": bar_date,
                "last_price": row.get("raw_close_usd") or execution_price,
                "last_mark_bar_date": bar_date,
                "last_action_bar_date": bar_date,
                "processed_actions": {},
                "legacy": False,
                "issuer_key": order.get("issuer_key"),
                "sector": order.get("sector"),
                "country": order.get("country"),
                "currency": "USD",
            }
        else:
            position = portfolio["positions"].get(order["symbol"])
            if not position:
                continue
            quantity = min(float(order.get("quantity") or 0), position["quantity"])
            gross = quantity * execution_price
            commission = gross * COMMISSION_BPS / 10_000
            portfolio["cash"] += gross - commission
            position["quantity"] -= quantity
            if position["quantity"] <= 1e-12:
                del portfolio["positions"][order["symbol"]]

        portfolio["ledger"].append(
            {
                "transaction_id": str(uuid.uuid4()),
                "type": "FILL",
                "order_id": order["order_id"],
                "action": action,
                "symbol": order["symbol"],
                "signal_bar_date": order["signal_bar_date"],
                "signal_timestamp": order.get("signal_timestamp"),
                "created_at": order.get("created_at"),
                "not_before_bar_date": order.get("not_before_bar_date"),
                "fill_bar_date": bar_date,
                "fill_timestamp": row.get("bar_timestamp"),
                "fill_session_open_timestamp": row.get("session_open_timestamp"),
                "fill_observed_at": observed_at.isoformat(timespec="seconds"),
                "quantity": quantity,
                "raw_fill_price": open_price,
                "execution_price": execution_price,
                "gross_value": gross,
                "commission": commission,
                "slippage_bps": SLIPPAGE_BPS,
                "timestamp": utc_now(),
            }
        )
    portfolio["pending_orders"] = remaining


def _queue_signals(
    portfolio: dict[str, Any],
    rows: list[dict[str, Any]],
    observed_at: datetime,
) -> None:
    if portfolio.get("migration_requires_review"):
        return
    by_symbol = {row["symbol"]: row for row in rows}
    pending_symbols = {order["symbol"] for order in portfolio["pending_orders"]}
    for symbol, position in list(portfolio["positions"].items()):
        row = by_symbol.get(symbol)
        if not row or symbol in pending_symbols or position.get("legacy"):
            continue
        score = row.get("radar_score")
        direction = row.get("daily_signal_direction")
        if (isinstance(score, (int, float)) and score < SELL_SCORE) or direction == "NEGATIVE":
            portfolio["pending_orders"].append(
                _order(
                    "SELL",
                    row,
                    "core heuristic weakened; simulated long exit queued",
                    observed_at,
                    quantity=position["quantity"],
                )
            )
            pending_symbols.add(symbol)

    open_or_pending = len(portfolio["positions"]) + sum(
        order["action"] == "BUY" for order in portfolio["pending_orders"]
    )
    slots = max(0, TARGET_POSITIONS - open_or_pending)
    if not slots:
        return
    candidates = sorted(
        (
            row
            for row in rows
            if row.get("asset_type") == COMPANY_EQUITY
            and row.get("currency") == "USD"
            and (row.get("paper_eligibility") or {}).get("eligible") is True
            and isinstance(row.get("radar_score"), (int, float))
            and row["radar_score"] >= BUY_SCORE
            and row.get("daily_signal_direction") != "NEGATIVE"
            and isinstance(row.get("avg_dollar_volume_20_usd"), (int, float))
            and row["avg_dollar_volume_20_usd"] >= MIN_AVG_DOLLAR_VOLUME
            and isinstance(row.get("atr_pct"), (int, float))
            and row["atr_pct"] <= MAX_ATR_PCT
            and isinstance(row.get("vol_annual_pct"), (int, float))
            and row["vol_annual_pct"] <= MAX_ANNUAL_VOL_PCT
            and row["symbol"] not in portfolio["positions"]
            and row["symbol"] not in pending_symbols
        ),
        key=lambda row: row["radar_score"],
        reverse=True,
    )
    available_equity = portfolio["cash"] + sum(
        position["quantity"] * (position.get("last_price") or position["entry_price"])
        for position in portfolio["positions"].values()
    )
    target = available_equity / TARGET_POSITIONS
    issuer_keys = {
        position.get("issuer_key")
        for position in portfolio["positions"].values()
        if position.get("issuer_key")
    }
    sector_counts = Counter(
        position.get("sector") or "unknown"
        for position in portfolio["positions"].values()
    )
    country_counts = Counter(
        position.get("country") or "unknown"
        for position in portfolio["positions"].values()
    )
    for order in portfolio["pending_orders"]:
        if order.get("action") != "BUY":
            continue
        if order.get("issuer_key"):
            issuer_keys.add(order["issuer_key"])
        sector_counts[order.get("sector") or "unknown"] += 1
        country_counts[order.get("country") or "unknown"] += 1

    queued = 0
    for row in candidates:
        if queued >= slots:
            break
        issuer_key = row.get("issuer_key") or f"symbol:{row['symbol']}"
        sector = row.get("sector") or "unknown"
        country = row.get("cc") or row.get("provider_country") or "unknown"
        if issuer_key in issuer_keys:
            continue
        sector_limit = 1 if sector == "unknown" else MAX_PER_SECTOR
        country_limit = 1 if country == "unknown" else MAX_PER_COUNTRY
        if sector_counts[sector] >= sector_limit or country_counts[country] >= country_limit:
            continue
        portfolio["pending_orders"].append(
            _order(
                "BUY",
                row,
                "research ranking signal queued for conservative later bar",
                observed_at,
                target_notional=target,
                issuer_key=issuer_key,
                sector=sector,
                country=country,
                currency="USD",
            )
        )
        issuer_keys.add(issuer_key)
        sector_counts[sector] += 1
        country_counts[country] += 1
        queued += 1


def _mark_positions(
    portfolio: dict[str, Any],
    rows: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, str | None]]:
    invested = 0.0
    valuation_dates: dict[str, str | None] = {}
    for symbol, position in portfolio["positions"].items():
        row = rows.get(symbol)
        price = (
            position.get("last_price")
            if position.get("legacy")
            else row.get("raw_close_usd") if row else position.get("last_price")
        )
        if isinstance(price, (int, float)) and price > 0:
            position["last_price"] = price
            if row and not position.get("legacy"):
                position["last_mark_bar_date"] = row.get("bar_date")
        valuation_dates[symbol] = (
            row.get("bar_date")
            if row and not position.get("legacy") and row.get("bar_date")
            else None
        )
        invested += position["quantity"] * (position.get("last_price") or position["entry_price"])
    return invested, valuation_dates


def _mark_and_snapshot(
    portfolio: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    today: str,
    benchmarks: dict[str, dict[str, Any]] | None,
    *,
    record_snapshot: bool = True,
) -> float:
    invested, position_valuation_dates = _mark_positions(portfolio, rows)
    equity = portfolio["cash"] + invested
    valuation_dates = set(position_valuation_dates.values())
    if not position_valuation_dates:
        row_dates = {row.get("bar_date") for row in rows.values() if row.get("bar_date")}
        valuation_dates = row_dates if len(row_dates) == 1 else set()
    as_of_bar_date = (
        next(iter(valuation_dates))
        if len(valuation_dates) == 1 and None not in valuation_dates
        else None
    )
    snapshot = {
        "date": today,
        "as_of_bar_date": as_of_bar_date,
        "valuation_status": (
            "aligned_completed_bar" if as_of_bar_date else "mixed_or_stale"
        ),
        "position_valuation_dates": position_valuation_dates,
        "equity": equity,
        "cash": portfolio["cash"],
        "invested": invested,
        "n_positions": len(portfolio["positions"]),
    }
    benchmark_status = {}
    for key, benchmark in (benchmarks or {}).items():
        benchmark_date = benchmark.get("bar_date") if isinstance(benchmark, dict) else None
        benchmark_status[key] = {
            "portfolio_bar_date": as_of_bar_date,
            "benchmark_bar_date": benchmark_date,
            "aligned": bool(as_of_bar_date and benchmark_date == as_of_bar_date),
        }
        if benchmark_status[key]["aligned"] and isinstance(benchmark.get("value"), (int, float)):
            snapshot[f"bench_{key}"] = benchmark["value"]
            snapshot[f"bench_{key}_bar_date"] = benchmark_date
    snapshot["benchmark_status"] = benchmark_status
    if record_snapshot:
        curve = portfolio["equity_curve"]
        if curve and curve[-1].get("date") == today:
            curve[-1] = snapshot
        else:
            curve.append(snapshot)
    return equity


def _max_drawdown(curve: list[dict[str, Any]]) -> float | None:
    values = [point.get("equity") for point in curve if isinstance(point.get("equity"), (int, float))]
    if not values:
        return None
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1 if peak else 0)
    return worst * 100


def update_portfolio(
    rows: list[dict[str, Any]],
    today: str | None = None,
    benchmarks: dict[str, dict[str, Any]] | None = None,
    *,
    action_data_allowed: bool = True,
    allow_orders: bool = True,
    observed_at: datetime | None = None,
    data_allowed: bool | None = None,
) -> dict[str, Any]:
    if data_allowed is not None:
        action_data_allowed = data_allowed
        allow_orders = data_allowed
    observed_at = observed_at or datetime.now(timezone.utc)
    today = today or _today()
    portfolio = load_portfolio(today)
    by_symbol = {row["symbol"]: row for row in rows}
    migration_blocked = bool(portfolio.get("migration_requires_review"))
    if action_data_allowed:
        _apply_corporate_actions(portfolio, by_symbol)
        _mark_positions(portfolio, by_symbol)
    if not migration_blocked:
        _execute_pending(
            portfolio,
            by_symbol,
            observed_at,
            allow_fills=bool(action_data_allowed and allow_orders),
        )
        if action_data_allowed:
            _mark_positions(portfolio, by_symbol)
        if allow_orders:
            _queue_signals(portfolio, rows, observed_at)
    equity = _mark_and_snapshot(
        portfolio,
        by_symbol,
        today,
        benchmarks,
        record_snapshot=action_data_allowed,
    )
    _save(portfolio)
    return {
        "schema_version": SCHEMA_VERSION,
        "simulation_status": portfolio.get("simulation_status", "unvalidated"),
        "performance_actionable": False,
        "equity": equity,
        "cash": portfolio["cash"],
        "invested": equity - portfolio["cash"],
        "n_positions": len(portfolio["positions"]),
        "pending_orders": len(portfolio["pending_orders"]),
        "ledger_entries": len(portfolio["ledger"]),
        "max_drawdown_pct": _max_drawdown(portfolio["equity_curve"]),
        "legacy_migrated": bool(portfolio.get("legacy_migrated")),
    }
