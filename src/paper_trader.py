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
HARD_STOP_LOSS_PCT = -10.0
TRAILING_ACTIVATION_PCT = 12.0
TRAILING_STOP_PCT = 8.0
TAKE_PROFIT_PCT = 30.0
MAX_HOLDING_DAYS = 180
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
        "base_currency": "EUR",
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
            "hard_stop_loss_pct": HARD_STOP_LOSS_PCT,
            "trailing_activation_pct": TRAILING_ACTIVATION_PCT,
            "trailing_stop_pct": TRAILING_STOP_PCT,
            "take_profit_pct": TAKE_PROFIT_PCT,
            "max_holding_days": MAX_HOLDING_DAYS,
            "min_avg_dollar_volume": MIN_AVG_DOLLAR_VOLUME,
            "max_atr_pct": MAX_ATR_PCT,
            "max_annual_vol_pct": MAX_ANNUAL_VOL_PCT,
            "max_per_sector": MAX_PER_SECTOR,
            "max_per_country": MAX_PER_COUNTRY,
            "base_currency": "EUR",
            "fx_accounting": "daily USD-per-EUR rate stored with every fill and mark",
            "diversification": "issuer uniqueness plus sector/country caps; no correlation claim",
            "order_not_before": (
                "first completed bar whose session open is after order creation"
            ),
            "corporate_actions": (
                "best effort from Yahoo daily action columns; missed-run gaps may remain incomplete"
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


def _start_clean_v2_after_review(portfolio: dict[str, Any], today: str) -> dict[str, Any]:
    legacy_archive = copy.deepcopy(portfolio)
    clean = _initial(today)
    clean["legacy_archive"] = legacy_archive
    clean["legacy_frozen_positions"] = legacy_archive.get("positions") or {}
    clean["legacy_migrated"] = True
    clean["migration_requires_review"] = False
    clean["clean_v2_started_at"] = utc_now()
    clean["ledger"].append(
        {
            "transaction_id": str(uuid.uuid4()),
            "type": "CLEAN_V2_SIMULATION_STARTED",
            "timestamp": utc_now(),
            "note": "Explicit STOCK_RADAR_START_NEW_PAPER=1 decision; legacy data preserved",
        }
    )
    return clean


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
            return _start_clean_v2_after_review(raw, today)
        return raw
    migrated = _migrate_legacy(raw, today)
    if os.environ.get("STOCK_RADAR_START_NEW_PAPER") == "1":
        return _start_clean_v2_after_review(migrated, today)
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
        "not_before_bar_date": created.date().isoformat(),
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


def _apply_corporate_actions(
    portfolio: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    base_fx_bars: dict[str, dict[str, float]] | None,
) -> None:
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
                fx = (
                    (base_fx_bars.get(bar_date) or {}).get("close")
                    if base_fx_bars is not None
                    else 1.0
                )
                if not isinstance(fx, (int, float)) or fx <= 0:
                    continue
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
                amount = quantity_basis * delta / fx
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
                        "base_fx_usd": fx,
                        "fx_bar_date": bar_date,
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
    allow_buy_fills: bool = True,
    base_fx_bars: dict[str, dict[str, float]] | None = None,
    eligible_buy_symbols: set[str] | None = None,
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
        if order["action"] == "BUY" and not allow_buy_fills:
            remaining.append(order)
            continue
        if (
            order["action"] == "BUY"
            and eligible_buy_symbols is not None
            and order["symbol"] not in eligible_buy_symbols
        ):
            order["status"] = "cancelled"
            order["cancelled_at"] = observed_at.isoformat(timespec="seconds")
            order["cancel_reason"] = "strict ideal entry thesis no longer holds"
            portfolio["ledger"].append({**order, "type": "ORDER_CANCELLED"})
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
        base_fx_usd = (
            (base_fx_bars.get(bar_date) or {}).get("open")
            if base_fx_bars is not None
            else 1.0
        )
        if not isinstance(base_fx_usd, (int, float)) or base_fx_usd <= 0:
            order["last_error"] = "later bar has no aligned EURUSD open"
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
        raw_fill_price = open_price / base_fx_usd
        execution_price = raw_fill_price * (
            1 + SLIPPAGE_BPS / 10_000 if action == "BUY" else 1 - SLIPPAGE_BPS / 10_000
        )
        transaction_thesis = copy.deepcopy(order.get("thesis") or {})
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
                "last_price": (
                    row.get("raw_close_usd") / base_fx_usd
                    if isinstance(row.get("raw_close_usd"), (int, float))
                    else execution_price
                ),
                "high_watermark": (
                    row.get("raw_close_usd") / base_fx_usd
                    if isinstance(row.get("raw_close_usd"), (int, float))
                    else execution_price
                ),
                "last_mark_bar_date": bar_date,
                "last_action_bar_date": bar_date,
                "processed_actions": {},
                "legacy": False,
                "issuer_key": order.get("issuer_key"),
                "sector": order.get("sector"),
                "country": order.get("country"),
                "instrument_currency": "USD",
                "base_currency": "EUR",
                "entry_thesis": copy.deepcopy(order.get("thesis") or {}),
            }
        else:
            position = portfolio["positions"].get(order["symbol"])
            if not position:
                continue
            transaction_thesis = copy.deepcopy(
                position.get("entry_thesis") or {}
            )
            quantity = min(float(order.get("quantity") or 0), position["quantity"])
            allocated_cost = position.get("cost_basis", 0.0) * (
                quantity / position["quantity"]
            )
            gross = quantity * execution_price
            commission = gross * COMMISSION_BPS / 10_000
            realized_pnl = gross - commission - allocated_cost
            portfolio["cash"] += gross - commission
            position["quantity"] -= quantity
            position["cost_basis"] -= allocated_cost
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
                "raw_fill_price": raw_fill_price,
                "raw_fill_price_usd": open_price,
                "execution_price": execution_price,
                "gross_value": gross,
                "commission": commission,
                "slippage_bps": SLIPPAGE_BPS,
                "base_fx_usd": base_fx_usd,
                "reason": order.get("reason"),
                "exit_trigger": order.get("exit_trigger"),
                "thesis": transaction_thesis,
                "signal_return_pct": order.get("return_pct"),
                "signal_drawdown_from_peak_pct": order.get(
                    "drawdown_from_peak_pct"
                ),
                "realized_pnl": realized_pnl if action == "SELL" else None,
                "timestamp": utc_now(),
            }
        )
    portfolio["pending_orders"] = remaining


def _queue_signals(
    portfolio: dict[str, Any],
    rows: list[dict[str, Any]],
    observed_at: datetime,
    entry_symbols: set[str] | None = None,
    entry_theses: dict[str, dict[str, Any]] | None = None,
    allow_entries: bool = True,
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
        price = position.get("last_price")
        entry = position.get("entry_price")
        peak = position.get("high_watermark") or price
        return_pct = (
            (price / entry - 1.0) * 100.0
            if isinstance(price, (int, float))
            and isinstance(entry, (int, float))
            and entry > 0
            else None
        )
        drawdown_from_peak_pct = (
            (price / peak - 1.0) * 100.0
            if isinstance(price, (int, float))
            and isinstance(peak, (int, float))
            and peak > 0
            else None
        )
        peak_return_pct = (
            (peak / entry - 1.0) * 100.0
            if isinstance(peak, (int, float))
            and isinstance(entry, (int, float))
            and entry > 0
            else None
        )
        try:
            holding_days = (
                datetime.fromisoformat(row["bar_date"]).date()
                - datetime.fromisoformat(position["entry_bar_date"]).date()
            ).days
        except (KeyError, TypeError, ValueError):
            holding_days = None
        reason = None
        trigger = None
        if return_pct is not None and return_pct <= HARD_STOP_LOSS_PCT:
            trigger, reason = "hard_stop", "Hard-Stop bei minus 10 Prozent erreicht"
        elif (
            peak_return_pct is not None
            and peak_return_pct >= TRAILING_ACTIVATION_PCT
            and drawdown_from_peak_pct is not None
            and drawdown_from_peak_pct <= -TRAILING_STOP_PCT
        ):
            trigger, reason = "trailing_stop", "Gewinnsicherung vom Kurshoch ausgelöst"
        elif return_pct is not None and return_pct >= TAKE_PROFIT_PCT:
            trigger, reason = "take_profit", "Gewinnziel von 30 Prozent erreicht"
        elif (isinstance(score, (int, float)) and score < SELL_SCORE) or direction == "NEGATIVE":
            trigger, reason = "signal_break", "Kernsignal ist klar gebrochen"
        elif holding_days is not None and holding_days >= MAX_HOLDING_DAYS:
            trigger, reason = "time_exit", "Maximale Haltedauer von 180 Tagen erreicht"
        if reason:
            portfolio["pending_orders"].append(
                _order(
                    "SELL",
                    row,
                    reason,
                    observed_at,
                    quantity=position["quantity"],
                    exit_trigger=trigger,
                    return_pct=round(return_pct, 2) if return_pct is not None else None,
                    drawdown_from_peak_pct=(
                        round(drawdown_from_peak_pct, 2)
                        if drawdown_from_peak_pct is not None
                        else None
                    ),
                )
            )
            pending_symbols.add(symbol)

    open_or_pending = len(portfolio["positions"]) + sum(
        order["action"] == "BUY" for order in portfolio["pending_orders"]
    )
    slots = max(0, TARGET_POSITIONS - open_or_pending)
    if not slots or not allow_entries:
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
            and (entry_symbols is None or row["symbol"] in entry_symbols)
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
                "Strikter Fundamental-plus-Timing-Idealfall",
                observed_at,
                target_notional=target,
                issuer_key=issuer_key,
                sector=sector,
                country=country,
                currency="USD",
                thesis=copy.deepcopy((entry_theses or {}).get(row["symbol"]) or {}),
            )
        )
        issuer_keys.add(issuer_key)
        sector_counts[sector] += 1
        country_counts[country] += 1
        queued += 1


def _mark_positions(
    portfolio: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    base_fx_bars: dict[str, dict[str, float]] | None = None,
) -> tuple[float, dict[str, str | None]]:
    invested = 0.0
    valuation_dates: dict[str, str | None] = {}
    for symbol, position in portfolio["positions"].items():
        row = rows.get(symbol)
        bar_date = row.get("bar_date") if row else None
        base_fx_usd = (
            (base_fx_bars.get(bar_date) or {}).get("close")
            if base_fx_bars is not None and bar_date
            else 1.0 if base_fx_bars is None else None
        )
        price = (
            position.get("last_price")
            if position.get("legacy")
            else (
                row.get("raw_close_usd") / base_fx_usd
                if row
                and isinstance(row.get("raw_close_usd"), (int, float))
                and isinstance(base_fx_usd, (int, float))
                and base_fx_usd > 0
                else position.get("last_price")
            )
        )
        if isinstance(price, (int, float)) and price > 0:
            position["last_price"] = price
            position["high_watermark"] = max(
                float(position.get("high_watermark") or price),
                price,
            )
            if isinstance(base_fx_usd, (int, float)) and base_fx_usd > 0:
                position["last_base_fx_usd"] = base_fx_usd
            if (
                row
                and not position.get("legacy")
                and isinstance(base_fx_usd, (int, float))
                and base_fx_usd > 0
            ):
                position["last_mark_bar_date"] = row.get("bar_date")
        valuation_dates[symbol] = (
            row.get("bar_date")
            if row
            and not position.get("legacy")
            and row.get("bar_date")
            and isinstance(base_fx_usd, (int, float))
            and base_fx_usd > 0
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
    base_fx_bars: dict[str, dict[str, float]] | None = None,
) -> float:
    invested, position_valuation_dates = _mark_positions(
        portfolio, rows, base_fx_bars
    )
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
    entry_symbols: set[str] | None = None,
    entry_theses: dict[str, dict[str, Any]] | None = None,
    allow_entries: bool | None = None,
    base_fx_bars: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    if data_allowed is not None:
        action_data_allowed = data_allowed
        allow_orders = data_allowed
    if allow_entries is None:
        allow_entries = allow_orders
    observed_at = observed_at or datetime.now(timezone.utc)
    today = today or _today()
    portfolio = load_portfolio(today)
    by_symbol = {row["symbol"]: row for row in rows}
    migration_blocked = bool(portfolio.get("migration_requires_review"))
    if action_data_allowed:
        _apply_corporate_actions(portfolio, by_symbol, base_fx_bars)
        _mark_positions(portfolio, by_symbol, base_fx_bars)
    if not migration_blocked:
        _execute_pending(
            portfolio,
            by_symbol,
            observed_at,
            allow_fills=bool(action_data_allowed and allow_orders),
            allow_buy_fills=bool(allow_entries),
            base_fx_bars=base_fx_bars,
            eligible_buy_symbols=entry_symbols,
        )
        if action_data_allowed:
            _mark_positions(portfolio, by_symbol, base_fx_bars)
        if allow_orders:
            _queue_signals(
                portfolio,
                rows,
                observed_at,
                entry_symbols=entry_symbols,
                entry_theses=entry_theses,
                allow_entries=bool(allow_entries),
            )
    equity = _mark_and_snapshot(
        portfolio,
        by_symbol,
        today,
        benchmarks,
        record_snapshot=action_data_allowed,
        base_fx_bars=base_fx_bars,
    )
    _save(portfolio)
    positions = []
    for position in portfolio["positions"].values():
        value = position["quantity"] * position["last_price"]
        cost = position.get("cost_basis") or value
        positions.append(
            {
                "symbol": position["symbol"],
                "name": position.get("name"),
                "quantity": position["quantity"],
                "entry_price": position["entry_price"],
                "last_price": position["last_price"],
                "value": value,
                "pnl": value - cost,
                "pnl_pct": (value / cost - 1.0) * 100.0 if cost else None,
                "entry_bar_date": position.get("entry_bar_date"),
            }
        )
    fills = [
        entry for entry in portfolio["ledger"] if entry.get("type") == "FILL"
    ]
    realized_pnl = sum(
        float(entry.get("realized_pnl") or 0.0) for entry in fills
    )
    commissions = sum(
        float(entry.get("commission") or 0.0) for entry in fills
    )
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
        "total_return_pct": (
            (equity / portfolio["starting_cash"] - 1.0) * 100.0
            if portfolio.get("starting_cash")
            else None
        ),
        "realized_pnl": realized_pnl,
        "commissions": commissions,
        "legacy_migrated": bool(portfolio.get("legacy_migrated")),
        "migration_requires_review": bool(
            portfolio.get("migration_requires_review")
        ),
        "base_currency": portfolio.get("base_currency", "EUR"),
        "starting_cash": portfolio.get("starting_cash", START_CAPITAL),
        "positions": sorted(positions, key=lambda item: item["symbol"]),
        "orders": copy.deepcopy(portfolio["pending_orders"]),
        "recent_activity": copy.deepcopy(portfolio["ledger"][-20:]),
        "activity": copy.deepcopy(portfolio["ledger"]),
        "equity_curve": copy.deepcopy(portfolio["equity_curve"]),
        "strategy": copy.deepcopy(portfolio.get("assumptions") or {}),
    }
