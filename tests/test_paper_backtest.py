from __future__ import annotations

import random
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import src.paper_trader as paper
from src.analyze import _paper_eligibility
from src.backtest import _average_ranks, _spearman, _tie_aware_bucket_ids
from src.persistence import atomic_write_json, load_json
from tests.helpers import ProjectTempMixin


def observed(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def research_row(
    bar_date,
    *,
    symbol="ABC",
    open_price=100.0,
    close_price=101.0,
    issuer_key=None,
    sector="Technology",
    country="us",
    currency="USD",
    actions=None,
    **extra,
):
    row = {
        "symbol": symbol,
        "name": f"{symbol} Corp",
        "asset_type": "company_equity",
        "bar_date": bar_date,
        "bar_timestamp": f"{bar_date}T00:00:00+00:00",
        "session_open_timestamp": f"{bar_date}T13:30:00+00:00",
        "raw_open_usd": open_price,
        "raw_close_usd": close_price,
        "avg_dollar_volume_20_usd": 50_000_000,
        "radar_score": 80,
        "daily_signal_direction": "POSITIVE",
        "feature_coverage": {"rank_eligible": True},
        "paper_eligibility": {"eligible": currency == "USD", "reasons": []},
        "currency": currency,
        "issuer_key": issuer_key or f"issuer:{symbol}",
        "sector": sector,
        "cc": country,
        "atr_pct": 2.0,
        "vol_annual_pct": 30.0,
        "corporate_actions": actions or [],
        "stock_split": 0.0,
        "dividend_usd": 0.0,
        **extra,
    }
    return row


class PaperBacktestTests(ProjectTempMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.original = paper.PORTFOLIO_FILE
        paper.PORTFOLIO_FILE = self.work / "portfolio.json"

    def tearDown(self):
        paper.PORTFOLIO_FILE = self.original
        super().tearDown()

    def test_production_schedule_never_fills_historical_open_before_creation(self):
        # At Tue 23:15 the newest completed signal bar is Monday. Tuesday's open
        # already happened and must never be used retroactively.
        created = observed("2026-08-11T23:15:00")
        paper.update_portfolio(
            [research_row("2026-08-10")],
            today="2026-08-11",
            observed_at=created,
        )
        order = load_json(paper.PORTFOLIO_FILE)["pending_orders"][0]
        self.assertEqual(order["not_before_bar_date"], "2026-08-11")

        paper.update_portfolio(
            [research_row("2026-08-11", open_price=105.0)],
            today="2026-08-12",
            observed_at=observed("2026-08-12T23:15:00"),
        )
        self.assertEqual(load_json(paper.PORTFOLIO_FILE)["positions"], {})

        paper.update_portfolio(
            [research_row("2026-08-12", open_price=106.0)],
            today="2026-08-13",
            observed_at=observed("2026-08-13T23:15:00"),
        )
        state = load_json(paper.PORTFOLIO_FILE)
        fill = next(entry for entry in state["ledger"] if entry.get("type") == "FILL")
        self.assertEqual(fill["fill_bar_date"], "2026-08-12")
        self.assertGreater(
            datetime.fromisoformat(fill["fill_session_open_timestamp"]),
            datetime.fromisoformat(fill["created_at"]),
        )
        self.assertEqual(fill["fill_observed_at"], "2026-08-13T23:15:00+00:00")

    def test_non_usd_instrument_is_labelled_and_never_queued(self):
        row = research_row("2026-08-10", symbol="SAP.DE", currency="EUR")
        row["paper_eligibility"] = _paper_eligibility(row)
        self.assertFalse(row["paper_eligibility"]["eligible"])
        self.assertTrue(any("point-in-time FX" in reason for reason in row["paper_eligibility"]["reasons"]))
        paper.update_portfolio(
            [row],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
        )
        self.assertEqual(load_json(paper.PORTFOLIO_FILE)["pending_orders"], [])

    def test_replays_skipped_actions_once_even_when_orders_blocked(self):
        paper.update_portfolio(
            [research_row("2026-08-10")],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
        )
        paper.update_portfolio(
            [research_row("2026-08-13")],
            today="2026-08-14",
            observed_at=observed("2026-08-14T23:15:00"),
        )
        before = load_json(paper.PORTFOLIO_FILE)
        quantity = before["positions"]["ABC"]["quantity"]
        cash = before["cash"]
        paper.update_portfolio(
            [research_row("2026-08-16")],
            today="2026-08-17",
            observed_at=observed("2026-08-17T23:15:00"),
            action_data_allowed=True,
            allow_orders=False,
        )
        action_history = [
            {
                "bar_date": "2026-08-14",
                "stock_split": 0.0,
                "dividend_usd": 1.0,
            },
            {
                "bar_date": "2026-08-15",
                "stock_split": 2.0,
                "dividend_usd": 0.0,
            },
        ]
        row = research_row(
            "2026-08-17",
            close_price=50.5,
            actions=action_history,
        )
        paper.update_portfolio(
            [row],
            today="2026-08-18",
            observed_at=observed("2026-08-18T23:15:00"),
            action_data_allowed=True,
            allow_orders=False,
        )
        after = load_json(paper.PORTFOLIO_FILE)
        self.assertAlmostEqual(after["positions"]["ABC"]["quantity"], quantity * 2)
        self.assertAlmostEqual(after["cash"], cash + quantity)
        action_entries = [
            entry for entry in after["ledger"] if entry["type"] in {"DIVIDEND", "SPLIT"}
        ]
        self.assertEqual(len(action_entries), 2)

        paper.update_portfolio(
            [row],
            today="2026-08-19",
            observed_at=observed("2026-08-19T23:15:00"),
            action_data_allowed=True,
            allow_orders=False,
        )
        replay = load_json(paper.PORTFOLIO_FILE)
        self.assertEqual(
            len([entry for entry in replay["ledger"] if entry["type"] in {"DIVIDEND", "SPLIT"}]),
            2,
        )
        corrected = research_row(
            "2026-08-20",
            close_price=50.5,
            actions=[
                {
                    "bar_date": "2026-08-14",
                    "stock_split": 0.0,
                    "dividend_usd": 1.5,
                },
                {
                    "bar_date": "2026-08-15",
                    "stock_split": 2.0,
                    "dividend_usd": 0.0,
                },
            ],
        )
        cash_before_correction = replay["cash"]
        paper.update_portfolio(
            [corrected],
            today="2026-08-21",
            observed_at=observed("2026-08-21T23:15:00"),
            action_data_allowed=True,
            allow_orders=False,
        )
        correction = load_json(paper.PORTFOLIO_FILE)
        self.assertAlmostEqual(
            correction["cash"],
            cash_before_correction + quantity * 0.5,
        )
        self.assertEqual(correction["ledger"][-1]["type"], "DIVIDEND_CORRECTION")

    def test_migration_review_blocks_existing_pending_and_new_orders(self):
        portfolio = paper._initial("2026-08-10")
        portfolio["migration_requires_review"] = True
        portfolio["pending_orders"] = [
            paper._order(
                "BUY",
                research_row("2026-08-01"),
                "legacy pending",
                observed("2026-08-02T23:15:00"),
                target_notional=1000,
            )
        ]
        atomic_write_json(paper.PORTFOLIO_FILE, portfolio)
        paper.update_portfolio(
            [research_row("2026-08-10")],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
        )
        state = load_json(paper.PORTFOLIO_FILE)
        self.assertEqual(state["positions"], {})
        self.assertEqual(len(state["pending_orders"]), 1)
        self.assertFalse(any(entry.get("type") == "FILL" for entry in state["ledger"]))

    def test_ten_for_one_split_preserves_equity_and_order_sizing(self):
        portfolio = paper._initial("2026-08-01")
        portfolio["cash"] = 9000.0
        portfolio["positions"]["ABC"] = {
            "symbol": "ABC",
            "quantity": 10.0,
            "entry_price": 100.0,
            "cost_basis": 1000.0,
            "last_price": 100.0,
            "last_action_bar_date": "2026-08-10",
            "legacy": False,
            "issuer_key": "issuer:ABC",
            "sector": "Industrials",
            "country": "us",
        }
        atomic_write_json(paper.PORTFOLIO_FILE, portfolio)
        split_row = research_row(
            "2026-08-12",
            close_price=10.0,
            actions=[
                {
                    "bar_date": "2026-08-11",
                    "stock_split": 10.0,
                    "dividend_usd": 0.0,
                }
            ],
        )
        candidate = research_row(
            "2026-08-12",
            symbol="XYZ",
            sector="Healthcare",
            country="ca",
        )
        result = paper.update_portfolio(
            [split_row, candidate],
            today="2026-08-13",
            observed_at=observed("2026-08-13T23:15:00"),
        )
        state = load_json(paper.PORTFOLIO_FILE)
        self.assertEqual(state["positions"]["ABC"]["quantity"], 100.0)
        self.assertEqual(state["positions"]["ABC"]["last_price"], 10.0)
        self.assertAlmostEqual(result["equity"], 10_000.0)
        xyz = next(order for order in state["pending_orders"] if order["symbol"] == "XYZ")
        self.assertAlmostEqual(xyz["target_notional"], 1000.0)

    def test_missing_symbol_order_expires_before_row_checks(self):
        portfolio = paper._initial("2026-08-01")
        portfolio["pending_orders"] = [
            paper._order(
                "BUY",
                research_row("2026-08-01"),
                "pending",
                observed("2026-08-01T23:15:00"),
                target_notional=1000,
            )
        ]
        atomic_write_json(paper.PORTFOLIO_FILE, portfolio)
        paper.update_portfolio(
            [],
            today="2026-08-10",
            observed_at=observed("2026-08-10T23:15:00"),
            action_data_allowed=False,
            allow_orders=False,
        )
        state = load_json(paper.PORTFOLIO_FILE)
        self.assertEqual(state["pending_orders"], [])
        self.assertEqual(state["ledger"][-1]["type"], "ORDER_CANCELLED")

    def test_concentrated_candidate_pool_respects_caps_and_issuer_uniqueness(self):
        rows = [
            research_row(
                "2026-08-10",
                symbol=f"T{index}",
                issuer_key=("issuer:duplicate" if index < 2 else f"issuer:{index}"),
                sector="Technology",
                country="us",
            )
            for index in range(8)
        ]
        paper.update_portfolio(
            rows,
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
        )
        orders = load_json(paper.PORTFOLIO_FILE)["pending_orders"]
        self.assertLessEqual(len(orders), paper.MAX_PER_SECTOR)
        self.assertEqual(
            len({order["issuer_key"] for order in orders}),
            len(orders),
        )

        paper.PORTFOLIO_FILE.unlink()
        country_rows = [
            research_row(
                "2026-08-10",
                symbol=f"C{index}",
                sector=f"Sector {index}",
                country="us",
            )
            for index in range(8)
        ]
        paper.update_portfolio(
            country_rows,
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
        )
        self.assertLessEqual(
            len(load_json(paper.PORTFOLIO_FILE)["pending_orders"]),
            paper.MAX_PER_COUNTRY,
        )

    def test_entries_are_restricted_to_explicit_ideal_symbols(self):
        rows = [
            research_row("2026-08-10", symbol="IDEAL"),
            research_row("2026-08-10", symbol="RANKED"),
        ]
        paper.update_portfolio(
            rows,
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
            entry_symbols={"IDEAL"},
            entry_theses={
                "IDEAL": {
                    "sentences": [
                        "Deutlich unter der fairen Grenze.",
                        "Technischer Einstieg bestätigt.",
                    ]
                }
            },
        )
        orders = load_json(paper.PORTFOLIO_FILE)["pending_orders"]
        self.assertEqual([order["symbol"] for order in orders], ["IDEAL"])
        self.assertEqual(
            orders[0]["thesis"]["sentences"],
            [
                "Deutlich unter der fairen Grenze.",
                "Technischer Einstieg bestätigt.",
            ],
        )

    def test_hard_stop_and_trailing_profit_queue_deterministic_exits(self):
        for symbol, close, peak, expected_trigger in (
            ("STOP", 89.0, 100.0, "hard_stop"),
            ("TRAIL", 110.0, 120.0, "trailing_stop"),
        ):
            portfolio = paper._initial("2026-08-01")
            portfolio["cash"] = 9000.0
            portfolio["positions"][symbol] = {
                "symbol": symbol,
                "name": symbol,
                "quantity": 10.0,
                "entry_price": 100.0,
                "cost_basis": 1000.0,
                "last_price": peak,
                "high_watermark": peak,
                "entry_bar_date": "2026-08-01",
                "last_action_bar_date": "2026-08-01",
                "legacy": False,
            }
            atomic_write_json(paper.PORTFOLIO_FILE, portfolio)
            paper.update_portfolio(
                [research_row("2026-08-10", symbol=symbol, close_price=close)],
                today="2026-08-11",
                observed_at=observed("2026-08-11T23:15:00"),
                entry_symbols=set(),
            )
            order = load_json(paper.PORTFOLIO_FILE)["pending_orders"][0]
            self.assertEqual(order["action"], "SELL")
            self.assertEqual(order["exit_trigger"], expected_trigger)
            paper.PORTFOLIO_FILE.unlink()

    def test_entry_gate_does_not_block_risk_exit(self):
        portfolio = paper._initial("2026-08-01")
        portfolio["cash"] = 9000.0
        portfolio["positions"]["STOP"] = {
            "symbol": "STOP",
            "name": "STOP",
            "quantity": 10.0,
            "entry_price": 100.0,
            "cost_basis": 1000.0,
            "last_price": 100.0,
            "high_watermark": 100.0,
            "entry_bar_date": "2026-08-01",
            "last_action_bar_date": "2026-08-01",
            "legacy": False,
        }
        atomic_write_json(paper.PORTFOLIO_FILE, portfolio)
        paper.update_portfolio(
            [research_row("2026-08-10", symbol="STOP", close_price=89.0)],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
            allow_entries=False,
            entry_symbols=set(),
        )
        order = load_json(paper.PORTFOLIO_FILE)["pending_orders"][0]
        self.assertEqual(order["action"], "SELL")
        self.assertEqual(order["exit_trigger"], "hard_stop")

    def test_pending_buy_is_cancelled_when_ideal_thesis_disappears(self):
        paper.update_portfolio(
            [research_row("2026-08-10")],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
            entry_symbols={"ABC"},
        )
        paper.update_portfolio(
            [research_row("2026-08-12")],
            today="2026-08-13",
            observed_at=observed("2026-08-13T23:15:00"),
            entry_symbols=set(),
        )
        state = load_json(paper.PORTFOLIO_FILE)
        self.assertEqual(state["positions"], {})
        self.assertEqual(state["pending_orders"], [])
        self.assertEqual(state["ledger"][-1]["type"], "ORDER_CANCELLED")
        self.assertIn("no longer holds", state["ledger"][-1]["cancel_reason"])

    def test_eur_fx_is_applied_to_fills_and_marks(self):
        paper.update_portfolio(
            [research_row("2026-08-10")],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
        )
        result = paper.update_portfolio(
            [research_row("2026-08-13", open_price=100.0, close_price=110.0)],
            today="2026-08-14",
            observed_at=observed("2026-08-14T23:15:00"),
            base_fx_bars={"2026-08-13": {"open": 2.0, "close": 2.0}},
            entry_symbols={"ABC"},
        )
        state = load_json(paper.PORTFOLIO_FILE)
        self.assertEqual(state["base_currency"], "EUR")
        self.assertAlmostEqual(state["positions"]["ABC"]["entry_price"], 50.05)
        self.assertAlmostEqual(state["positions"]["ABC"]["last_price"], 55.0)
        self.assertEqual(result["base_currency"], "EUR")

    def test_benchmark_values_require_common_completed_bar_date(self):
        paper.update_portfolio(
            [research_row("2026-08-10")],
            today="2026-08-11",
            observed_at=observed("2026-08-11T23:15:00"),
            allow_orders=False,
            benchmarks={"sp500": {"value": 6000.0, "bar_date": "2026-08-09"}},
        )
        first = load_json(paper.PORTFOLIO_FILE)["equity_curve"][-1]
        self.assertNotIn("bench_sp500", first)
        self.assertFalse(first["benchmark_status"]["sp500"]["aligned"])

        paper.update_portfolio(
            [research_row("2026-08-11")],
            today="2026-08-12",
            observed_at=observed("2026-08-12T23:15:00"),
            allow_orders=False,
            benchmarks={"sp500": {"value": 6010.0, "bar_date": "2026-08-11"}},
        )
        second = load_json(paper.PORTFOLIO_FILE)["equity_curve"][-1]
        self.assertEqual(second["bench_sp500_bar_date"], second["as_of_bar_date"])

    def test_mixed_position_valuation_dates_block_benchmark_point(self):
        portfolio = paper._initial("2026-08-01")
        portfolio["cash"] = 0.0
        for symbol in ("AAA", "BBB"):
            portfolio["positions"][symbol] = {
                "symbol": symbol,
                "quantity": 10.0,
                "entry_price": 100.0,
                "cost_basis": 1000.0,
                "last_price": 100.0,
                "entry_bar_date": "2026-08-01",
                "last_action_bar_date": "2026-08-01",
                "legacy": False,
            }
        atomic_write_json(paper.PORTFOLIO_FILE, portfolio)
        paper.update_portfolio(
            [
                research_row("2026-08-10", symbol="AAA"),
                research_row("2026-08-11", symbol="BBB"),
            ],
            today="2026-08-12",
            observed_at=observed("2026-08-12T23:15:00"),
            allow_orders=False,
            benchmarks={"sp500": {"value": 6000.0, "bar_date": "2026-08-10"}},
        )
        point = load_json(paper.PORTFOLIO_FILE)["equity_curve"][-1]
        self.assertEqual(point["valuation_status"], "mixed_or_stale")
        self.assertIsNone(point["as_of_bar_date"])
        self.assertNotIn("bench_sp500", point)

    def test_legacy_portfolio_is_preserved_and_explicit_start_is_required(self):
        legacy = {
            "created": "2026-01-01",
            "cash": 500.0,
            "positions": {
                "ABC": {
                    "name": "ABC",
                    "entry_price": 10.0,
                    "stake_eur": 100.0,
                    "last_price": 11.0,
                    "entry_date": "2026-01-02",
                }
            },
        }
        atomic_write_json(paper.PORTFOLIO_FILE, legacy)
        migrated = paper.load_portfolio("2026-08-12")
        self.assertTrue(migrated["migration_requires_review"])
        self.assertEqual(migrated["legacy_archive"], legacy)
        with patch.dict("os.environ", {"STOCK_RADAR_START_NEW_PAPER": "1"}):
            started = paper.load_portfolio("2026-08-12")
        self.assertEqual(started["cash"], paper.START_CAPITAL)
        self.assertEqual(started["base_currency"], "EUR")
        self.assertEqual(started["positions"], {})
        self.assertIn("ABC", started["legacy_frozen_positions"])

    def test_tie_ranks_and_buckets_are_order_invariant(self):
        self.assertEqual(_average_ranks([1, 1, 3]), [0.5, 0.5, 2.0])
        self.assertAlmostEqual(_spearman([1, 1, 3, 4, 5], [2, 2, 6, 8, 10]), 1.0)
        scores = [50.0] * 25
        baseline = _tie_aware_bucket_ids(scores, 5)
        shuffled = list(scores)
        random.Random(42).shuffle(shuffled)
        self.assertEqual(set(baseline), set(_tie_aware_bucket_ids(shuffled, 5)))
        self.assertEqual(len(set(baseline)), 1)
        self.assertNotIn(0, set(baseline))
        self.assertNotIn(4, set(baseline))
        buckets = {bucket: [] for bucket in range(5)}
        for bucket, value in zip(baseline, range(25)):
            buckets[bucket].append(value)
        spread = (
            sum(buckets[4]) / len(buckets[4]) - sum(buckets[0]) / len(buckets[0])
            if buckets[4] and buckets[0]
            else None
        )
        self.assertIsNone(spread)


if __name__ == "__main__":
    unittest.main()
