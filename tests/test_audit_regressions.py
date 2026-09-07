import copy
import ast
import inspect
import json
import subprocess
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from src import paper_trader as paper
from src import analyze
from src.fetch import fetch_prices_with_status
from src.freshness import build_session_freshness, evaluate_session_freshness
from src.question_views import build_question_views
from src.question_views import decision_overlay
from src import opportunity_history
from tests.helpers import ProjectTempMixin, ROOT
from tests.test_paper_backtest import research_row, observed
from tests.test_question_views import row


class AuditRegressionTests(unittest.TestCase):
    def test_pipeline_dates_orders_after_inputs_have_been_collected(self):
        tree = ast.parse(inspect.getsource(analyze.run))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "update_portfolio"]
        self.assertEqual(len(calls), 1)
        timestamp = next(arg.value for arg in calls[0].keywords if arg.arg == "observed_at")
        self.assertEqual(ast.unparse(timestamp), "datetime.now(timezone.utc)")
        # A collection spanning the open must target tomorrow, not that past open.
        order = paper._order("BUY", research_row("2026-09-08"), "collected",
                             observed("2026-09-09T13:35:00"), target_notional=1000)
        self.assertEqual(order["expected_fill_bar_date"], "2026-09-10")

    def test_usd_foreign_listing_cannot_use_us_execution_calendar(self):
        state = paper._initial("2026-09-02")
        candidate = research_row("2026-09-02", symbol="ABC.L", currency="USD")
        paper._queue_signals(state, [candidate], observed("2026-09-02T22:00:00"))
        self.assertEqual(state["pending_orders"], [])

    def test_legacy_benchmark_points_are_not_eur_comparison_observations(self):
        point = {"as_of_bar_date": "2026-09-04", "bench_sp500_bar_date": "2026-09-04",
                 "benchmark_status": {"sp500": {"aligned": True}}}
        self.assertFalse(paper.eur_benchmark_observation(point, "sp500"))
        point["benchmark_status"]["sp500"]["currency"] = "USD"
        self.assertFalse(paper.eur_benchmark_observation(point, "sp500"))
        point["benchmark_status"]["sp500"]["currency"] = "EUR"
        self.assertTrue(paper.eur_benchmark_observation(point, "sp500"))

    def test_today_overlay_cannot_promote_a_failed_discount_filter_to_ideal(self):
        candidate = row(price=86, lower=100, upper=120)
        with patch("src.question_views.traffic_light", return_value="green"):
            self.assertNotEqual(decision_overlay(candidate)["situation"]["code"], "ideal")

    def test_legacy_history_does_not_count_toward_new_reference_span(self):
        legacy = {"snapshots": [{"date": "2026-07-01", "scores": [99] * 100}]}
        current = {"cheap_with_potential": [{"deal_quality": {"score": 60}}] * 100}
        with patch.object(opportunity_history, "load_json", return_value=legacy), patch.object(opportunity_history, "atomic_write_json"):
            result = opportunity_history.update_opportunity_history(
                current, observed("2026-09-07T12:00:00"))
        self.assertEqual(result["calendar_days"], 1)
        self.assertEqual(result["from_date"], "2026-09-07")
        self.assertFalse(result["reference_ready"])
        self.assertFalse(result["reliable"])

    def test_discount_and_upside_are_distinct_and_threshold_is_true_discount(self):
        result = build_question_views([row("BR", price=178.91, lower=265.3709, upper=280)])
        item = result["cheap_with_potential"][0]
        self.assertEqual(item["discount_pct"], 32.6)
        self.assertEqual(item["upside_to_fair_lower_pct"], 48.3)
        result = build_question_views([
            row("BELOW", price=86, lower=100, upper=120),
            row("PASS", price=85, lower=100, upper=120),
        ])
        self.assertEqual([item["symbol"] for item in result["cheap_with_potential"]], ["PASS"])

    def test_neutral_split_preserves_peak_and_does_not_create_exit(self):
        state = paper._initial("2026-09-01")
        state["positions"]["ABC"] = dict(
            symbol="ABC", quantity=10, entry_price=100, cost_basis=1000,
            last_price=100, high_watermark=100, entry_bar_date="2026-09-01",
        )
        current = research_row("2026-09-03", close_price=50,
                               actions=[dict(bar_date="2026-09-03", stock_split=2)])
        paper._apply_corporate_actions(state, {"ABC": current}, None)
        paper._mark_positions(state, {"ABC": current})
        paper._queue_signals(state, [current], observed("2026-09-03T22:00:00"), allow_entries=False)
        self.assertEqual(state["positions"]["ABC"]["high_watermark"], 50)
        self.assertEqual(state["pending_orders"], [])
        paper._apply_corporate_actions(state, {"ABC": current}, None)
        self.assertEqual(state["positions"]["ABC"]["high_watermark"], 50)

    def test_missing_execution_bar_cancels_instead_of_fabricating_later_fill(self):
        state = paper._initial("2026-09-02")
        paper._queue_order(state, paper._order(
            "BUY", research_row("2026-09-02"), "ideal",
            observed("2026-09-02T22:00:00"), target_notional=1000,
        ))
        created = copy.deepcopy(state["ledger"][0])
        paper._execute_pending(state, {"ABC": research_row("2026-09-04")},
                               observed("2026-09-04T22:00:00"))
        self.assertEqual(state["positions"], {})
        self.assertEqual(state["ledger"][0], created)
        self.assertEqual([event["type"] for event in state["ledger"]],
                         ["ORDER_CREATED", "ORDER_CANCELLED"])
        self.assertIn("missed execution", state["ledger"][-1]["cancel_reason"])

    def test_fx_requires_preceding_date_and_ignores_same_day_open(self):
        self.assertEqual(paper._execution_fx({"2026-09-03": {"open": 1.2, "close": 1.3}},
                                            "2026-09-03"), (None, None))
        self.assertEqual(paper._execution_fx({"2026-09-02": {"close": 1.1},
                                             "2026-09-03": {"open": 9.0}}, "2026-09-03"),
                         (1.1, "2026-09-02"))

    def test_stale_tokyo_and_crypto_do_not_hide_current_us_research(self):
        rows = [{"symbol": "AAPL", "bar_date": "2026-09-04"},
                {"symbol": "7203.T", "bar_date": "2026-09-04"},
                {"symbol": "BTC-USD", "bar_date": "2026-09-04"}]
        contract = build_session_freshness(rows)
        now = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
        result = evaluate_session_freshness(contract, now=now)
        self.assertTrue(result["blocking_reasons"])
        self.assertEqual(result["research_blocking_reasons"], [])
        self.assertEqual(result["fresh_symbols"], ["AAPL"])
        data = {"generated_at": "2026-09-05T01:00:00Z", "instruments": rows,
                "data_status": {"session_freshness": contract}}
        code = """const fs=require('fs'); const {sessionFreshness}=require('./docs/freshness.js');
const r=sessionFreshness(JSON.parse(fs.readFileSync(0,'utf8')),Date.parse('2026-09-07T12:00:00Z'));
console.log(JSON.stringify({symbols:r.freshSymbols,failures:r.researchFailures}));"""
        r = subprocess.run(["node", "-e", code], input=json.dumps(data),
                           text=True, capture_output=True, cwd=ROOT, check=True)
        self.assertEqual(json.loads(r.stdout), {"symbols": ["AAPL"], "failures": []})

    def test_nonempty_stale_yahoo_response_is_retried_for_expected_session(self):
        index = pd.bdate_range(end="2026-09-04", periods=40)
        full = pd.DataFrame({key: 100.0 for key in ("Open", "High", "Low", "Close")}, index=index)
        full["Volume"] = 10000
        responses = iter([full.iloc[:-1], full])
        calls = []
        def download(symbols, period):
            calls.append(symbols)
            return next(responses)
        result = fetch_prices_with_status(["AAPL"], downloader=download,
                                          now=observed("2026-09-07T12:00:00"), verbose=False)
        self.assertEqual(calls, [["AAPL"], ["AAPL"]])
        self.assertEqual(result.bar_info["AAPL"]["bar_date"], "2026-09-04")
        self.assertFalse(result.bar_info["AAPL"]["missing_latest_session"])


class AuditMigrationTests(ProjectTempMixin, unittest.TestCase):
    def test_upgrade_preserves_historic_asb_and_cancels_legacy_pending_explicitly(self):
        state = paper._initial("2026-09-03")
        state.pop("strategy_version")
        historic = {"type": "ORDER_CANCELLED", "symbol": "ASB", "action": "BUY",
                    "cancel_reason": "strict ideal entry thesis no longer holds"}
        state["ledger"] = [copy.deepcopy(historic)]
        state["pending_orders"] = [paper._order(
            "BUY", research_row("2026-09-03"), "legacy",
            observed("2026-09-05T01:00:00"), target_notional=1000,
        )]
        with patch.object(paper, "load_portfolio", return_value=state), patch.object(paper, "_save"):
            paper.update_portfolio([], observed_at=observed("2026-09-07T12:00:00"),
                                   allow_orders=False, action_data_allowed=False)
        self.assertEqual(state["ledger"][0], historic)
        self.assertEqual(state["ledger"][1]["type"], "STRATEGY_VERSION_CHANGED")
        self.assertEqual(state["ledger"][2]["type"], "ORDER_CANCELLED")
        self.assertEqual(state["pending_orders"], [])


if __name__ == "__main__":
    unittest.main()
