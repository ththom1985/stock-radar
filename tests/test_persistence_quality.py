from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.data_quality import (
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    DataContractError,
    build_data_status,
    dashboard_gate,
    validate_output_contract,
    validate_portfolio_contract,
)
from src.persistence import (
    SCHEMA_VERSION,
    CorruptDataError,
    atomic_write_json,
    load_json,
)
from src.insights import enrich_row
from tests.helpers import ProjectTempMixin


class PersistenceQualityTests(ProjectTempMixin, unittest.TestCase):
    @staticmethod
    def _empty_insights(enabled=True):
        categories = {}
        for key in (
            "in_sweet_spot",
            "approaching_sweet_spot",
            "daily_setups",
            "undervalued_quality",
            "analyst_potential",
            "entry_watchlist",
            "falling_knives",
            "bottoming_watch",
            "risk_watch",
            "quality_momentum",
        ):
            categories[key] = {
                "label": key,
                "formula": "deterministic test formula",
                "partitioned_by_currency": True,
                "model_status": "heuristic_unvalidated",
                "actionable": False,
                "items_by_currency": {},
                "eligible_count": 0,
            }
        return {
            "contract_version": 3,
            "model_status": "heuristic_unvalidated",
            "actionable": False,
            "enabled": enabled,
            "blocking_reasons": [] if enabled else ["blocked fixture"],
            "categories": categories,
        }

    @staticmethod
    def _insight_metadata():
        return {
            "model_status": "heuristic_unvalidated",
            "actionable": False,
        }

    def test_atomic_json_roundtrip_and_no_temporary_sibling(self):
        path = self.work / "state.json"
        atomic_write_json(path, {"value": 3})
        self.assertEqual(load_json(path, required=True), {"value": 3})
        self.assertEqual(list(self.work.glob(".*.tmp")), [])

    def test_existing_corrupt_json_raises(self):
        path = self.work / "state.json"
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(CorruptDataError):
            load_json(path)

    def test_output_contract_and_stale_dashboard_gate(self):
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        status = build_data_status(
            universe_size=1,
            rows=[{"symbol": "ABC", "bar_date": "2026-08-11"}],
            failed_symbols={},
            now=now,
        )
        output = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": (now - timedelta(hours=48)).isoformat(),
            "data_status": status,
            "model_status": {"validation": "unvalidated", "actionable": False},
            "rankings_by_currency_asset": {},
            "insight_rankings": self._empty_insights(),
            "insight_metadata": self._insight_metadata(),
            "all": [],
        }
        self.assertIs(validate_output_contract(output), output)
        allowed, reasons = dashboard_gate(output, now=now, max_output_age_hours=36)
        self.assertFalse(allowed)
        self.assertTrue(any("hours old" in reason for reason in reasons))

    def test_completeness_gate_blocks(self):
        status = build_data_status(
            universe_size=100,
            rows=[{"symbol": str(i), "bar_date": "2026-08-11"} for i in range(90)],
            failed_symbols={str(i): "failed" for i in range(10)},
            now=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )
        self.assertEqual(status["status"], "blocked")
        self.assertFalse(status["data_actionable"])

    def test_synthetic_v3_output_roundtrip_contract(self):
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        row = enrich_row({
            "symbol": "ABC",
            "bar_date": "2026-08-11",
            "asset_type": "company_equity",
            "currency": "USD",
            "feature_coverage": {
                "rank_eligible": True,
                "technical_complete": False,
                "fundamental_complete": False,
                "fundamental_current": False,
            },
            "scenario_long": [],
        })
        output = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "data_status": build_data_status(
                universe_size=1, rows=[row], failed_symbols={}, now=now
            ),
            "model_status": {"validation": "unvalidated", "actionable": False},
            "rankings_by_currency_asset": {
                "USD": {"company_equity": [row]}
            },
            "insight_rankings": self._empty_insights(),
            "insight_metadata": self._insight_metadata(),
            "all": [row],
        }
        path = self.work / "latest.json"
        atomic_write_json(path, output)
        loaded = validate_output_contract(load_json(path, required=True))
        allowed, reasons = dashboard_gate(loaded, now=now)
        self.assertTrue(allowed, reasons)
        self.assertFalse(loaded["model_status"]["actionable"])
        broken = copy.deepcopy(output)
        del broken["rankings_by_currency_asset"]["USD"]["company_equity"][0][
            "valuation_thesis"
        ]
        with self.assertRaises(DataContractError):
            validate_output_contract(broken)

    def test_one_percent_rank_eligibility_blocks_complete_prices(self):
        rows = [
            {"symbol": str(index), "bar_date": "2026-08-11"}
            for index in range(100)
        ]
        status = build_data_status(
            universe_size=100,
            rows=rows,
            failed_symbols={},
            now=datetime(2026, 8, 12, tzinfo=timezone.utc),
            feature_coverage={
                "company_equity": {
                    "total": 100,
                    "technical_complete": 100,
                    "rank_eligible": 1,
                    "fundamental_complete_current": 100,
                }
            },
        )
        self.assertEqual(status["coverage_pct"], 100)
        self.assertEqual(status["status"], "blocked")
        self.assertTrue(any("rank-eligible coverage" in reason for reason in status["blocking_reasons"]))

    def test_zero_returned_expected_asset_class_blocks(self):
        rows = [
            {"symbol": f"C{index}", "bar_date": "2026-08-11"}
            for index in range(95)
        ]
        status = build_data_status(
            universe_size=100,
            rows=rows,
            failed_symbols={},
            now=datetime(2026, 8, 12, tzinfo=timezone.utc),
            min_coverage_pct=90,
            feature_coverage={
                "company_equity": {
                    "total": 95,
                    "analyzed_successfully": 95,
                    "technical_complete": 95,
                    "rank_eligible": 95,
                    "fundamental_complete_current": 95,
                },
                "etf_fund": {
                    "total": 5,
                    "analyzed_successfully": 0,
                    "technical_complete": 0,
                    "rank_eligible": 0,
                    "fundamental_complete_current": 0,
                },
            },
        )
        self.assertEqual(status["coverage_pct"], 95)
        self.assertEqual(status["status"], "blocked")
        self.assertTrue(
            any("etf_fund rank-eligible coverage 0.00%" in reason for reason in status["blocking_reasons"])
        )

    def test_inconsistent_blocked_status_cannot_render(self):
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        output = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "data_status": {
                "status": "blocked",
                "data_actionable": False,
                "blocking_reasons": [],
                "feature_coverage": {},
                "coverage_pct": 100.0,
                "fresh_bar_coverage_pct": 100.0,
                "failed_symbol_count": 0,
            },
            "model_status": {"validation": "unvalidated", "actionable": False},
            "rankings_by_currency_asset": {},
            "insight_rankings": self._empty_insights(False),
            "insight_metadata": self._insight_metadata(),
            "all": [],
        }
        allowed, reasons = dashboard_gate(output, now=now)
        self.assertFalse(allowed)
        self.assertIn("data_status.status is not 'ok'", reasons)
        self.assertIn("data_status.data_actionable is not true", reasons)

    def test_malformed_optional_portfolio_is_rejected_in_isolation(self):
        malformed = {
            "schema": "stock-radar-paper-portfolio",
            "schema_version": SCHEMA_VERSION,
            "cash": 1000.0,
            "positions": {"ABC": {"quantity": "ten"}},
            "pending_orders": [],
            "ledger": [],
            "equity_curve": [],
        }
        with self.assertRaises(DataContractError):
            validate_portfolio_contract(malformed)
        malformed["positions"] = {"ABC": ["not", "an", "object"]}
        with self.assertRaises(DataContractError):
            validate_portfolio_contract(malformed)

    def test_malformed_coverage_is_rejected_before_dashboard_formatting(self):
        output = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": "2026-08-12T12:00:00+00:00",
            "data_status": {
                "status": "ok",
                "data_actionable": True,
                "blocking_reasons": [],
                "feature_coverage": {},
                "coverage_pct": "one hundred",
                "fresh_bar_coverage_pct": 100.0,
                "failed_symbol_count": 0,
            },
            "model_status": {"validation": "unvalidated", "actionable": False},
            "rankings_by_currency_asset": {},
            "insight_rankings": self._empty_insights(),
            "insight_metadata": self._insight_metadata(),
            "all": [],
        }
        with self.assertRaises(DataContractError):
            validate_output_contract(output)

    def test_nonnumeric_fill_commission_is_rejected(self):
        portfolio = {
            "schema": "stock-radar-paper-portfolio",
            "schema_version": SCHEMA_VERSION,
            "cash": 1000.0,
            "positions": {},
            "pending_orders": [],
            "ledger": [
                {
                    "type": "FILL",
                    "quantity": 1.0,
                    "raw_fill_price": 100.0,
                    "execution_price": 100.1,
                    "gross_value": 100.1,
                    "commission": "free",
                }
            ],
            "equity_curve": [],
        }
        with self.assertRaises(DataContractError):
            validate_portfolio_contract(portfolio)

    def test_dry_run_redirects_writes_and_subsequent_reads(self):
        production = self.work / "data" / "output" / "latest.json"
        dry = self.work / "isolated"
        with patch.dict(
            "os.environ",
            {
                "STOCK_RADAR_DRY_RUN": "1",
                "STOCK_RADAR_DRY_RUN_DIR": str(dry),
            },
        ):
            atomic_write_json(production, {"dry": True})
            self.assertFalse(production.exists())
            self.assertEqual(load_json(production), {"dry": True})
        self.assertTrue((dry / "output" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
