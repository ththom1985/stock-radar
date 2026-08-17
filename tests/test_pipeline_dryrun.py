from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.analyze as analyze
from src.fetch import PriceFetchResult
from src.fx import FXResult
from src.persistence import load_json
from tests.helpers import ROOT, ProjectTempMixin


class DryRunPipelineTests(ProjectTempMixin, unittest.TestCase):
    def test_network_independent_pipeline_contract_does_not_mutate_tracked_data(self):
        now = datetime.now(timezone.utc)
        bar_date = (now.date() - timedelta(days=1)).isoformat()
        index = pd.date_range(end=bar_date, periods=260, freq="B")
        close = pd.Series(np.linspace(80, 120, len(index)), index=index)
        frame = pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 2_000_000.0,
                "RawOpen": close - 0.5,
                "RawHigh": close + 1,
                "RawLow": close - 1,
                "RawClose": close,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            }
        )
        symbols = ["AAA", "BBB"]
        bar_info = {
            symbol: {
                "bar_date": bar_date,
                "bar_timestamp": f"{bar_date}T00:00:00-04:00",
                "session_open_timestamp": f"{bar_date}T13:30:00+00:00",
                "session_close_timestamp": f"{bar_date}T20:00:00+00:00",
                "source_interval": "1d",
                "completed_bars_only": True,
                "session_mapping_status": "verified_conservative",
                "corporate_actions": [],
            }
            for symbol in symbols
        }
        prices = PriceFetchResult(
            prices={symbol: frame.copy() for symbol in symbols},
            bar_info=bar_info,
        )
        fundamentals = {
            symbol: {
                "quote_type": "EQUITY",
                "pe": 15.0,
                "pb": 2.0,
                "roe": 0.2,
                "profit_margin": 0.15,
                "debt_to_equity": 20.0,
                "revenue_growth": 0.1,
                "earnings_growth": 0.12,
                "sector": "Technology",
                "industry": "Software",
                "fetched_at": now.isoformat(),
                "last_success_at": now.isoformat(),
                "issuer_uuid": "issuer-a" if symbol == "AAA" else "issuer-b",
            }
            for symbol in symbols
        }
        production_latest = ROOT / "data" / "output" / "latest.json"
        before = production_latest.read_bytes()
        dry_dir = self.work / "dry-output"

        with patch.dict(
            "os.environ",
            {
                "STOCK_RADAR_DRY_RUN": "1",
                "STOCK_RADAR_DRY_RUN_DIR": str(dry_dir),
                "STOCK_RADAR_MAX_SYMBOLS": "2",
            },
        ), patch.object(
            analyze,
            "load_universe",
            return_value=[
                {"symbol": "AAA", "name": "AAA Corp", "exchange": "NYSE"},
                {"symbol": "BBB", "name": "BBB Corp", "exchange": "NASDAQ"},
            ],
        ), patch.object(
            analyze, "fetch_prices_with_status", return_value=prices
        ), patch.object(
            analyze, "fetch_fundamentals", return_value=fundamentals
        ), patch.object(
            analyze,
            "get_fx_rates_with_status",
            return_value=FXResult(
                rates={"USD": 1.0},
                status={"USD": {"rate": 1.0, "status": "fixed", "source": "definition"}},
                missing={},
            ),
        ), patch.object(
            analyze, "fetch_all_ticker_news", return_value=({}, {})
        ), patch.object(
            analyze,
            "fetch_market_news",
            return_value={
                "headlines": [],
                "market_sentiment": 50,
                "market_label": "neutral",
                "model_status": "unvalidated_context_only",
            },
        ), patch.object(
            analyze,
            "fetch_earnings",
            return_value={symbol: {"next_earnings": None} for symbol in symbols},
        ), patch.object(
            analyze, "fetch_deep", return_value={}
        ), patch.object(
            analyze,
            "fetch_macro",
            return_value={
                "regime": "neutral",
                "rate_dir": "flat",
                "context_only": True,
            },
        ), patch.object(
            analyze,
            "load_aschenbrenner",
            return_value={"holdings": {}, "report_quarter": None},
        ), patch.object(
            analyze, "_fetch_benchmarks", return_value=({}, {})
        ):
            result = analyze.run()

        self.assertEqual(production_latest.read_bytes(), before)
        self.assertEqual(result["run_mode"], "dry_run")
        dry_output = dry_dir / "output" / "latest.json"
        self.assertTrue(dry_output.exists())
        loaded = load_json(dry_output, required=True)
        self.assertEqual(loaded["schema"], "stock-radar-output")
        self.assertEqual(loaded["universe_size"], 2)
        self.assertEqual(loaded["model_status"]["validation"], "unvalidated")
        self.assertIn("probability_validation", loaded)
        for row in loaded["all"]:
            self.assertEqual(row["probability_forecast"]["status"], "withheld")
            self.assertTrue(
                row["probability_forecast"]["separate_from_radar_score"]
            )
            self.assertTrue(
                row["probability_forecast"]["separate_from_sweet_spot"]
            )

        market_dir = self.work / "market-contract"
        with patch.dict(
            "os.environ",
            {
                "STOCK_RADAR_DRY_RUN": "1",
                "STOCK_RADAR_DRY_RUN_DIR": str(market_dir),
                "STOCK_RADAR_MAX_SYMBOLS": "2",
                "STOCK_RADAR_MARKET_DATA_ONLY": "1",
            },
        ), patch.object(
            analyze,
            "load_universe",
            return_value=[
                {"symbol": "AAA", "name": "AAA Corp", "exchange": "NYSE"},
                {"symbol": "BBB", "name": "BBB Corp", "exchange": "NASDAQ"},
            ],
        ), patch.object(
            analyze, "fetch_prices_with_status", return_value=prices
        ), patch.object(
            analyze,
            "get_fx_rates_with_status",
            return_value=FXResult(
                rates={"USD": 1.0},
                status={"USD": {"rate": 1.0, "status": "fixed", "source": "definition"}},
                missing={},
            ),
        ):
            market_result = analyze.run()
        self.assertEqual(market_result["pipeline_scope"], "market_data_contract")
        self.assertIn("fundamentals", market_result["model_status"]["skipped_layers"])
        self.assertEqual(market_result["market_data_contract"]["status"], "ok")
        self.assertFalse(market_result["market_data_contract"]["full_model_ready"])
        self.assertEqual(production_latest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
