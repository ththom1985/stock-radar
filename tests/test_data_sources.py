from __future__ import annotations

import unittest
import csv
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.assets import COMPANY_EQUITY, CRYPTO, ETF_FUND, INDEX_OTHER, classify_asset
from src.analyze import (
    COMPARABLE_TECHNICAL_FIELDS,
    _apply_company_fundamentals,
    _partition_rankings,
    _technical_complete,
)
from src.earnings import _earnings_dates, _next_date
from src.fetch import completed_daily_bars, fetch_prices_with_status
from src.news_engine import _filter_entries
from src.projection import project
from src.markets import market_profile, session_bounds
from tests.helpers import ROOT


class DataSourceTests(unittest.TestCase):
    def test_current_local_daily_bar_is_excluded_and_raw_prices_preserved(self):
        index = pd.date_range("2026-08-10", periods=3, freq="D", tz="UTC")
        raw = pd.DataFrame(
            {
                "Open": [100, 102, 104],
                "High": [101, 103, 105],
                "Low": [99, 101, 103],
                "Close": [100, 102, 104],
                "Adj Close": [50, 51, 52],
                "Volume": [1000, 1100, 1200],
            },
            index=index,
        )
        completed, info = completed_daily_bars(
            raw, now=datetime(2026, 8, 12, 12, tzinfo=timezone.utc)
        )
        self.assertEqual(len(completed), 2)
        self.assertEqual(info["bar_date"], "2026-08-11")
        self.assertEqual(info["excluded_partial_rows"], 1)
        self.assertEqual(info["exchange_timezone"], "UTC")
        self.assertEqual(info["timezone_source"], "yfinance_index_timezone")
        self.assertEqual(completed["RawOpen"].iloc[-1], 102)
        self.assertEqual(completed["Open"].iloc[-1], 51)

    def test_failed_batch_splits_and_reports_only_failed_symbol(self):
        index = pd.date_range("2026-06-01", periods=50, freq="D")
        single = pd.DataFrame(
            {
                "Open": np.arange(50) + 100,
                "High": np.arange(50) + 101,
                "Low": np.arange(50) + 99,
                "Close": np.arange(50) + 100,
                "Adj Close": np.arange(50) + 100,
                "Volume": 1000,
            },
            index=index,
        )

        def downloader(symbols, _period):
            if len(symbols) > 1 or symbols == ["BAD"]:
                raise RuntimeError("provider batch error")
            return single

        result = fetch_prices_with_status(
            ["GOOD", "BAD"],
            retries=0,
            downloader=downloader,
            now=datetime(2026, 8, 12, tzinfo=timezone.utc),
            verbose=False,
        )
        self.assertIn("GOOD", result.prices)
        self.assertEqual(set(result.failed_symbols), {"BAD"})

    def test_exchange_session_boundaries_for_naive_indexes(self):
        def daily_frame(session_date):
            index = pd.DatetimeIndex([pd.Timestamp(session_date)])
            return pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.0],
                    "Volume": [1000],
                },
                index=index,
            )

        us = daily_frame("2026-01-06")
        before_us, _ = completed_daily_bars(
            us,
            symbol="AAPL",
            now=datetime(2026, 1, 6, 22, 0, tzinfo=timezone.utc),
        )
        after_us, us_info = completed_daily_bars(
            us,
            symbol="AAPL",
            now=datetime(2026, 1, 6, 22, 31, tzinfo=timezone.utc),
        )
        self.assertTrue(before_us.empty)
        self.assertEqual(len(after_us), 1)
        self.assertEqual(us_info["exchange_timezone"], "America/New_York")

        europe, _ = completed_daily_bars(
            daily_frame("2026-01-06"),
            symbol="SAP.DE",
            now=datetime(2026, 1, 6, 18, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(len(europe), 1)

        before_tokyo, _ = completed_daily_bars(
            daily_frame("2026-01-06"),
            symbol="7203.T",
            now=datetime(2026, 1, 6, 7, 0, tzinfo=timezone.utc),
        )
        after_tokyo, _ = completed_daily_bars(
            daily_frame("2026-01-06"),
            symbol="7203.T",
            now=datetime(2026, 1, 6, 7, 31, tzinfo=timezone.utc),
        )
        self.assertTrue(before_tokyo.empty)
        self.assertEqual(len(after_tokyo), 1)

        before_crypto, _ = completed_daily_bars(
            daily_frame("2026-01-06"),
            symbol="BTC-USD",
            now=datetime(2026, 1, 6, 23, 59, tzinfo=timezone.utc),
        )
        after_crypto, _ = completed_daily_bars(
            daily_frame("2026-01-06"),
            symbol="BTC-USD",
            now=datetime(2026, 1, 7, 1, 31, tzinfo=timezone.utc),
        )
        self.assertTrue(before_crypto.empty)
        self.assertEqual(len(after_crypto), 1)

    def test_all_configured_suffixes_have_known_session_profiles(self):
        with (ROOT / "data" / "tickers.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            symbols = [
                row["symbol"].strip()
                for row in csv.DictReader(handle)
                if (row.get("symbol") or "").strip()
                and not row["symbol"].strip().startswith("#")
            ]
        unknown = sorted(
            {
                symbol
                for symbol in symbols
                if market_profile(symbol).mapping_status != "verified_conservative"
            }
        )
        self.assertEqual(unknown, [])

    def test_reviewed_market_boundaries_use_verified_local_profiles(self):
        cases = {
            "2222.SR": "Asia/Riyadh",
            "OMV.VI": "Europe/Vienna",
            "PETR4.SA": "America/Sao_Paulo",
            "WALMEX.MX": "America/Mexico_City",
            "NPN.JO": "Africa/Johannesburg",
            "TEST.NE": "America/Toronto",
            "TEST.CN": "America/Toronto",
        }
        frame = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-06")]),
        )
        for symbol, timezone_name in cases.items():
            with self.subTest(symbol=symbol):
                profile = market_profile(symbol)
                self.assertEqual(profile.timezone_name, timezone_name)
                self.assertEqual(profile.mapping_status, "verified_conservative")
                _, closed = session_bounds(date(2026, 1, 6), profile)
                before, _ = completed_daily_bars(
                    frame,
                    symbol=symbol,
                    now=closed.astimezone(timezone.utc) + timedelta(minutes=89),
                )
                after, info = completed_daily_bars(
                    frame,
                    symbol=symbol,
                    now=closed.astimezone(timezone.utc) + timedelta(minutes=91),
                )
                self.assertTrue(before.empty)
                self.assertEqual(len(after), 1)
                self.assertEqual(info["session_mapping_status"], "verified_conservative")

    def test_unknown_suffix_is_explicitly_blocked_not_silent_utc(self):
        profile = market_profile("UNKNOWN.QQ")
        self.assertEqual(profile.mapping_status, "unknown_blocked")
        self.assertIn("unknown_suffix", profile.source)

    def test_earnings_is_future_only_and_previous_is_separate(self):
        today = date.today()
        past = today - timedelta(days=3)
        future = today + timedelta(days=7)
        calendar = {"Earnings Date": [past, future]}
        self.assertEqual(_next_date(calendar), future.isoformat())
        self.assertEqual(_earnings_dates(calendar), (future.isoformat(), past.isoformat()))
        self.assertIsNone(_next_date({"Earnings Date": [past]}))

    def test_asset_classification(self):
        self.assertEqual(classify_asset("ABC", "ABC Inc", {"quote_type": "EQUITY"}), COMPANY_EQUITY)
        self.assertEqual(
            classify_asset("FCH", "Funding Circle Holdings", {"quote_type": "EQUITY"}),
            COMPANY_EQUITY,
        )
        self.assertEqual(classify_asset("SPY", "SPDR S&P ETF", {}), ETF_FUND)
        self.assertEqual(classify_asset("BTC-USD", "Bitcoin", {}), CRYPTO)
        self.assertEqual(classify_asset("^GSPC", "S&P 500", {}), INDEX_OTHER)

    def test_fundamental_company_score_is_disabled_for_funds(self):
        row = {"symbol": "SPY", "asset_type": ETF_FUND, "longterm_score": 80}
        _apply_company_fundamentals(
            row,
            {
                "pe": 10,
                "pb": 1,
                "roe": 0.2,
                "profit_margin": 0.2,
                "debt_to_equity": 20,
                "revenue_growth": 0.2,
                "earnings_growth": 0.2,
            },
            {},
        )
        self.assertIsNone(row["fundamental_score"])
        self.assertIsNone(row["investment_score"])

    def test_missing_rvol_blocks_technical_rank_eligibility(self):
        row = {field: 1.0 for field in COMPARABLE_TECHNICAL_FIELDS}
        self.assertTrue(_technical_complete(row))
        row["rvol"] = None
        self.assertFalse(_technical_complete(row))

    def test_currency_partitions_never_share_one_global_rank(self):
        rows = [
            {
                "symbol": "USD1",
                "currency": "USD",
                "asset_type": "company_equity",
                "radar_score": 60,
            },
            {
                "symbol": "EUR1.DE",
                "currency": "EUR",
                "asset_type": "company_equity",
                "radar_score": 99,
            },
        ]
        partitions = _partition_rankings(rows)
        self.assertEqual(
            [row["symbol"] for row in partitions["USD"]["company_equity"]],
            ["USD1"],
        )
        self.assertEqual(
            [row["symbol"] for row in partitions["EUR"]["company_equity"]],
            ["EUR1.DE"],
        )
        self.assertNotIn("rankings_by_asset", partitions)

    def test_scenarios_are_positive_and_unvalidated(self):
        scenarios = project(
            {
                "price": 10.0,
                "vol_daily": 0.25,
                "longterm_score": 40,
                "daily_signal_direction": "NEGATIVE",
                "daily_signal_score": 80,
            }
        )
        self.assertTrue(scenarios)
        for scenario in scenarios:
            self.assertGreater(scenario["range_low_price"], 0)
            self.assertGreater(scenario["range_high_price"], 0)
            self.assertEqual(scenario["model_status"], "unvalidated")
            self.assertIn("not statistically calibrated", scenario["interpretation"])

    def test_news_filters_old_and_future_items(self):
        now = datetime(2026, 8, 12, 12, tzinfo=timezone.utc)
        feed = SimpleNamespace(
            entries=[
                {
                    "title": "current issuer item",
                    "link": "https://example.com/current",
                    "published": format_datetime(now - timedelta(hours=2)),
                },
                {
                    "title": "old item",
                    "link": "https://example.com/old",
                    "published": format_datetime(now - timedelta(days=20)),
                },
            ]
        )
        result = _filter_entries(feed, limit=10, now=now)
        self.assertEqual([item["title"] for item in result], ["current issuer item"])


if __name__ == "__main__":
    unittest.main()
