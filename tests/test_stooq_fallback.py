import unittest
from datetime import datetime, timezone

import pandas as pd

from src.fetch import fetch_prices_with_status


def frame():
    index = pd.date_range("2026-05-01", periods=60, freq="B")
    return pd.DataFrame(
        {
            "Open": range(100, 160),
            "High": range(101, 161),
            "Low": range(99, 159),
            "Close": range(100, 160),
            "Adj Close": range(100, 160),
            "Volume": [1_000_000] * 60,
        },
        index=index,
    )


class StooqFallbackTests(unittest.TestCase):
    def test_uses_stooq_only_after_yahoo_failure(self):
        calls = []

        def failed_yahoo(symbols, period):
            raise RuntimeError("Yahoo unavailable")

        def stooq(symbol, period):
            calls.append(symbol)
            return frame()

        result = fetch_prices_with_status(
            ["AAPL"],
            downloader=failed_yahoo,
            fallback_downloader=stooq,
            now=datetime(2026, 8, 25, 23, tzinfo=timezone.utc),
            verbose=False,
        )
        self.assertEqual(calls, ["AAPL"])
        self.assertIn("AAPL", result.prices)
        self.assertEqual(result.bar_info["AAPL"]["price_provider"], "stooq")
        self.assertNotIn("AAPL", result.failed_symbols)

    def test_does_not_guess_non_us_stooq_symbol(self):
        def failed_yahoo(symbols, period):
            raise RuntimeError("Yahoo unavailable")

        calls = []
        result = fetch_prices_with_status(
            ["SAP.DE"],
            downloader=failed_yahoo,
            fallback_downloader=lambda symbol, period: calls.append(symbol),
            now=datetime(2026, 8, 25, 23, tzinfo=timezone.utc),
            verbose=False,
        )
        self.assertEqual(calls, [])
        self.assertIn("SAP.DE", result.failed_symbols)


if __name__ == "__main__":
    unittest.main()
