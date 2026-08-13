from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.indicators import _rsi
from src.tech_advanced import advanced_indicators


def frame(close):
    index = pd.date_range("2025-01-01", periods=len(close), freq="D")
    close = pd.Series(close, index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.linspace(1000, 2000, len(close)),
        }
    )


class IndicatorTests(unittest.TestCase):
    def test_rsi_zero_loss_gain_and_flat_edges(self):
        self.assertEqual(_rsi(pd.Series(range(1, 50), dtype=float)).iloc[-1], 100.0)
        self.assertEqual(_rsi(pd.Series(range(50, 1, -1), dtype=float)).iloc[-1], 0.0)
        self.assertEqual(_rsi(pd.Series([10.0] * 49)).iloc[-1], 50.0)

    def test_aroon_recency_is_not_inverted(self):
        values = np.arange(1, 101, dtype=float)
        result = advanced_indicators(frame(values))
        self.assertEqual(result["aroon_up"], 100.0)
        self.assertEqual(result["aroon_down"], 0.0)

    def test_cci_uses_current_window_mean_deviation(self):
        values = 100 + np.sin(np.arange(100) / 3) * 7 + np.arange(100) * 0.1
        prices = frame(values)
        typical = (prices["High"] + prices["Low"] + prices["Close"]) / 3
        window = typical.iloc[-20:].to_numpy()
        mean = window.mean()
        expected = (window[-1] - mean) / (0.015 * np.mean(np.abs(window - mean)))
        self.assertAlmostEqual(advanced_indicators(prices)["cci"], expected, places=10)

    def test_ichimoku_is_shifted_and_pivots_use_prior_session(self):
        prices = frame(np.arange(1, 121, dtype=float))
        result = advanced_indicators(prices)
        high, low = prices["High"], prices["Low"]
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        expected_a = ((tenkan + kijun) / 2).shift(26).iloc[-1]
        expected_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26).iloc[-1]
        self.assertAlmostEqual(result["ichimoku_span_a"], expected_a)
        self.assertAlmostEqual(result["ichimoku_span_b"], expected_b)
        expected_pivot = (
            prices["High"].iloc[-1] + prices["Low"].iloc[-1] + prices["Close"].iloc[-1]
        ) / 3
        self.assertAlmostEqual(result["pivot"], expected_pivot)


if __name__ == "__main__":
    unittest.main()
