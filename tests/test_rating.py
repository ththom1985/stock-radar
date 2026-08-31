import unittest

from src.rating import entry_label, entry_score


class EntryTimingTests(unittest.TestCase):
    def test_entry_score_is_independent_of_value_score(self):
        technical = {
            "price": 100,
            "sma20": 98,
            "sma50": 95,
            "sma200": 90,
            "rsi": 50,
            "macd_hist": 1,
            "macd_hist_prev": 0,
            "daily_signal_direction": "POSITIVE",
            "longterm_score": 70,
            "atr_pct": 2,
        }
        cheap = entry_score({**technical, "value_score": 100})
        expensive = entry_score({**technical, "value_score": 0})
        self.assertEqual(cheap, expensive)

    def test_low_timing_label_does_not_use_valuation_language(self):
        label, _ = entry_label(20)
        self.assertEqual(label, "technisch noch kein Einstieg")
        self.assertNotIn("teuer", label)


if __name__ == "__main__":
    unittest.main()
