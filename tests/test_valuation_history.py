import unittest
from datetime import datetime, timezone

from src.expert_layer import build_valuation_assessment, sector_valuation_medians
from src.valuation_history import five_year_averages


class ValuationHistoryTests(unittest.TestCase):
    def test_five_year_average_is_withheld_until_enough_months_exist(self):
        history = {
            "symbols": {
                "AAA": [
                    {"month": f"2026-{month:02d}", "metrics": {"pe": 10 + month}}
                    for month in range(1, 13)
                ]
            }
        }
        result = five_year_averages(history, "AAA")
        self.assertFalse(result["complete"])
        self.assertIsNone(result["metrics"]["pe"])

    def test_sector_median_requires_peer_coverage(self):
        rows = [
            {"sector": "Tech", "pe": value, "symbol": str(value)}
            for value in (10, 20, 30, 40, 50)
        ]
        medians = sector_valuation_medians(rows)
        self.assertEqual(medians["Tech"]["pe"], 30)

    def test_fair_range_uses_available_sector_references(self):
        row = {
            "sector": "Tech",
            "currency": "USD",
            "price_local": 100,
            "pe": 20,
            "price_to_sales": 4,
        }
        peers = {
            "Tech": {
                "pe": 16,
                "price_to_sales": 3,
                "peer_counts": {"pe": 10, "price_to_sales": 10},
            }
        }
        result = build_valuation_assessment(
            row,
            peers,
            {"metrics": {}, "months_available": 1, "complete": False},
        )
        self.assertEqual(result["verdict"], "overpriced")
        self.assertIsNotNone(result["fair_value_range"])
        self.assertFalse(result["own_5y_complete"])


if __name__ == "__main__":
    unittest.main()
