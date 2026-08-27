import unittest

import pandas as pd

from src.financial_sector_history import (
    financial_history_baseline,
    financial_model_group,
    financial_peer_benchmarks,
    parse_financial_statements,
)


class FinancialSectorHistoryTests(unittest.TestCase):
    def test_parses_four_clean_common_equity_years(self):
        columns = pd.to_datetime(
            ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
        )
        balance = pd.DataFrame(
            {
                column: [1000 + index * 100, 100]
                for index, column in enumerate(columns)
            },
            index=["Common Stock Equity", "Ordinary Shares Number"],
        )
        income = pd.DataFrame(
            {
                column: [100 + index * 10]
                for index, column in enumerate(columns)
            },
            index=["Net Income Common Stockholders"],
        )
        points = parse_financial_statements(balance, income)
        self.assertEqual(len(points), 4)
        self.assertEqual(points[-1]["common_equity"], 1300)
        self.assertEqual(points[-1]["ordinary_shares"], 100)

    def test_builds_four_year_pb_to_roe_history(self):
        annual = [
            {
                "period_end": f"{year}-12-31",
                "common_equity": 1000,
                "ordinary_shares": 100,
                "net_income_common": 100,
            }
            for year in range(2022, 2026)
        ]
        prices = pd.DataFrame(
            {"Close": [20, 21, 22, 23]},
            index=pd.to_datetime(
                [f"{year}-12-31" for year in range(2022, 2026)]
            ),
        )
        result = financial_history_baseline(
            {"annual": annual, "source": "test"},
            prices,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["annual_points_available"], 4)
        self.assertAlmostEqual(result["pb_to_roe_median"], 21.5)

    def test_peer_groups_use_current_pb_relative_to_roe(self):
        rows = [
            {"industry": "Banks - Regional", "pb": 1.5, "roe_pct": 10},
            {"industry": "Banks - Diversified", "pb": 2.0, "roe_pct": 10},
            {"industry": "Insurance - Diversified", "pb": 1.2, "roe_pct": 12},
        ]
        result = financial_peer_benchmarks(rows)
        self.assertEqual(result["bank"]["peer_count"], 2)
        self.assertEqual(result["bank"]["pb_to_roe_median"], 17.5)
        self.assertEqual(result["insurance"]["peer_count"], 1)

    def test_policy_keeps_reits_and_unassigned_financials_withheld(self):
        self.assertEqual(
            financial_model_group({"industry": "REIT - Retail"}),
            "reit",
        )
        self.assertEqual(
            financial_model_group(
                {"symbol": "BX", "sector": "Financial Services", "industry": "Asset Management"}
            ),
            "unsupported_financial",
        )
        self.assertEqual(
            financial_model_group(
                {"symbol": "PYPL", "sector": "Financial Services", "industry": "Credit Services"}
            ),
            "standard",
        )


if __name__ == "__main__":
    unittest.main()
