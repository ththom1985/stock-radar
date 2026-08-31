import unittest

from src.fundamentals import _extract


class FundamentalsTests(unittest.TestCase):
    def test_extracts_direct_fair_value_bridge_inputs(self):
        result = _extract(
            {
                "trailingEps": 5.0,
                "totalRevenue": 1_000,
                "freeCashflow": 100,
                "ebitda": 200,
                "sharesOutstanding": 50,
                "totalDebt": 300,
                "totalCash": 80,
            }
        )
        self.assertEqual(result["eps"], 5.0)
        self.assertEqual(result["revenue"], 1_000)
        self.assertEqual(result["free_cashflow"], 100)
        self.assertEqual(result["ebitda"], 200)
        self.assertEqual(result["shares_outstanding"], 50)
        self.assertEqual(result["total_debt"], 300)
        self.assertEqual(result["total_cash"], 80)


if __name__ == "__main__":
    unittest.main()
