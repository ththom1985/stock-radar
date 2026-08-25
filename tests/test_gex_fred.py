import unittest
from datetime import datetime, timezone

from src.fred_regime import build_regime
from src.options_gex import black_scholes_gamma, calculate_gex


class GexFredTests(unittest.TestCase):
    def test_black_scholes_gamma_is_positive(self):
        gamma = black_scholes_gamma(100, 100, 30 / 365.25, 0.25)
        self.assertGreater(gamma, 0)

    def test_call_and_put_open_interest_net_into_gex(self):
        expirations = [
            {
                "expiration": "2026-09-25",
                "calls": [
                    {"strike": 100, "openInterest": 2000, "impliedVolatility": 0.25}
                ],
                "puts": [
                    {"strike": 100, "openInterest": 500, "impliedVolatility": 0.25}
                ],
            }
        ]
        result = calculate_gex(
            100,
            expirations,
            market_cap=1_000_000_000,
            now=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )
        self.assertEqual(result["direction"], "dampening")
        self.assertGreater(result["net_gex_usd_per_1pct"], 0)
        self.assertEqual(result["contract_rows_used"], 2)

    def test_fred_regime_explains_inputs(self):
        result = build_regime(
            {
                "dgs10": {"value": 4.0},
                "dgs2": {"value": 3.5},
                "high_yield_spread": {"value": 3.0},
                "financial_conditions": {"value": -0.5},
            }
        )
        self.assertEqual(result["regime"], "risk_on")
        self.assertEqual(len(result["reasons"]), 3)


if __name__ == "__main__":
    unittest.main()
