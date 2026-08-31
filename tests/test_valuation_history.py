import unittest
from datetime import datetime, timezone

import pandas as pd

from src.expert_layer import (
    _apply_minimum_fair_band,
    _direct_implied_price,
    build_valuation_assessment,
    sector_valuation_medians,
)
from src.valuation_history import _sec_annual_backfill, five_year_averages


class ValuationHistoryTests(unittest.TestCase):
    def test_minimum_fair_band_uses_typical_daily_move(self):
        lower, upper, audit = _apply_minimum_fair_band(
            {"atr": 3, "vol_annual_pct": 20},
            99,
            101,
        )
        self.assertEqual((lower, upper), (97, 103))
        self.assertEqual(audit["status"], "expanded")
        self.assertEqual(audit["minimum_half_width_pct"], 3)

    def test_minimum_fair_band_never_shrinks_existing_range(self):
        lower, upper, audit = _apply_minimum_fair_band(
            {"atr": 2, "vol_annual_pct": 20},
            90,
            110,
        )
        self.assertEqual((lower, upper), (90, 110))
        self.assertEqual(audit["status"], "already_wide_enough")

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

    def test_sec_backfill_creates_multi_year_valuation_history(self):
        annual = [
            {
                "period_end": f"{year}-12-31",
                "filed": f"{year + 1}-02-01",
                "diluted_eps": 5 + index,
                "diluted_shares": 100,
                "revenue": 1000 + index * 100,
                "free_cash_flow": 100 + index * 10,
            }
            for index, year in enumerate(range(2021, 2026))
        ]
        frame = pd.DataFrame(
            {"Close": [100, 110, 120, 130, 140]},
            index=pd.to_datetime([f"{year}-12-31" for year in range(2021, 2026)]),
        )
        points = _sec_annual_backfill({"annual": annual}, frame)
        result = five_year_averages({"symbols": {"AAA": points}}, "AAA")
        self.assertTrue(result["complete"])
        self.assertEqual(result["history_type"], "sec_annual_backfill")
        self.assertGreaterEqual(len(result["supported_metrics"]), 2)

    def test_fair_range_uses_available_sector_references(self):
        row = {
            "sector": "Tech",
            "currency": "USD",
            "price_local": 100,
            "pe": 20,
            "price_to_sales": 4,
            "sec_companyfacts": {
                "latest": {
                    "diluted_eps": 5,
                    "diluted_shares": 100,
                    "revenue": 3000,
                }
            },
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
        self.assertEqual(result["verdict"], "data_review_required")
        self.assertEqual(
            result["plausibility_gate"]["status"], "withheld_narrow_basis"
        )
        self.assertIsNotNone(result["raw_fair_value_range"])
        self.assertIsNone(result["fair_value_range"])
        self.assertFalse(result["own_5y_complete"])

    def test_wide_fair_range_is_withheld(self):
        row = {
            "sector": "Tech",
            "currency": "USD",
            "price_local": 100,
            "pe": 20,
            "price_to_sales": 4,
            "sec_companyfacts": {
                "latest": {
                    "diluted_eps": 5,
                    "diluted_shares": 100,
                    "revenue": 4000,
                }
            },
        }
        peers = {
            "Tech": {
                "pe": 10,
                "price_to_sales": 8,
                "peer_counts": {"pe": 10, "price_to_sales": 10},
            }
        }
        own = {
            "metrics": {"pe": 12, "price_to_sales": 7},
            "complete": True,
            "annual_points_available": 5,
        }
        result = build_valuation_assessment(row, peers, own)
        self.assertEqual(result["plausibility_gate"]["status"], "withheld_wide_range")
        self.assertIsNone(result["fair_value_range"])

    def test_extreme_fair_value_is_withheld_fail_closed(self):
        row = {
            "sector": "Industrials",
            "currency": "EUR",
            "price_local": 176,
            "pe": 7.7,
            "ev_ebitda": 5.5,
            "price_to_sales": 0.35,
            "sec_companyfacts": {
                "latest": {
                    "diluted_eps": 20,
                    "diluted_shares": 100,
                    "revenue": 50000,
                }
            },
        }
        peers = {
            "Industrials": {
                "pe": 26.25,
                "ev_ebitda": 18.75,
                "price_to_sales": 1.193,
                "peer_counts": {"pe": 100, "ev_ebitda": 100, "price_to_sales": 100},
            }
        }
        result = build_valuation_assessment(
            row,
            peers,
            {"metrics": {}, "months_available": 1, "complete": False},
        )
        self.assertEqual(result["verdict"], "data_review_required")
        self.assertIsNone(result["fair_value_range"])
        self.assertIsNotNone(result["raw_fair_value_range"])
        self.assertEqual(
            result["plausibility_gate"]["status"],
            "withheld_extreme_deviation",
        )

    def test_bank_model_uses_four_year_pb_relative_to_roe(self):
        row = {
            "symbol": "BANK",
            "industry": "Banks - Diversified",
            "currency": "USD",
            "price_local": 100,
            "bvps": 50,
            "pb": 2,
            "roe_pct": 10,
        }
        result = build_valuation_assessment(
            row,
            financial_history={
                "complete": True,
                "annual_points_available": 4,
                "annual_points": [{"period_end": "2025-12-31"}] * 4,
                "pb_to_roe_median": 20,
            },
            financial_peer={"pb_to_roe_median": 15, "peer_count": 32},
        )
        self.assertEqual(result["plausibility_gate"]["status"], "pass")
        self.assertEqual(result["sector_model"], "bank_pb_to_roe_4y")
        self.assertEqual(
            result["fair_value_range"],
            {
                "lower": 75.0,
                "upper": 100.0,
                "currency": "USD",
                "method": (
                    "P/B relativ zum ROE, abgeleitet aus eigener Vierjahreshistorie "
                    "und aktueller Sektor-Peergroup"
                ),
                "input_count": 2,
                "implied_price_count": 2,
            },
        )
        self.assertIn("4-Jahres-Basis", result["history_note"])

    def test_reit_remains_withheld_with_specific_reason(self):
        result = build_valuation_assessment(
            {
                "symbol": "REIT",
                "sector": "Real Estate",
                "industry": "REIT - Retail",
                "currency": "USD",
                "price_local": 100,
                "pe": 20,
                "price_to_sales": 4,
                "sec_companyfacts": {
                    "latest": {
                        "diluted_eps": 5,
                        "diluted_shares": 100,
                        "revenue": 3000,
                    }
                },
            },
            {
                "Real Estate": {
                    "pe": 18,
                    "price_to_sales": 3,
                    "peer_counts": {"pe": 10, "price_to_sales": 10},
                }
            },
            {
                "metrics": {"pe": 18, "price_to_sales": 3},
                "complete": True,
                "annual_points_available": 5,
            },
        )
        self.assertEqual(
            result["plausibility_gate"]["status"],
            "withheld_sector_model",
        )
        self.assertIn("FFO-/AFFO", result["plausibility_gate"]["reason"])

    def test_standard_fair_value_does_not_change_with_current_price(self):
        row = {
            "symbol": "DIRECT",
            "sector": "Tech",
            "currency": "USD",
            "price_local": 100,
            "sec_companyfacts": {
                "latest": {
                    "diluted_eps": 5,
                    "diluted_shares": 100,
                    "revenue": 3000,
                    "free_cash_flow": 500,
                }
            },
        }
        peers = {
            "Tech": {
                "pe": 20,
                "price_to_sales": 4,
                "price_to_fcf": 20,
                "peer_counts": {
                    "pe": 10,
                    "price_to_sales": 10,
                    "price_to_fcf": 10,
                },
            }
        }
        own = {
            "metrics": {
                "pe": 18,
                "price_to_sales": 3.5,
                "price_to_fcf": 18,
            },
            "complete": True,
            "annual_points_available": 5,
        }
        first = build_valuation_assessment(row, peers, own)
        second = build_valuation_assessment(
            {**row, "price_local": 150},
            peers,
            own,
        )
        self.assertEqual(
            first["fair_value_range"],
            second["fair_value_range"],
        )

    def test_ev_ebitda_converts_enterprise_to_equity_value(self):
        price, label = _direct_implied_price(
            "ev_ebitda",
            10,
            {
                "ebitda": 1_000,
                "net_debt": 200,
                "shares": 100,
            },
        )
        self.assertEqual(price, 98)
        self.assertIn("Nettoverschuldung", label)

    def test_financial_fair_value_does_not_use_current_price(self):
        row = {
            "symbol": "BANK",
            "industry": "Banks - Diversified",
            "currency": "USD",
            "price_local": 100,
            "bvps": 50,
            "pb": 2,
            "roe_pct": 10,
        }
        history = {
            "complete": True,
            "annual_points_available": 4,
            "annual_points": [{"period_end": "2025-12-31"}] * 4,
            "pb_to_roe_median": 20,
        }
        peer = {"pb_to_roe_median": 15, "peer_count": 32}
        first = build_valuation_assessment(
            row,
            financial_history=history,
            financial_peer=peer,
        )
        second = build_valuation_assessment(
            {**row, "price_local": 150},
            financial_history=history,
            financial_peer=peer,
        )
        self.assertEqual(first["fair_value_range"], second["fair_value_range"])


if __name__ == "__main__":
    unittest.main()
