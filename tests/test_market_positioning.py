import unittest

from src.market_positioning import (
    CBOE_STATUS,
    build_positioning_context,
    summarize_contract,
)


class MarketPositioningTests(unittest.TestCase):
    def test_summarizes_positions_and_weekly_change(self):
        rows = [
            {
                "report_date_as_yyyy_mm_dd": "2026-08-18T00:00:00.000",
                "open_interest_all": "1000",
                "asset_mgr_positions_long": "600",
                "asset_mgr_positions_short": "100",
                "lev_money_positions_long": "100",
                "lev_money_positions_short": "300",
            },
            {
                "report_date_as_yyyy_mm_dd": "2026-08-11T00:00:00.000",
                "open_interest_all": "1000",
                "asset_mgr_positions_long": "550",
                "asset_mgr_positions_short": "100",
                "lev_money_positions_long": "100",
                "lev_money_positions_short": "250",
            },
        ]
        result = summarize_contract(rows)
        self.assertEqual(result["report_date"], "2026-08-18")
        self.assertEqual(result["asset_manager_weekly_change"], 5.0)
        self.assertGreater(result["score"], 50)

    def test_vix_positioning_is_inverted(self):
        rows = [
            {
                "report_date_as_yyyy_mm_dd": "2026-08-18",
                "open_interest_all": "1000",
                "asset_mgr_positions_long": "700",
                "asset_mgr_positions_short": "100",
                "lev_money_positions_long": "300",
                "lev_money_positions_short": "100",
            }
        ]
        self.assertLess(summarize_contract(rows, invert=True)["score"], 50)

    def test_cboe_gap_is_explicit_not_synthesized(self):
        context = build_positioning_context({})
        self.assertEqual(context["regime"], "unavailable")
        self.assertEqual(
            context["cboe_put_call"]["status"],
            "official_free_machine_endpoint_unavailable",
        )
        self.assertEqual(CBOE_STATUS["substitute"], context["cboe_put_call"]["substitute"])


if __name__ == "__main__":
    unittest.main()
