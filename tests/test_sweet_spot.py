from __future__ import annotations

import copy
import json
import math
import unittest

from src.insights import build_insight_rankings
from src.sweet_spot import build_sweet_spot, format_price, price_display_decimals


def model_row() -> dict:
    return {
        "symbol": "SYN",
        "asset_type": "company_equity",
        "currency": "USD",
        "price": 100.0,
        "atr": 2.0,
        "atr_pct": 2.0,
        "vol_annual_pct": 30.0,
        "sma50": 99.6,
        "ema21": 100.0,
        "pivot": 100.2,
        "sma200": 90.0,
        "weinstein_stage": 2,
        "trend_phase": {"tone": "up", "phase": "Aufwärtstrend"},
        "longterm_score": 75.0,
        "entry_timing_score": 70.0,
        "daily_signal_direction": "POSITIVE",
        "rsi": 50.0,
        "macd_hist": 0.3,
        "macd_hist_prev": 0.2,
        "downside_structure": {"risk": "niedrig"},
        "falling_knife": None,
        "bottoming": None,
        "earnings_in_days": 30,
        "completed_bars_only": True,
        "bar_age_days": 1,
        "feature_coverage": {
            "technical_complete": True,
            "fundamental_complete": True,
            "fundamental_current": True,
        },
        "fundamental_source_status": {"status": "current"},
        "jurisdiction_risk": {"level": "low"},
        "valuation_context": {"value_score": 70.0},
        "valuation_thesis": {
            "value_trap_risk": "low",
            "penalty_components": {},
        },
    }


class SweetSpotModelTests(unittest.TestCase):
    def test_confluence_cluster_has_weighted_ideal_and_bounded_nonzero_zone(self):
        row = model_row()
        sweet = build_sweet_spot(row, data_ready=True)
        expected = (99.6 * 1.2 + 100.0 * 1.1 + 100.2 * 0.85) / 3.15
        self.assertTrue(sweet["available"])
        self.assertEqual(sweet["confluence_count"], 3)
        self.assertEqual(sweet["independent_family_count"], 3)
        self.assertAlmostEqual(sweet["ideal"], expected, places=4)
        self.assertNotEqual(sweet["ideal"], round(sweet["ideal"], 4))
        self.assertLess(sweet["lower"], sweet["ideal"])
        self.assertLess(sweet["ideal"], sweet["upper"])
        self.assertGreaterEqual(sweet["zone_width_atr"], 0.7)
        self.assertLessEqual(sweet["zone_width_atr"], 1.2)

    def test_duplicate_levels_do_not_inflate_confluence(self):
        row = model_row()
        row.update({"sma50": 99.6, "ema21": 99.6, "pivot": 100.2})
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertEqual(sweet["confluence_count"], 2)
        self.assertEqual(
            len([item for item in sweet["components"] if item["value"] == 99.6]),
            1,
        )

    def test_correlated_sources_share_one_independent_family(self):
        pivot = model_row()
        for key in ("sma50", "ema21", "sma200"):
            pivot.pop(key, None)
        pivot.update({"pivot": 99.8, "pivot_s1": 100.1})
        sweet = build_sweet_spot(pivot, data_ready=True)
        self.assertEqual(sweet["independent_family_count"], 1)
        self.assertNotEqual(sweet["combined_status"], "in_zone_confirmed")
        self.assertEqual(
            {item["source_family"] for item in sweet["components"]},
            {"pivot"},
        )

        fast = model_row()
        for key in ("sma50", "pivot", "sma200"):
            fast.pop(key, None)
        fast.update({"sma20": 99.8, "ema21": 100.1})
        sweet = build_sweet_spot(fast, data_ready=True)
        self.assertEqual(sweet["independent_family_count"], 1)
        self.assertNotEqual(sweet["combined_status"], "in_zone_confirmed")

    def test_distant_level_is_excluded(self):
        row = model_row()
        row["sma150"] = 70.0
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertNotIn("SMA150", {item["label"] for item in sweet["components"]})
        self.assertIn("SMA150", {item["label"] for item in sweet["excluded_components"]})

    def test_no_price_atr_or_confluence_is_unavailable(self):
        self.assertFalse(build_sweet_spot({}, data_ready=True)["available"])
        row = model_row()
        for key in ("sma50", "ema21", "pivot", "sma200"):
            row.pop(key, None)
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertFalse(sweet["available"])
        self.assertEqual(sweet["combined_status"], "unavailable")

    def test_one_strong_level_in_valid_trend_is_reference_only(self):
        row = model_row()
        for key in ("ema21", "pivot", "sma200"):
            row.pop(key, None)
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertTrue(sweet["available"])
        self.assertEqual(sweet["confluence_count"], 1)
        self.assertLess(sweet["reliability_score"], 65)
        self.assertNotEqual(sweet["combined_status"], "in_zone_confirmed")

    def test_two_independent_families_do_not_automatically_pass_reliability(self):
        row = model_row()
        for key in ("sma50", "ema21", "pivot", "sma200"):
            row.pop(key, None)
        row.update({"sma150": 92.2, "pivot_s1": 93.9})
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertTrue(sweet["available"])
        self.assertEqual(sweet["independent_family_count"], 2)
        self.assertLess(sweet["reliability_score"], 65)
        self.assertNotEqual(sweet["combined_status"], "in_zone_confirmed")

    def test_dense_cluster_evidence_quality_does_not_saturate_at_100(self):
        row = model_row()
        row.update(
            {
                "sma20": 99.7,
                "sma150": 99.8,
                "sma200": 99.9,
                "pivot_s1": 100.1,
                "low20": 99.6,
            }
        )
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertGreaterEqual(sweet["independent_family_count"], 4)
        self.assertLess(sweet["reliability_score"], 100)

    def test_all_gates_pass_inside_zone_is_green(self):
        sweet = build_sweet_spot(model_row(), data_ready=True)
        self.assertEqual(sweet["current_position"], "in")
        self.assertEqual(sweet["technical_status"], "in_zone_confirmed")
        self.assertEqual(sweet["combined_status"], "in_zone_confirmed")
        self.assertEqual(sweet["tone"], "green")
        self.assertEqual(sweet["zone_tier"], "confirmed_confluence")

    def test_non_stage2_single_anchor_gets_reference_only_numbers(self):
        row = model_row()
        for key in ("ema21", "pivot", "sma200"):
            row.pop(key, None)
        row.update(
            {
                "sma50": 95.0,
                "weinstein_stage": 4,
                "trend_phase": {"tone": "down", "phase": "Abwärtstrend"},
            }
        )
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertTrue(sweet["available"])
        self.assertEqual(sweet["zone_tier"], "reference_only")
        self.assertEqual(sweet["confluence_count"], 1)
        self.assertEqual(sweet["independent_family_count"], 1)
        self.assertLessEqual(sweet["reliability_score"], 49)
        self.assertNotEqual(sweet["combined_status"], "in_zone_confirmed")
        self.assertLess(sweet["lower"], sweet["ideal"])
        self.assertLess(sweet["ideal"], sweet["upper"])

    def test_pivot_equal_current_is_rejected_as_fallback_anchor(self):
        row = model_row()
        for key in ("sma50", "ema21", "sma200"):
            row.pop(key, None)
        row.update({"pivot": row["price"], "pivot_s1": None})
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertFalse(sweet["available"])
        self.assertTrue(
            any(
                item["reason"] == "degenerate pivot equal to current close"
                for item in sweet["excluded_components"]
            )
        )

    def test_falling_knife_and_bottoming_never_green(self):
        knife = model_row()
        knife["falling_knife"] = {"warning": "active"}
        knife_sweet = build_sweet_spot(knife, data_ready=True)
        self.assertEqual(knife_sweet["combined_status"], "safety_blocked")
        self.assertEqual(knife_sweet["tone"], "red")

        bottom = model_row()
        bottom["bottoming"] = {"speculative": True}
        bottom_sweet = build_sweet_spot(bottom, data_ready=True)
        self.assertEqual(bottom_sweet["combined_status"], "safety_blocked")
        self.assertNotEqual(bottom_sweet["tone"], "green")

    def test_high_jurisdiction_risk_filters_only_combined_status(self):
        row = model_row()
        row["jurisdiction_risk"] = {"level": "high"}
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertEqual(sweet["technical_status"], "in_zone_confirmed")
        self.assertEqual(sweet["combined_status"], "in_zone_risk_filtered")
        self.assertEqual(sweet["tone"], "amber")

    def test_earnings_and_high_volatility_block_green(self):
        earnings = model_row()
        earnings["earnings_in_days"] = 7
        sweet = build_sweet_spot(earnings, data_ready=True)
        self.assertEqual(sweet["combined_status"], "setup_waiting_confirmation")
        self.assertNotEqual(sweet["tone"], "green")

        volatile = model_row()
        volatile.update({"atr_pct": 5.0, "vol_annual_pct": 60.0})
        sweet = build_sweet_spot(volatile, data_ready=True)
        self.assertEqual(sweet["combined_status"], "setup_waiting_confirmation")
        self.assertNotEqual(sweet["tone"], "green")

    def test_below_invalidation_is_red(self):
        row = model_row()
        row.update(
            {
                "price": 96.0,
                "sma50": 98.8,
                "ema21": 99.0,
                "pivot": None,
                "sma200": 90.0,
            }
        )
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertTrue(sweet["available"])
        self.assertLess(row["price"], sweet["invalidation_reference"]["value"])
        self.assertEqual(sweet["combined_status"], "broken_below")
        self.assertEqual(sweet["tone"], "red")

    def test_above_zone_within_one_atr_is_amber_approaching(self):
        row = model_row()
        row["price"] = 101.5
        sweet = build_sweet_spot(row, data_ready=True)
        self.assertEqual(sweet["current_position"], "above")
        self.assertEqual(sweet["combined_status"], "approaching")
        self.assertEqual(sweet["tone"], "amber")

    def test_geometry_is_always_positive_finite_and_language_is_safe(self):
        variants = []
        for offset in (-1.0, 0.0, 1.0):
            row = model_row()
            row["price"] += offset
            variants.append(build_sweet_spot(row, data_ready=True))
        for sweet in variants:
            self.assertTrue(
                all(
                    math.isfinite(sweet[key]) and sweet[key] > 0
                    for key in ("lower", "ideal", "upper")
                )
            )
            self.assertLess(sweet["lower"], sweet["ideal"])
            self.assertLess(sweet["ideal"], sweet["upper"])
            self.assertGreater(sweet["zone_width_pct"], 0)
            prose = json.dumps(
                {
                    key: value
                    for key, value in sweet.items()
                    if key
                    in {
                        "label",
                        "technical_label",
                        "formula",
                        "why_zone_here",
                        "why_green_or_not",
                        "confirmation_needed",
                        "invalidation_signals",
                        "investor_overlay_reasons",
                        "note",
                    }
                },
                ensure_ascii=False,
            ).casefold()
            for term in ("buy", "stop", "target", "probability"):
                self.assertNotIn(term, prose)
            self.assertIn("invalidation reference", prose)

    def test_result_is_deterministic(self):
        row = model_row()
        self.assertEqual(
            build_sweet_spot(copy.deepcopy(row), data_ready=True),
            build_sweet_spot(copy.deepcopy(row), data_ready=True),
        )

    def test_doge_scale_zone_formats_distinctly(self):
        row = model_row()
        scale = 0.0023
        for key in ("price", "atr", "sma50", "ema21", "pivot", "sma200"):
            row[key] *= scale
        row["macd_hist"] *= scale
        row["macd_hist_prev"] *= scale
        sweet = build_sweet_spot(row, data_ready=True)
        zone = [sweet["lower"], sweet["ideal"], sweet["upper"]]
        rendered = [format_price(value, zone) for value in zone]
        self.assertGreaterEqual(price_display_decimals(zone), 4)
        self.assertEqual(len(set(rendered)), 3)
        self.assertTrue(all("." in value for value in rendered))

    def test_confirmed_category_is_deterministic_and_currency_partitioned(self):
        rows = []
        for symbol, currency in (("BBB", "EUR"), ("CCC", "USD"), ("AAA", "USD")):
            row = model_row()
            row.update({"symbol": symbol, "currency": currency})
            row["sweet_spot"] = build_sweet_spot(row, data_ready=True)
            rows.append(row)
        first = build_insight_rankings(rows, enabled=True)
        second = build_insight_rankings(list(reversed(rows)), enabled=True)

        def symbols(ranking, currency):
            return [
                item["symbol"]
                for item in ranking["categories"]["in_sweet_spot"][
                    "items_by_currency"
                ].get(currency, [])
            ]

        self.assertEqual(symbols(first, "USD"), ["AAA", "CCC"])
        self.assertEqual(symbols(first, "EUR"), ["BBB"])
        self.assertEqual(symbols(first, "USD"), symbols(second, "USD"))

if __name__ == "__main__":
    unittest.main()
