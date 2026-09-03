import unittest

from unittest.mock import patch

from src.question_views import build_question_views


def row(symbol="AAA", *, verdict="clearly_undervalued", price=50, lower=100, upper=120):
    return {
        "symbol": symbol,
        "name": symbol,
        "display_name_full": symbol,
        "asset_type": "company_equity",
        "currency": "USD",
        "price_local": price,
        "growth_score": 80,
        "quality_score": 85,
        "entry_timing_score": 80,
        "sweet_spot": {"combined_status": "approaching"},
        "expert_analysis": {
            "signal": "wait_for_pullback",
            "evidence_quality": "high",
            "coverage_basis": {"status": "standard_us"},
            "long_term": {"score": 78, "coverage_pct": 80},
            "valuation": {
                "verdict": verdict,
                "fair_value_range": {"lower": lower, "upper": upper, "input_count": 5},
                "plausibility_gate": {"status": "pass"},
                "basis_quality": {"status": "broad", "definition": "two families"},
                "own_5y_complete": True,
            },
            "risks": {"top_risks": ["Kein einzelnes dominantes Risiko."]},
        },
    }


class QuestionViewsTests(unittest.TestCase):
    def setUp(self):
        self.enabled = patch(
            "src.question_views.VALUATION_LISTS_ENABLED",
            True,
        )
        self.enabled.start()

    def tearDown(self):
        self.enabled.stop()

    def test_lists_are_disabled_by_default_during_repair(self):
        with patch("src.question_views.VALUATION_LISTS_ENABLED", False):
            result = build_question_views([row()])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["cheap_with_potential"], [])
        self.assertEqual(result["expensive_now"], [])
        self.assertIn("überarbeitet", result["empty_state"])

    def test_cheap_requires_gate_and_keeps_falling_knives_as_watch_items(self):
        valid = row()
        withheld = row("WITHHELD")
        withheld["expert_analysis"]["valuation"]["plausibility_gate"]["status"] = "withheld_extreme_deviation"
        knife = row("KNIFE")
        knife["falling_knife"] = {"active": True}
        result = build_question_views([valid, withheld, knife])
        self.assertEqual(
            [item["symbol"] for item in result["cheap_with_potential"]],
            ["AAA", "KNIFE"],
        )
        knife_item = result["cheap_with_potential"][1]
        self.assertEqual(knife_item["course_state"], "falling")
        self.assertEqual(knife_item["entry_guidance"]["code"], "watch_falling")
        self.assertIn("nicht greifen", knife_item["badge"]["label"])

    def test_high_value_trap_excludes_even_with_course_warning(self):
        trapped = row()
        trapped["falling_knife"] = {"active": True}
        trapped["valuation_context"] = {"value_trap_risk": "high"}
        result = build_question_views([trapped])
        self.assertEqual(result["cheap_with_potential"], [])
        self.assertEqual(
            result["excluded_cheap"][0]["reasons"],
            ["Das Value-Trap-Risiko ist hoch."],
        )

    def test_medium_value_trap_is_visible_but_not_excluded(self):
        medium = row()
        medium["valuation_context"] = {"value_trap_risk": "medium"}
        item = build_question_views([medium])["cheap_with_potential"][0]
        self.assertEqual(item["value_trap_risk"], "medium")
        self.assertIn("kleiner positionieren", item["risk_note"])

    def test_timing_safety_block_is_guidance_not_exclusion(self):
        waiting = row()
        waiting["sweet_spot"] = {
            "combined_status": "safety_blocked",
            "lower": 45,
            "ideal": 47,
            "upper": 49,
        }
        waiting["fx_usd"] = 1
        item = build_question_views([waiting])["cheap_with_potential"][0]
        self.assertEqual(item["entry_guidance"]["code"], "watch_entry")
        self.assertEqual(item["entry_guidance"]["target_price"], 49)

    def test_acute_fundamental_warning_excludes(self):
        warned = row()
        warned["risk_warnings"] = ["⚠️ Kritischer Altman-Z-Wert (1.2)"]
        result = build_question_views([warned])
        self.assertEqual(result["cheap_with_potential"], [])
        self.assertEqual(
            result["excluded_cheap"][0]["reasons"],
            ["⚠️ Kritischer Altman-Z-Wert (1.2)"],
        )

    def test_cheap_uses_momentum_free_quality_growth_potential(self):
        valid = row()
        valid["expert_analysis"]["long_term"] = {"score": 1, "coverage_pct": 1}
        valid["longterm_score"] = 0
        valid["rs_rating"] = 0
        valid["news_score"] = 0
        result = build_question_views([valid])
        self.assertEqual([item["symbol"] for item in result["cheap_with_potential"]], ["AAA"])
        item = result["cheap_with_potential"][0]
        self.assertEqual(item["potential_score"], 83.0)
        self.assertNotIn("long_term_score", item)

    def test_cheap_requires_both_quality_and_growth_inputs(self):
        missing_growth = row()
        missing_growth["growth_score"] = None
        self.assertEqual(build_question_views([missing_growth])["cheap_with_potential"], [])

    def test_cheap_ranks_discount_potential_over_risk(self):
        first = row("FIRST", price=60)
        second = row("SECOND", price=70)
        result = build_question_views([second, first])
        self.assertEqual(result["cheap_with_potential"][0]["symbol"], "FIRST")
        self.assertGreaterEqual(result["cheap_with_potential"][0]["attractiveness_score"], 0)

    def test_all_eligible_rows_are_returned_without_hard_limit(self):
        rows = [row(f"C{index}", price=50 + index) for index in range(10)]
        result = build_question_views(rows)
        self.assertEqual(len(result["cheap_with_potential"]), 10)
        self.assertEqual(result["selection_counts"]["visible"], 10)

    def test_expensive_is_sorted_by_premium_without_warning_score(self):
        high = row("HIGH", verdict="overpriced", price=220, lower=80, upper=100)
        low = row("LOW", verdict="overpriced", price=150, lower=80, upper=100)
        result = build_question_views([low, high])
        self.assertEqual([item["symbol"] for item in result["expensive_now"]], ["HIGH", "LOW"])
        self.assertNotIn("score", result["expensive_now"][0])
        self.assertEqual(result["expensive_now"][0]["badge"]["label"], "Kein Setup")

    def test_expensive_requires_constructive_technical_timing(self):
        cold = row("COLD", verdict="overpriced", price=220, lower=80, upper=100)
        cold["entry_timing_score"] = 20
        self.assertEqual(build_question_views([cold])["expensive_now"], [])

    def test_deal_quality_explains_available_comparison_basis(self):
        result = build_question_views(
            [row()],
            historical_deal_scores={
                "scores": [20, 40, 60],
                "snapshot_count": 2,
                "from_date": "2026-08-30",
                "to_date": "2026-08-31",
            },
        )
        quality = result["cheap_with_potential"][0]["deal_quality"]
        self.assertIn("Deal-Qualität", quality["label"])
        self.assertIn("3 Gelegenheiten seit 30.08.2026", quality["comparison_basis"])
        self.assertIn("belastbar ab", quality["comparison_basis"])
        self.assertFalse(quality["history_reliable"])

    def test_near_triggers_are_exposed_for_today_summary(self):
        candidate = row()
        candidate["price_local"] = 50.05
        candidate["fx_usd"] = 1
        candidate["sweet_spot"] = {
            "combined_status": "approaching",
            "lower": 45,
            "ideal": 48,
            "upper": 50,
        }
        result = build_question_views([candidate])
        self.assertEqual(
            [item["symbol"] for item in result["near_triggers"]],
            ["AAA"],
        )

    def test_triggered_today_is_not_also_exposed_as_near_trigger(self):
        candidate = row(price=49.95)
        candidate["fx_usd"] = 1
        candidate["sweet_spot"] = {
            "combined_status": "approaching",
            "lower": 45,
            "ideal": 48,
            "upper": 50,
        }
        previous = {"all": [{"symbol": "AAA", "price_local": 50.1}]}
        result = build_question_views([candidate], previous_snapshot=previous)
        self.assertEqual(
            [item["symbol"] for item in result["triggered_today"]],
            ["AAA"],
        )
        self.assertEqual(result["near_triggers"], [])

    def test_non_us_origin_does_not_reduce_quality_label(self):
        eu = row()
        eu["expert_analysis"]["coverage_basis"]["status"] = "narrower_non_us"
        item = build_question_views([eu])["cheap_with_potential"][0]
        self.assertEqual(item["basis"], "breit")
        self.assertEqual(item["geographic_coverage"], "narrower_non_us")

    def test_narrow_basis_is_excluded(self):
        narrow = row()
        narrow["expert_analysis"]["valuation"]["basis_quality"]["status"] = "narrow"
        self.assertEqual(build_question_views([narrow])["cheap_with_potential"], [])

    def test_question_view_uses_valuation_status_not_actionable(self):
        result = build_question_views([row()])
        self.assertNotIn("actionable", result)
        self.assertEqual(result["valuation_status"], "evidence_qualified_unbacktested")
        self.assertIn("noch nicht rückgeprüft", result["valuation_status_label"])


if __name__ == "__main__":
    unittest.main()
