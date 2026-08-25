import unittest

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
                "own_5y_complete": True,
            },
            "risks": {"top_risks": ["Kein einzelnes dominantes Risiko."]},
        },
    }


class QuestionViewsTests(unittest.TestCase):
    def test_cheap_requires_gate_and_excludes_falling_knives(self):
        valid = row()
        withheld = row("WITHHELD")
        withheld["expert_analysis"]["valuation"]["plausibility_gate"]["status"] = "withheld_extreme_deviation"
        knife = row("KNIFE")
        knife["falling_knife"] = {"active": True}
        result = build_question_views([valid, withheld, knife])
        self.assertEqual([item["symbol"] for item in result["cheap_with_potential"]], ["AAA"])

    def test_cheap_ranks_discount_potential_over_risk(self):
        first = row("FIRST", price=60)
        second = row("SECOND", price=70)
        result = build_question_views([second, first])
        self.assertEqual(result["cheap_with_potential"][0]["symbol"], "FIRST")
        self.assertGreaterEqual(result["cheap_with_potential"][0]["attractiveness_score"], 0)

    def test_expensive_is_sorted_by_premium_without_warning_score(self):
        high = row("HIGH", verdict="overpriced", price=220, lower=80, upper=100)
        low = row("LOW", verdict="overpriced", price=150, lower=80, upper=100)
        result = build_question_views([low, high])
        self.assertEqual([item["symbol"] for item in result["expensive_now"]], ["HIGH", "LOW"])
        self.assertNotIn("score", result["expensive_now"][0])
        self.assertEqual(result["expensive_now"][0]["badge"]["label"], "Kein Setup")

    def test_non_us_basis_is_explicit(self):
        eu = row()
        eu["expert_analysis"]["coverage_basis"]["status"] = "narrower_non_us"
        self.assertEqual(build_question_views([eu])["cheap_with_potential"][0]["basis"], "EU-renormalisiert")


if __name__ == "__main__":
    unittest.main()
