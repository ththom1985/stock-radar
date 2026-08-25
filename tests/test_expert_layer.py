import tempfile
import unittest
from pathlib import Path

from src.expert_layer import (
    DEFAULT_WEIGHTS,
    attach_expert_analysis,
    build_expert_analysis,
    build_expert_rankings,
    load_score_weights,
)


class ExpertLayerTests(unittest.TestCase):
    def test_scores_available_components_without_inventing_missing_data(self):
        row = {
            "value_score": 80,
            "quality_score": 70,
            "growth_score": 60,
            "longterm_score": 75,
            "rs_rating": 65,
            "entry_timing_score": 72,
            "tech_momentum": 68,
            "tech_trend": 70,
            "tech_volume": 62,
            "news_score": 55,
            "analyst_mean": 2,
            "analyst_n": 8,
            "sweet_spot": {"combined_status": "in_zone_confirmed"},
        }
        result = build_expert_analysis(row, DEFAULT_WEIGHTS)
        self.assertFalse(result["actionable"])
        self.assertEqual(result["model_status"], "heuristic_unvalidated")
        self.assertIsNotNone(result["long_term"]["score"])
        self.assertIn(
            "alternative_data", result["long_term"]["missing_components"]
        )
        self.assertLess(result["long_term"]["coverage_pct"], 100)

    def test_invalid_weight_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(
                '{"long_term":{"value":100},"short_term":{"valuation":100}}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_score_weights(path)

    def test_rankings_require_minimum_weight_coverage(self):
        rows = [
            {
                "symbol": "AAA",
                "name": "AAA",
                "currency": "USD",
                "value_score": 90,
                "quality_score": 90,
                "growth_score": 90,
                "longterm_score": 90,
                "rs_rating": 90,
                "news_score": 90,
                "analyst_mean": 1.5,
                "analyst_n": 10,
                "entry_timing_score": 90,
                "tech_momentum": 90,
                "tech_trend": 90,
                "tech_volume": 90,
            },
            {"symbol": "EMPTY", "name": "Empty", "currency": "USD"},
        ]
        attach_expert_analysis(rows, DEFAULT_WEIGHTS)
        rankings = build_expert_rankings(rows)
        self.assertEqual([item["symbol"] for item in rankings["long_term"]], ["AAA"])
        self.assertEqual([item["symbol"] for item in rankings["short_term"]], ["AAA"])

    def test_non_us_basis_is_narrower_and_renormalized(self):
        result = build_expert_analysis(
            {
                "listing_country": "Germany",
                "value_score": 70,
                "quality_score": 70,
                "growth_score": 70,
                "longterm_score": 70,
            },
            DEFAULT_WEIGHTS,
        )
        basis = result["coverage_basis"]
        self.assertEqual(basis["status"], "narrower_non_us")
        self.assertTrue(basis["weights_renormalized"])
        self.assertIn("SEC 13F", basis["structurally_unavailable"])


if __name__ == "__main__":
    unittest.main()
