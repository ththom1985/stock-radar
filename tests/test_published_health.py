import unittest

from src.published_health import classify_unavailable


class PublishedHealthTests(unittest.TestCase):
    def test_classifies_missing_price_without_using_count_thresholds(self):
        self.assertEqual(
            classify_unavailable(
                {
                    "symbol": "AAA",
                    "price": None,
                    "atr": 2.0,
                    "feature_coverage": {"technical_complete": False},
                }
            ),
            "missing_or_nonpositive_price",
        )
        self.assertEqual(
            classify_unavailable(
                {
                    "price": 10.0,
                    "atr": 1.0,
                    "feature_coverage": {"technical_complete": True},
                }
            ),
            "no_eligible_reference_anchor",
        )


if __name__ == "__main__":
    unittest.main()
