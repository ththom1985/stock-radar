import unittest

from src.wikipedia_attention import (
    _cached_article_valid,
    select_article,
    summarize_pageviews,
)


class WikipediaAttentionTests(unittest.TestCase):
    def test_scores_trend_not_absolute_pageviews(self):
        items = [
            {"timestamp": f"202601{index + 1:02d}", "views": 100}
            for index in range(28)
        ] + [
            {"timestamp": f"202602{index + 1:02d}", "views": 150}
            for index in range(7)
        ]
        result = summarize_pageviews(items)
        self.assertEqual(result["trend_change_pct"], 50.0)
        self.assertGreater(result["score"], 50)

    def test_insufficient_history_is_unavailable(self):
        self.assertIsNone(
            summarize_pageviews([{"timestamp": "20260101", "views": 100}])
        )

    def test_article_resolution_rejects_zero_overlap(self):
        self.assertIsNone(
            select_article(
                "Krystal Biotech Inc.",
                [{"title": "Beremagene geperpavec"}],
            )
        )
        self.assertEqual(
            select_article("Apple Inc.", [{"title": "Apple Inc."}]),
            "Apple Inc.",
        )

    def test_invalid_cached_article_is_reprocessed(self):
        self.assertFalse(
            _cached_article_valid(
                {"display_name_full": "Krystal Biotech Inc."},
                {"status": "ok", "article": "Beremagene geperpavec"},
            )
        )


if __name__ == "__main__":
    unittest.main()
