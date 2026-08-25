import unittest
from datetime import date

from src.job_postings import summarize_job_history


class JobPostingTests(unittest.TestCase):
    def test_collects_history_before_scoring(self):
        result = summarize_job_history(
            [{"date": "2026-08-25", "count": 100}],
            today=date(2026, 8, 25),
        )
        self.assertEqual(result["status"], "collecting_history")
        self.assertNotIn("score", result)

    def test_scores_change_after_seven_days(self):
        result = summarize_job_history(
            [
                {"date": "2026-08-15", "count": 100},
                {"date": "2026-08-25", "count": 120},
            ],
            today=date(2026, 8, 25),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["change_pct"], 20.0)
        self.assertGreater(result["score"], 50)


if __name__ == "__main__":
    unittest.main()
