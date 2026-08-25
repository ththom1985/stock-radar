import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import recommendation_journal as journal


class RecommendationJournalTests(unittest.TestCase):
    def test_records_once_and_appends_mature_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "recommendations.jsonl"
            outcome_path = Path(directory) / "outcomes.jsonl"
            rows = [
                {
                    "symbol": "AAA",
                    "name": "AAA",
                    "bar_date": "2026-01-02",
                    "price_local": 100.0,
                    "currency": "USD",
                    "expert_analysis": {
                        "signal": "positive_setup",
                        "evidence_quality": "medium",
                        "long_term": {
                            "score": 80,
                            "coverage_pct": 75,
                            "components": {"value": {"value": 80}},
                        },
                    },
                }
            ]
            rankings = {
                "long_term": [{"symbol": "AAA"}],
                "short_term": [],
            }
            dates = pd.date_range("2026-01-02", periods=30, freq="B")
            prices = pd.DataFrame(
                {"RawClose": [100.0 + index for index in range(30)]},
                index=dates,
            )
            with patch.object(journal, "LOG_PATH", log_path), patch.object(
                journal, "OUTCOME_PATH", outcome_path
            ):
                first = journal.record_top_observations(
                    rows, rankings, "2026-01-02T23:00:00+00:00"
                )
                second = journal.record_top_observations(
                    rows, rankings, "2026-01-02T23:00:00+00:00"
                )
                outcomes = journal.evaluate_mature_observations({"AAA": prices})
                summary = journal.journal_summary()
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]["horizon"], "1m")
            self.assertTrue(outcomes[0]["positive"])
            self.assertEqual(summary["observation_count"], 1)
            self.assertEqual(summary["by_horizon"]["1m"]["hit_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
