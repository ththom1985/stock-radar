import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.opportunity_history import update_opportunity_history


class OpportunityHistoryTests(unittest.TestCase):
    def test_replaces_same_day_and_flattens_rolling_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            with patch("src.opportunity_history.HISTORY_PATH", path):
                first = update_opportunity_history(
                    {
                        "cheap_with_potential": [
                            {"deal_quality": {"score": 60}},
                            {"deal_quality": {"score": 70}},
                        ]
                    },
                    datetime(2026, 8, 31, tzinfo=timezone.utc),
                )
                second = update_opportunity_history(
                    {
                        "cheap_with_potential": [
                            {"deal_quality": {"score": 80}},
                        ]
                    },
                    datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
                )
        self.assertEqual(first["scores"], [60, 70])
        self.assertEqual(second["scores"], [80])
        self.assertEqual(second["snapshot_count"], 1)
        self.assertEqual(second["calendar_days"], 1)
        self.assertFalse(second["reliable"])
        self.assertIn("100 Gelegenheiten", second["reliability_requirement"])


if __name__ == "__main__":
    unittest.main()
