import os
import unittest
from unittest.mock import patch

from src.finra_short_interest import credential_status, summarize_records


class FinraShortInterestTests(unittest.TestCase):
    def test_missing_credentials_are_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            status = credential_status()
        self.assertFalse(status["configured"])
        self.assertEqual(
            status["missing"], ["FINRA_CLIENT_ID", "FINRA_CLIENT_SECRET"]
        )

    def test_summarizes_trend_days_to_cover_and_revision(self):
        records = [
            {
                "settlementDate": "2026-07-15",
                "currentShortPositionQuantity": 100,
                "previousShortPositionQuantity": 90,
                "averageDailyVolumeQuantity": 50,
                "daysToCoverQuantity": 2,
                "changePercent": 11.11,
                "revisionFlag": None,
            },
            {
                "settlementDate": "2026-08-14",
                "currentShortPositionQuantity": 150,
                "previousShortPositionQuantity": 100,
                "averageDailyVolumeQuantity": 30,
                "daysToCoverQuantity": 5,
                "changePercent": 50,
                "revisionFlag": "R",
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["settlement_date"], "2026-08-14")
        self.assertEqual(summary["days_to_cover"], 5)
        self.assertEqual(summary["period_trend_pct"], 50)
        self.assertLess(summary["score"], 50)
        self.assertTrue(summary["periods"][-1]["revision"])

    def test_empty_records_are_unavailable(self):
        self.assertIsNone(summarize_records([]))


if __name__ == "__main__":
    unittest.main()
