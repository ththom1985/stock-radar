import unittest
from datetime import datetime, timedelta, timezone

from src.automation_guard import recovery_needed
from src.verify_live import live_matches


class AutomationGuardTests(unittest.TestCase):
    def test_fresh_matching_live_payload_skips_recovery(self):
        now = datetime(2026, 9, 3, 6, tzinfo=timezone.utc)
        generated = (now - timedelta(hours=7)).isoformat()
        rebuild, _ = recovery_needed(
            {"generated_at": generated},
            {"generated_at": generated},
            {"generated_at": generated},
            now=now,
            max_age_hours=12,
        )
        self.assertFalse(rebuild)

    def test_stale_snapshot_requires_recovery(self):
        now = datetime(2026, 9, 3, 6, tzinfo=timezone.utc)
        generated = (now - timedelta(hours=31)).isoformat()
        rebuild, reason = recovery_needed(
            {"generated_at": generated},
            {"generated_at": generated},
            {"generated_at": generated},
            now=now,
            max_age_hours=12,
        )
        self.assertTrue(rebuild)
        self.assertIn("31.0 hours", reason)

    def test_live_mismatch_requires_recovery(self):
        now = datetime(2026, 9, 3, 6, tzinfo=timezone.utc)
        generated = (now - timedelta(hours=7)).isoformat()
        older = (now - timedelta(hours=8)).isoformat()
        rebuild, _ = recovery_needed(
            {"generated_at": generated},
            {"generated_at": generated},
            {"generated_at": older},
            now=now,
            max_age_hours=12,
        )
        self.assertTrue(rebuild)

    def test_matching_but_stale_export_and_live_require_recovery(self):
        now = datetime(2026, 9, 3, 6, tzinfo=timezone.utc)
        fresh = (now - timedelta(hours=7)).isoformat()
        stale = (now - timedelta(hours=31)).isoformat()
        rebuild, reason = recovery_needed(
            {"generated_at": fresh},
            {"generated_at": stale},
            {"generated_at": stale},
            now=now,
            max_age_hours=12,
        )
        self.assertTrue(rebuild)
        self.assertIn("timestamps differ", reason)

    def test_malformed_payload_is_rejected(self):
        now = datetime(2026, 9, 3, 6, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "root is not an object"):
            recovery_needed(
                {"generated_at": now.isoformat()},
                {},
                [],
                now=now,
                max_age_hours=12,
            )
        with self.assertRaisesRegex(ValueError, "roots must be objects"):
            live_matches({}, [])

    def test_live_verification_checks_timestamp_and_instrument_count(self):
        expected = {"generated_at": "now", "instruments": [{}, {}]}
        self.assertEqual(
            live_matches(expected, expected),
            (True, "live payload matches now with 2 instruments"),
        )
        matched, reason = live_matches(
            expected, {"generated_at": "now", "instruments": [{}]}
        )
        self.assertFalse(matched)
        self.assertIn("instrument count", reason)
        matched, reason = live_matches(
            expected,
            {
                "generated_at": "now",
                "instruments": [{}, {}],
                "unexpected": True,
            },
        )
        self.assertFalse(matched)
        self.assertIn("content differs", reason)


if __name__ == "__main__":
    unittest.main()
