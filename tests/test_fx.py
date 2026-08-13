from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import src.fx as fx
from src.persistence import atomic_write_json
from tests.helpers import ProjectTempMixin


class FXTests(ProjectTempMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.original = fx.FX_CACHE
        fx.FX_CACHE = self.work / "fx.json"

    def tearDown(self):
        fx.FX_CACHE = self.original
        super().tearDown()

    def test_refresh_failure_preserves_stale_good_rate(self):
        atomic_write_json(
            fx.FX_CACHE,
            {
                "rates": {
                    "EUR": {
                        "rate": 1.2,
                        "fetched_at": "2020-01-01T00:00:00+00:00",
                        "status": "fresh",
                    }
                }
            },
        )
        with patch.object(fx, "_pair_rate", side_effect=fx.FXUnavailableError("offline")):
            result = fx.get_fx_rates_with_status(
                {"EUR"}, now=datetime(2026, 8, 12, tzinfo=timezone.utc), verbose=False
            )
        self.assertEqual(result.rates["EUR"], 1.2)
        self.assertEqual(result.status["EUR"]["status"], "stale")

    def test_missing_rate_never_becomes_one_to_one(self):
        with patch.object(fx, "_pair_rate", side_effect=fx.FXUnavailableError("offline")):
            result = fx.get_fx_rates_with_status({"JPY"}, verbose=False)
        self.assertNotIn("JPY", result.rates)
        self.assertIn("JPY", result.missing)
        with patch.object(fx, "_pair_rate", side_effect=fx.FXUnavailableError("offline")):
            with self.assertRaises(fx.FXUnavailableError):
                fx.get_fx_rates({"JPY"}, verbose=False)

    def test_legacy_usd_is_unconditionally_canonicalized(self):
        atomic_write_json(
            fx.FX_CACHE,
            {
                "_fetched_at": "2020-01-01T00:00:00+00:00",
                "USD": 0.75,
            },
        )
        result = fx.get_fx_rates_with_status({"USD"}, verbose=False)
        self.assertEqual(result.rates["USD"], 1.0)
        self.assertEqual(result.status["USD"]["status"], "fixed")
        self.assertEqual(result.status["USD"]["source"], "definition")

    def test_console_encoding_failure_cannot_turn_successful_fx_stale(self):
        console_error = UnicodeEncodeError(
            "charmap",
            "x",
            0,
            1,
            "character maps to undefined",
        )
        with patch("builtins.print", side_effect=console_error), patch.object(
            fx, "_pair_rate", return_value=1.2345
        ) as provider:
            result = fx.get_fx_rates_with_status({"EUR"}, verbose=True)
        provider.assert_called_once_with("EUR")
        self.assertEqual(result.rates["EUR"], 1.2345)
        self.assertEqual(result.status["EUR"]["status"], "fresh")
        self.assertNotIn("last_failure", result.status["EUR"])


if __name__ == "__main__":
    unittest.main()
