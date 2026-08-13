import json
import unittest
from pathlib import Path

from src.export_static import (
    DEFAULT_INPUT,
    MAX_STATIC_BYTES,
    STATIC_SCHEMA_VERSION,
    export_static,
    validate_static_payload,
)
from tests.helpers import ProjectTempMixin
from tests.helpers import ROOT


class StaticExportTests(ProjectTempMixin, unittest.TestCase):
    def test_exports_valid_compact_dashboard_payload(self):
        output = self.work / "data.json"
        payload = export_static(DEFAULT_INPUT, output)
        raw = output.read_bytes()
        loaded = json.loads(raw.decode("utf-8"))
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        size = len(raw)

        self.assertEqual("stock-radar-static", loaded["schema"])
        self.assertEqual(STATIC_SCHEMA_VERSION, loaded["schema_version"])
        self.assertEqual(len(payload["instruments"]), len(loaded["instruments"]))
        self.assertGreater(len(loaded["instruments"]), 1000)
        self.assertIn("USD", loaded["rankings"])
        self.assertIn("daily_setups", loaded["insight_rankings"]["categories"])
        self.assertIn("entry_timing_score", loaded["instruments"][0])
        self.assertIn("downside_structure", loaded["instruments"][0])
        self.assertIn("insight_provenance", loaded["instruments"][0])
        self.assertNotIn("trade_plan_long", loaded["instruments"][0])
        self.assertEqual(raw, expected)
        self.assertEqual(size, len(expected))
        self.assertLess(size, MAX_STATIC_BYTES)

    def test_static_validator_rejects_missing_insight_contract(self):
        with self.assertRaises(ValueError):
            validate_static_payload(
                {
                    "schema": "stock-radar-static",
                    "schema_version": STATIC_SCHEMA_VERSION,
                    "rankings": {},
                    "instruments": [],
                }
            )

    def test_static_validator_rejects_nested_actionable_true(self):
        payload = export_static(DEFAULT_INPUT, self.work / "nested.json")
        payload["insight_metadata"]["nested"] = {"actionable": True}
        with self.assertRaises(ValueError):
            validate_static_payload(payload)

    def test_pages_cockpit_exposes_required_navigation_and_single_detail(self):
        html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        for label in (
            "Tipps des Tages",
            "Unterbewertet",
            "Potenzial",
            "Guter Einstieg",
            "Fallende Messer",
            "Bodenbildung",
            "Alle suchen",
            "Datenqualität",
        ):
            self.assertIn(label, html)
        self.assertIn("--cp-success", html)
        self.assertIn("--cp-warning", html)
        self.assertIn("--cp-danger", html)
        self.assertEqual(html.count('id="detail"'), 1)
        self.assertIn("const MAX_OUTPUT_AGE_HOURS = 36", html)
        self.assertIn('status.status !== "ok"', html)
        self.assertIn("status.data_actionable !== true", html)
        self.assertIn("(status.blocking_reasons || []).length", html)
        self.assertIn("ageHours > MAX_OUTPUT_AGE_HOURS", html)
        self.assertIn("insights.actionable !== false", html)
        self.assertIn('button.disabled = button.dataset.view !== "health"', html)


if __name__ == "__main__":
    unittest.main()
