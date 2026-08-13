import json
import tempfile
import unittest
from pathlib import Path

from src.export_static import DEFAULT_INPUT, export_static


class StaticExportTests(unittest.TestCase):
    def test_exports_valid_compact_dashboard_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data.json"
            payload = export_static(DEFAULT_INPUT, output)
            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual("stock-radar-static", loaded["schema"])
        self.assertEqual(1, loaded["schema_version"])
        self.assertEqual(len(payload["instruments"]), len(loaded["instruments"]))
        self.assertGreater(len(loaded["instruments"]), 1000)
        self.assertIn("USD", loaded["rankings"])
        self.assertNotIn("trade_plan_long", loaded["instruments"][0])


if __name__ == "__main__":
    unittest.main()
