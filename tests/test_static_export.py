import json
import unittest
from pathlib import Path

from src.export_static import (
    DEFAULT_INPUT,
    MAX_STATIC_BYTES,
    STATIC_SCHEMA_VERSION,
    TARGET_STATIC_BYTES,
    export_static,
    validate_static_payload,
)
from src.data_quality import FORBIDDEN_RESEARCH_PHRASES
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
        for field in (
            "display_name_full",
            "headquarters_country",
            "legal_domicile",
            "economic_exposure_country",
            "listing_country",
            "industry_display",
            "jurisdiction_risk",
            "valuation_thesis",
            "entry_thesis",
        ):
            self.assertIn(field, loaded["instruments"][0])
        self.assertNotIn("trade_plan_long", loaded["instruments"][0])
        self.assertNotIn("identity_source", loaded["instruments"][0])
        self.assertNotIn("model_status", loaded["instruments"][0]["entry_thesis"])
        self.assertEqual(
            loaded["instrument_contract"]["group_provenance"],
            "insight_metadata.provenance_catalog",
        )
        self.assertEqual(raw, expected)
        self.assertEqual(size, len(expected))
        self.assertLess(size, MAX_STATIC_BYTES)
        self.assertLessEqual(size, TARGET_STATIC_BYTES)
        self.assertEqual(MAX_STATIC_BYTES, 10 * 1024 * 1024)
        self.assertEqual(TARGET_STATIC_BYTES, int(8.5 * 1024 * 1024))

        source = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        source_by_symbol = {row["symbol"]: row for row in source["all"]}
        compact_by_symbol = {row["symbol"]: row for row in loaded["instruments"]}
        for symbol in ("PDD", "AAPL", "SHEL.L"):
            source_row = source_by_symbol[symbol]
            compact_row = compact_by_symbol[symbol]
            for field in (
                "display_name_full",
                "headquarters_country",
                "legal_domicile",
                "economic_exposure_country",
                "economic_exposure_region",
                "listing_country",
                "listing_market",
                "industry_display",
                "sector_display",
                "rsi",
                "macd",
                "macd_signal",
                "ret_20d",
                "ret_60d",
                "atr_pct",
                "vol_annual_pct",
                "next_earnings",
                "earnings_in_days",
            ):
                self.assertEqual(compact_row[field], source_row[field])
            for key in (
                "why_it_looks_cheap",
                "why_discount_may_be_justified",
                "strongest_positive_evidence",
                "strongest_counterarguments",
                "penalty_reasons",
                "penalty_evidence_ids",
            ):
                self.assertEqual(
                    compact_row["valuation_thesis"][key],
                    source_row["valuation_thesis"][key],
                )
            for key in (
                "why_timing_may_be_good",
                "what_confirms",
                "what_invalidates",
                "strongest_supporting_evidence",
                "strongest_counterarguments",
            ):
                self.assertEqual(
                    compact_row["entry_thesis"][key],
                    source_row["entry_thesis"][key],
                )
            self.assertEqual(
                compact_row["jurisdiction_risk"]["reasons"],
                source_row["jurisdiction_risk"]["reasons"],
            )
            self.assertEqual(
                len(compact_row["scenario_long"]),
                min(4, len(source_row["scenario_long"])),
            )
            self.assertEqual(
                len(compact_row["news"]),
                min(3, len(source_row["news"])),
            )
        generated = json.dumps(
            [
                {
                    key: row.get(key)
                    for key in (
                        "research_summary",
                        "research_actions",
                        "entry_timing_reason",
                        "entry_thesis",
                        "valuation_thesis",
                        "longterm_reasons",
                        "daily_signal_reasons",
                        "weinstein_label",
                        "trend_phase",
                    )
                }
                for row in loaded["instruments"]
            ],
            ensure_ascii=False,
        ).casefold()
        for phrase in FORBIDDEN_RESEARCH_PHRASES:
            self.assertNotIn(phrase, generated)

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

        payload = export_static(DEFAULT_INPUT, self.work / "nested-row.json")
        payload["instruments"][0]["entry_thesis"]["actionable"] = True
        with self.assertRaises(ValueError):
            validate_static_payload(payload)

    def test_pages_cockpit_exposes_required_navigation_and_single_detail(self):
        html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        for label in (
            "Tages-Setups",
            "Unterbewertet",
            "Potenzial",
            "Einstiegs-Timing",
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
        self.assertIn("instrumentContract.actionable !== false", html)
        self.assertIn("state.data.schema_version !== 3", html)
        self.assertIn("hasActionableTrue(data?.instruments || [])", html)
        self.assertIn("China-Risikokontext", html)
        self.assertIn("Warum es günstig aussieht", html)
        self.assertIn("Benötigte Bestätigung", html)
        self.assertIn("Vollständiger Name", html)
        self.assertIn("Hauptsitz (Provider)", html)
        self.assertIn("Juristischer Sitz (verifiziert)", html)
        self.assertIn("Börsenland/Markt", html)
        self.assertIn('button.disabled = button.dataset.view !== "health"', html)


if __name__ == "__main__":
    unittest.main()
