import json
import unittest
from pathlib import Path

from src.export_static import (
    DEFAULT_INPUT,
    MAX_STATIC_BYTES,
    STATIC_SCHEMA_VERSION,
    TARGET_STATIC_BYTES,
    _hydrate_compact_probability,
    _hydrate_compact_sweet,
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
        self.assertIn("in_sweet_spot", loaded["insight_rankings"]["categories"])
        self.assertIn("approaching_sweet_spot", loaded["insight_rankings"]["categories"])
        self.assertIn("entry_timing_score", loaded["instruments"][0])
        self.assertIn("downside_structure", loaded["instruments"][0])
        self.assertIn("pf", loaded["instruments"][0])
        self.assertEqual(
            loaded["probability_contract"]["ranking_separation"],
            "probability fields are excluded from radar scores, insight rankings, "
            "Sweet Spot, and colors",
        )
        forward = loaded["forward_validation_status"]
        self.assertEqual(forward["retrospective_status"], "rejected")
        self.assertFalse(forward["actionable"])
        self.assertFalse(forward["shadow_values_published"])
        self.assertEqual(forward["detail_store"], "machine_local_only")
        forward_text = json.dumps(forward, sort_keys=True).casefold()
        for forbidden in (
            "raw_ordered_probabilities",
            "derived_probabilities",
            "feature_vector",
            "coefficient",
            "entry_open",
            "gross_return",
        ):
            self.assertNotIn(forbidden, forward_text)
        probability = _hydrate_compact_probability(
            loaded["instruments"][0]["pf"],
            reason_catalog=loaded["probability_reason_catalog"],
            model_catalog=loaded["probability_model_catalog"],
            baseline_catalog=loaded["probability_baseline_catalog"],
            contract=loaded["probability_contract"],
            listing_currency=loaded["instruments"][0]["currency"],
            signal_timestamp=loaded["instruments"][0]["bar_date"],
        )
        self.assertEqual(probability["status"], "withheld")
        self.assertEqual(
            probability["message"],
            "No validated stock-specific probability edge",
        )
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
            "sweet_spot",
        ):
            self.assertIn(field, loaded["instruments"][0])
        self.assertNotIn("trade_plan_long", loaded["instruments"][0])
        self.assertNotIn("identity_source", loaded["instruments"][0])
        self.assertNotIn("model_status", loaded["instruments"][0]["entry_thesis"])
        self.assertEqual(
            loaded["instrument_contract"]["group_provenance"],
            "insight_metadata.provenance_catalog",
        )
        self.assertEqual(
            loaded["sweet_spot_contract"]["model_status"],
            "heuristic_unvalidated",
        )
        self.assertFalse(loaded["sweet_spot_contract"]["actionable"])
        self.assertIsInstance(loaded["sweet_spot_reason_catalog"], list)
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
            sweet = compact_row["sweet_spot"]
            source_sweet = source_row["sweet_spot"]
            hydrated_sweet = _hydrate_compact_sweet(
                sweet,
                reason_catalog=loaded["sweet_spot_reason_catalog"],
                current_price=compact_row["price"],
            )
            self.assertEqual(
                hydrated_sweet["combined_status"],
                source_sweet["combined_status"],
            )
            self.assertEqual(hydrated_sweet["tone"], source_sweet["tone"])
            self.assertEqual(hydrated_sweet["components"], source_sweet["components"])
            if hydrated_sweet["available"]:
                self.assertLess(
                    hydrated_sweet["lower"],
                    hydrated_sweet["ideal"],
                )
                self.assertLess(
                    hydrated_sweet["ideal"],
                    hydrated_sweet["upper"],
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

    def test_static_validator_recomputes_green_invariants(self):
        payload = export_static(DEFAULT_INPUT, self.work / "green-invariants.json")
        row = next(
            item
            for item in payload["instruments"]
            if (item.get("sweet_spot") or {}).get("combined_status")
            == "in_zone_confirmed"
            and item.get("asset_type") == "company_equity"
        )
        gate_fields = payload["sweet_spot_contract"]["gate_evidence_fields"]
        gate_index = {name: index for index, name in enumerate(gate_fields)}
        pivot_family_ref = payload["sweet_spot_contract"]["source_families"].index(
            "pivot"
        )

        def evidence(name, value):
            row["sweet_spot"]["gate_evidence"][gate_index[name]] = value

        original = json.loads(json.dumps(row))
        mutations = {
            "outside": lambda: row.update(
                {"price": row["sweet_spot"]["upper"] + 1}
            ),
            "stage4": lambda: evidence("weinstein_stage", 4),
            "timing": lambda: row.update({"entry_timing_score": 54}),
            "macd": lambda: (
                evidence("macd_hist", -1.0),
                evidence("macd_hist_prev", 0.0),
                evidence("atr", 1.0),
            ),
            "stale": lambda: evidence("bar_age_days", 5),
            "reliability": lambda: row["sweet_spot"].update(
                {"reliability_score": 64.0}
            ),
            "families": lambda: (
                [
                    component.__setitem__(1, pivot_family_ref)
                    for component in row["sweet_spot"]["components"]
                ],
                row["sweet_spot"].update({"independent_family_count": 1}),
            ),
            "investor": lambda: row["jurisdiction_risk"].update({"level": "high"}),
        }
        row_index = payload["instruments"].index(row)
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload["instruments"][row_index] = json.loads(json.dumps(original))
                row = payload["instruments"][row_index]
                mutate()
                with self.assertRaises(ValueError):
                    validate_static_payload(payload)
        payload["instruments"][row_index] = original
        validate_static_payload(payload)

    def test_static_reference_only_cannot_claim_or_rank_as_green(self):
        payload = export_static(DEFAULT_INPUT, self.work / "reference-only.json")
        reference_tier_ref = payload["sweet_spot_contract"]["zone_tiers"].index(
            "reference_only"
        )
        row = next(
            item
            for item in payload["instruments"]
            if (item.get("sweet_spot") or {}).get("zone_tier_ref")
            == reference_tier_ref
        )
        original = json.loads(json.dumps(row))
        row["sweet_spot"].update(
            {
                "combined_status": "in_zone_confirmed",
                "technical_status": "in_zone_confirmed",
                "tone": "green",
            }
        )
        with self.assertRaises(ValueError):
            validate_static_payload(payload)

        row.update(original)
        categories = payload["insight_rankings"]["categories"]
        source_item = next(
            item
            for items in categories["approaching_sweet_spot"][
                "items_by_currency"
            ].values()
            for item in items
        )
        leaked = json.loads(json.dumps(source_item))
        leaked["symbol"] = row["symbol"]
        currency = row["currency"]
        categories["approaching_sweet_spot"]["items_by_currency"].setdefault(
            currency,
            [],
        ).append(leaked)
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
            "Sweet Spot",
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
        self.assertIn("state.selected !== row.symbol", html)
        self.assertIn("localValue(sweet.lower)", html)
        self.assertIn("Für Titel außerhalb USD", html)
        self.assertIn("Wahrscheinlichkeits-Selbstprüfung", html)
        self.assertIn("hydrateProbabilityForecasts", html)
        self.assertIn("Probability-Validierung", html)
        self.assertIn("Forward Validation", html)
        self.assertIn("baseline-only", html)
        self.assertIn("meaningful_1m_assessment_not_before", html)
        dashboard = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        for forbidden in (
            "will rise",
            "guaranteed",
            "expected price",
            "confidence",
            "wird steigen",
            "garantiert",
            "erwarteter preis",
        ):
            self.assertNotIn(forbidden, (html + dashboard).casefold())
        self.assertIn("hasActionableTrue(data?.instruments || [])", html)
        self.assertIn("China-Risikokontext", html)
        self.assertIn("Warum es günstig aussieht", html)
        self.assertIn("Benötigte Bestätigung", html)
        self.assertIn("Sweet-Spot-Beobachtungszone", html)
        self.assertIn("technische Einstiegsbeobachtung", html)
        self.assertIn("Beobachtungszone, keine Ordermarke", html)
        self.assertIn("IDEAL", html)
        self.assertIn("hydrateSweetSpotReasons", html)
        self.assertIn("function priceDigits(values)", html)
        self.assertIn("const $price =", html)
        self.assertIn("$price(zoneValues[0], zoneValues)", html)
        self.assertIn("localValue(item.value)", html)
        self.assertIn("var(--cp-success)", html)
        self.assertIn("var(--cp-warning)", html)
        self.assertIn("var(--cp-danger)", html)
        self.assertIn("Vollständiger Name", html)
        self.assertIn("Hauptsitz (Provider)", html)
        self.assertIn("Juristischer Sitz (verifiziert)", html)
        self.assertIn("Börsenland/Markt", html)
        self.assertIn('button.disabled = button.dataset.view !== "health"', html)


if __name__ == "__main__":
    unittest.main()
