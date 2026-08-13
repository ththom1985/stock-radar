from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.data_quality import (
    FORBIDDEN_RESEARCH_PHRASES,
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    DataContractError,
    validate_insight_contract,
)
from src.insights import (
    build_insight_rankings,
    enrich_row,
    enrich_rows_and_rankings,
    enrich_snapshot,
)
from src.export_static import _compact_row
from src.identity import normalize_country


def base_row(symbol="AAA"):
    return {
        "symbol": symbol,
        "name": f"{symbol} AG",
        "asset_type": "company_equity",
        "currency": "USD",
        "price": 100.0,
        "price_local": 100.0,
        "sma20": 98.0,
        "sma50": 95.0,
        "sma150": 90.0,
        "sma150_1m_ago": 88.0,
        "sma200": 85.0,
        "ema9": 99.0,
        "ema21": 97.0,
        "pivot": 97.0,
        "pivot_s1": 94.0,
        "low20": 90.0,
        "low52": 70.0,
        "high20": 102.0,
        "high52": 112.0,
        "rsi": 48.0,
        "macd": 1.0,
        "macd_signal": 0.7,
        "macd_hist": 0.3,
        "macd_hist_prev": 0.1,
        "stoch_k": 35.0,
        "stoch_d": 25.0,
        "bb_bandwidth": 10.0,
        "aroon_up": 75.0,
        "aroon_down": 25.0,
        "ret_5d": 2.0,
        "ret_20d": 5.0,
        "ret_60d": 10.0,
        "pct_from_high52": -10.0,
        "atr": 2.0,
        "atr_pct": 2.0,
        "vol_annual_pct": 30.0,
        "longterm_score": 70.0,
        "longterm_reasons": ["intakter Aufwärtstrend"],
        "daily_signal_score": 60.0,
        "daily_signal_direction": "POSITIVE",
        "daily_signal_reasons": ["positiver Tagesimpuls"],
        "weinstein_stage": 2,
        "minervini_score": 85.0,
        "value_score": 75.0,
        "quality_score": 80.0,
        "growth_score": 65.0,
        "fundamental_score": 75.0,
        "fundamental_reasons": ["Günstiges KGV (15)", "Sehr hohe Eigenkapitalrendite"],
        "feature_coverage": {
            "technical_complete": True,
            "fundamental_complete": True,
            "fundamental_current": True,
        },
        "fundamental_source_status": {"status": "current"},
        "sector": "Technology",
        "industry": "Software",
        "analyst_n": 8,
        "analyst_rating": "buy",
        "target_price": 120.0,
        "analyst_upside_pct": 20.0,
        "news_n": 2,
        "news_sentiment": "positiv",
        "next_earnings": "2026-09-01",
        "earnings_in_days": 19,
        "scenario_long": [
            {
                "label": "12 Monate",
                "reference_change_pct": 8.0,
                "reference_price": 108.0,
                "range_low_pct": -12.0,
                "range_high_pct": 30.0,
                "range_low_price": 88.0,
                "range_high_price": 130.0,
                "model_status": "unvalidated",
                "interpretation": "heuristic scenario range; not statistically calibrated",
            }
        ],
    }


def category_symbols(rankings, category, currency="USD"):
    return [
        item["symbol"]
        for item in rankings["categories"][category]["items_by_currency"].get(currency, [])
    ]


class InsightTests(unittest.TestCase):
    def test_falling_knife_cannot_enter_setup_or_entry_watchlist(self):
        row = base_row("KNIFE")
        row.update(
            {
                "price": 75.0,
                "sma20": 95.0,
                "ema9": 85.0,
                "ret_5d": -12.0,
                "ret_20d": -22.0,
                "ret_60d": -30.0,
                "macd_hist": -1.0,
                "macd_hist_prev": -0.5,
                "daily_signal_direction": "NEGATIVE",
                "weinstein_stage": 4,
                "longterm_score": 20.0,
            }
        )
        rows, rankings = enrich_rows_and_rankings([row], rankings_enabled=True)
        self.assertIsNotNone(rows[0]["falling_knife"])
        self.assertIn("KNIFE", category_symbols(rankings, "falling_knives"))
        self.assertNotIn("KNIFE", category_symbols(rankings, "daily_setups"))
        self.assertNotIn("KNIFE", category_symbols(rankings, "entry_watchlist"))

    def test_incomplete_or_fund_instrument_cannot_be_undervalued(self):
        incomplete = base_row("INCOMPLETE")
        incomplete["feature_coverage"]["fundamental_complete"] = False
        fund = base_row("ETF")
        fund["asset_type"] = "etf_fund"
        rows, rankings = enrich_rows_and_rankings(
            [incomplete, fund],
            rankings_enabled=True,
        )
        self.assertFalse(rows[0]["valuation_context"]["ranking_eligible"])
        self.assertFalse(rows[1]["valuation_context"]["ranking_eligible"])
        self.assertEqual(category_symbols(rankings, "undervalued_quality"), [])

    def test_verified_cayman_domicile_is_separate_from_provider_headquarters(self):
        cases = (
            ("BABA", "China", "Alibaba Group Holding Limited"),
            ("PDD", "Ireland", "PDD Holdings Inc."),
            ("TCOM", "Singapore", "Trip.com Group Limited"),
        )
        for symbol, headquarters, long_name in cases:
            row = base_row(symbol)
            row["provider_long_name"] = long_name
            row["provider_country"] = headquarters
            row["listing_market"] = "NASDAQ"
            enrich_row(row)
            expected_headquarters = {
                "BABA": "China",
                "PDD": "Irland",
                "TCOM": "Singapur",
            }[symbol]
            self.assertEqual(row["headquarters_country"], expected_headquarters)
            self.assertEqual(row["provider_country"], expected_headquarters)
            self.assertEqual(row["legal_domicile"], "Cayman Islands")
            self.assertEqual(row["issuer_country"], "Cayman Islands")
            self.assertTrue(row["legal_domicile_verified"])
            self.assertIn("SEC Form 20-F accession", row["legal_domicile_source"])
            self.assertIn("DEPRECATED", row["identity_semantics"]["issuer_country"])
            self.assertEqual(row["economic_exposure_country"], "China")
            self.assertEqual(row["jurisdiction_code"], "CN")
            self.assertEqual(row["jurisdiction_risk"]["level"], "high")
            self.assertGreater(row["jurisdiction_risk"]["penalty_points"], 0)
            text = " ".join(row["jurisdiction_risk"]["reasons"]).lower()
            self.assertIn("regulator", text)
            self.assertIn("adr/vie", text)

    def test_all_current_provider_country_names_normalize(self):
        cases = {
            "Japan": "JP",
            "Netherlands": "NL",
            "Switzerland": "CH",
            "Indonesia": "ID",
            "Kazakhstan": "KZ",
            "Saudi Arabia": "SA",
            "Luxembourg": "LU",
            "South Korea": "KR",
            "Bermuda": "BM",
        }
        for country, code in cases.items():
            stable, actual_code, _region, status = normalize_country(country)
            self.assertTrue(stable)
            self.assertEqual(actual_code, code)
            self.assertEqual(status, "normalized")
        stable, code, region, status = normalize_country("Example Territory")
        self.assertEqual(stable, "Example Territory")
        self.assertIsNone(code)
        self.assertEqual(region, "Nicht klassifiziert")
        self.assertEqual(status, "unclassified")
        row = base_row("UNKNOWNCOUNTRY")
        row["provider_country"] = "Example Territory"
        enrich_row(row)
        self.assertEqual(row["headquarters_country"], "Example Territory")
        self.assertEqual(row["economic_exposure_country"], "Nicht verfügbar")
        self.assertIsNone(row["economic_exposure_country_code"])
        self.assertEqual(row["jurisdiction_risk"]["level"], "unknown")
        self.assertIn("nicht klassifiziert", " ".join(row["jurisdiction_risk"]["reasons"]))

    def test_headquarters_never_implies_economic_exposure(self):
        for symbol, headquarters in (
            ("AAPL", "United States"),
            ("SHEL", "United Kingdom"),
        ):
            row = base_row(symbol)
            row["provider_country"] = headquarters
            enrich_row(row)
            self.assertTrue(row["headquarters_country"])
            self.assertEqual(row["economic_exposure_country"], "Nicht verfügbar")
            self.assertEqual(
                row["economic_exposure_classification_status"],
                "unavailable",
            )
            self.assertEqual(row["jurisdiction_risk"]["penalty_points"], 0)
        for symbol, expected in (("PBR", "Brasilien"), ("PAM", "Argentinien")):
            row = base_row(symbol)
            row["provider_country"] = "United States"
            enrich_row(row)
            self.assertEqual(row["economic_exposure_country"], expected)
            self.assertGreater(row["jurisdiction_risk"]["penalty_points"], 0)

    def test_listing_suffixes_are_complete_and_unknown_never_defaults_to_usa(self):
        cases = {
            "SAN.MC": ("Spanien", "ES"),
            "NPN.JO": ("Südafrika", "ZA"),
            "PKN.WA": ("Polen", "PL"),
            "OPAP.AT": ("Griechenland", "GR"),
            "BBRI.JK": ("Indonesien", "ID"),
            "2222.SR": ("Saudi-Arabien", "SA"),
        }
        for symbol, (country, code) in cases.items():
            row = base_row(symbol)
            row["listing_market"] = "US-ADR"
            enrich_row(row)
            self.assertEqual(row["listing_country"], country)
            self.assertEqual(row["listing_country_code"], code)
        unknown = base_row("MYSTERY.XY")
        unknown["listing_market"] = "Europa"
        enrich_row(unknown)
        self.assertIsNone(unknown["listing_country"])
        self.assertIsNone(unknown["listing_country_code"])

    def test_high_risk_microcap_cannot_win_on_raw_multiples_alone(self):
        stg = base_row("STG")
        stg.update(
            {
                "provider_country": "China",
                "provider_long_name": "Sunlands Technology Group",
                "value_score": 100.0,
                "quality_score": 95.0,
                "market_cap_usd": 50_000_000.0,
                "revenue_growth_pct": -9.6,
                "earnings_growth": 0.03,
            }
        )
        robust = base_row("ROBUST")
        robust.update(
            {
                "provider_country": "United States",
                "provider_long_name": "Robust Example Corporation",
                "value_score": 85.0,
                "quality_score": 80.0,
                "market_cap_usd": 20_000_000_000.0,
                "revenue_growth_pct": 8.0,
                "earnings_growth": 0.08,
            }
        )
        rows, rankings = enrich_rows_and_rankings(
            [stg, robust],
            rankings_enabled=True,
        )
        self.assertEqual([row["value_score"] for row in rows], [100.0, 85.0])
        self.assertGreater(
            rows[0]["valuation_thesis"]["raw_score"],
            rows[1]["valuation_thesis"]["raw_score"],
        )
        self.assertGreater(
            rows[1]["valuation_thesis"]["risk_adjusted_score"],
            rows[0]["valuation_thesis"]["risk_adjusted_score"],
        )
        self.assertEqual(
            category_symbols(rankings, "undervalued_quality")[:2],
            ["ROBUST", "STG"],
        )
        item = rankings["categories"]["undervalued_quality"]["items_by_currency"]["USD"][1]
        self.assertEqual(item["score"], item["components"]["risk_adjusted_score"])
        self.assertLessEqual(item["components"]["total_risk_penalty"], 45)

    def test_risk_adjustment_is_deterministic_bounded_and_raw_scores_unchanged(self):
        row = base_row("PDD")
        row.update(
            {
                "provider_country": "Ireland",
                "provider_long_name": "PDD Holdings Inc.",
                "market_cap_usd": 120_000_000_000.0,
                "earnings_growth": -0.15,
                "revenue_growth_pct": 11.0,
            }
        )
        raw = (row["value_score"], row["quality_score"])
        first = enrich_row(copy.deepcopy(row))
        second = enrich_row(copy.deepcopy(row))
        self.assertEqual(raw, (first["value_score"], first["quality_score"]))
        self.assertEqual(first["valuation_thesis"], second["valuation_thesis"])
        self.assertGreaterEqual(first["valuation_thesis"]["risk_adjusted_score"], 0)
        self.assertLessEqual(first["valuation_thesis"]["risk_penalty"], 45)

    def test_cyclical_peak_and_shrinking_penalties_use_disjoint_evidence(self):
        shrinking = base_row("WDS.AX")
        shrinking.update(
            {
                "sector": "Energy",
                "industry": "Oil & Gas",
                "pe": 16.19,
                "earnings_growth": -0.144,
                "revenue_growth_pct": -11.1,
            }
        )
        peak = base_row("OILPEAK")
        peak.update(
            {
                "sector": "Energy",
                "industry": "Oil & Gas",
                "pe": 8.0,
                "earnings_growth": 1.20,
                "revenue_growth_pct": 55.0,
            }
        )
        enrich_row(shrinking)
        enrich_row(peak)
        self.assertEqual(
            shrinking["valuation_thesis"]["penalty_components"]["cyclical_peak_penalty"],
            0,
        )
        self.assertEqual(
            shrinking["valuation_thesis"]["penalty_components"][
                "shrinking_fundamentals_penalty"
            ],
            8,
        )
        self.assertGreater(
            peak["valuation_thesis"]["penalty_components"]["cyclical_peak_penalty"],
            0,
        )
        self.assertEqual(
            peak["valuation_thesis"]["penalty_components"][
                "shrinking_fundamentals_penalty"
            ],
            0,
        )
        for row in (shrinking, peak):
            evidence = row["valuation_thesis"]["penalty_evidence_ids"]
            self.assertFalse(
                set(evidence["cyclical_peak_penalty"])
                & set(evidence["shrinking_fundamentals_penalty"])
            )

    def test_negative_forward_pe_is_counterargument_not_cheapness(self):
        row = base_row("STG")
        row.update(
            {
                "provider_country": "China",
                "forward_pe": -7.904762,
                "pe": 0.8279301,
            }
        )
        enrich_row(row)
        thesis = row["valuation_thesis"]
        self.assertNotIn(
            "Forward-KGV",
            " ".join(thesis["why_it_looks_cheap"]),
        )
        counterarguments = " ".join(thesis["strongest_counterarguments"])
        self.assertIn("Forward-KGV -7.90", counterarguments)
        self.assertIn("kein sinnvoller Bewertungsmultiplikator", counterarguments)

    def test_full_name_and_conservative_identity_fallbacks(self):
        provider = base_row("FULL")
        provider["provider_long_name"] = "Full Legal Company plc"
        provider["provider_country"] = "United Kingdom"
        provider["listing_market"] = "NYSE"
        enrich_row(provider)
        self.assertEqual(provider["display_name_full"], "Full Legal Company plc")
        self.assertEqual(provider["short_name"], "FULL AG")
        self.assertEqual(provider["identity_source"]["display_name"], "provider_long_name")

        fallback = base_row("FALLBACK")
        fallback["provider_long_name"] = "Broken \ufffd Name"
        enrich_row(fallback)
        self.assertEqual(fallback["display_name_full"], "FALLBACK AG")
        self.assertEqual(fallback["identity_source"]["display_name"], "short_name")

    def test_stale_retained_fundamental_scores_never_leak_into_narrative_or_export(self):
        row = base_row("STALE")
        row["feature_coverage"]["fundamental_complete"] = False
        row["feature_coverage"]["fundamental_current"] = False
        row["fundamental_source_status"] = {"status": "stale_or_unavailable"}
        enrich_row(row)
        self.assertFalse(row["valuation_context"]["available"])
        self.assertFalse(row["valuation_thesis"]["available"])
        self.assertIsNone(row["valuation_thesis"]["raw_score"])
        self.assertIn("fehlen", row["valuation_context"]["unavailable_reason"])
        narrative = " ".join(
            [
                row["research_summary"],
                row["bull_thesis"],
                " ".join(item["text"] for item in row["research_actions"]),
            ]
        ).lower()
        for forbidden in ("günstig bewertet", "qualität und value", "geschäftszahlen"):
            self.assertNotIn(forbidden, narrative)
        compact = _compact_row(row)
        for field in ("fundamental_score", "value_score", "quality_score", "growth_score"):
            self.assertNotIn(field, compact)
            self.assertIsNone(compact["valuation_context"][field])
        self.assertEqual(compact["valuation_context"]["reasons"], [])

    def test_generic_financial_company_is_excluded_from_value_rank(self):
        row = base_row("BANK")
        row["sector"] = "Financial Services"
        enrich_row(row)
        self.assertFalse(row["valuation_context"]["ranking_eligible"])
        self.assertIn("financial services", row["valuation_context"]["comparison_note"])

    def test_analyst_potential_requires_five_analysts(self):
        low_coverage = base_row("FOUR")
        low_coverage["analyst_n"] = 4
        covered = base_row("FIVE")
        covered["analyst_n"] = 5
        rows, rankings = enrich_rows_and_rankings(
            [low_coverage, covered],
            rankings_enabled=True,
        )
        self.assertFalse(rows[0]["analyst_context"]["available"])
        self.assertTrue(rows[1]["analyst_context"]["available"])
        self.assertEqual(category_symbols(rankings, "analyst_potential"), ["FIVE"])

    def test_bottoming_watch_is_explicitly_speculative(self):
        row = base_row("BOTTOM")
        row.update(
            {
                "price": 92.0,
                "ema9": 90.0,
                "low20": 85.0,
                "pct_from_high52": -30.0,
                "ret_5d": -2.0,
                "ret_20d": -5.0,
                "ret_60d": -25.0,
                "rsi": 40.0,
                "weinstein_stage": 1,
                "longterm_score": 45.0,
            }
        )
        rows, rankings = enrich_rows_and_rankings([row], rankings_enabled=True)
        self.assertTrue(rows[0]["bottoming"]["speculative"])
        item = rankings["categories"]["bottoming_watch"]["items_by_currency"]["USD"][0]
        self.assertTrue(item["speculative"])
        self.assertFalse(item["actionable"])

    def test_entry_thesis_has_metric_reasons_confirmation_and_invalidation(self):
        row = base_row("ENTRY")
        enrich_row(row)
        thesis = row["entry_thesis"]
        self.assertTrue(thesis["available"])
        self.assertIn("RSI 48.0", " ".join(thesis["why_timing_may_be_good"]))
        self.assertIn("SMA50 95.00", " ".join(thesis["what_confirms"]))
        self.assertIn("Support", " ".join(thesis["what_invalidates"]))
        self.assertEqual(thesis["falling_knife_bottoming_status"], "none")
        self.assertFalse(thesis["actionable"])

    def test_late_stage_weak_bottoming_remains_safety_capped_and_separate(self):
        row = base_row("LATEBOTTOM")
        row.update(
            {
                "price": 92.0,
                "ema9": 90.0,
                "low20": 85.0,
                "pct_from_high52": -30.0,
                "ret_5d": -2.0,
                "ret_20d": -5.0,
                "ret_60d": -25.0,
                "rsi": 40.0,
                "weinstein_stage": 3,
                "longterm_score": 0.0,
            }
        )
        rows, rankings = enrich_rows_and_rankings([row], rankings_enabled=True)
        self.assertIsNotNone(rows[0]["bottoming"])
        self.assertLessEqual(rows[0]["entry_timing_score"], 25)
        self.assertIn("LATEBOTTOM", category_symbols(rankings, "bottoming_watch"))
        self.assertNotIn("LATEBOTTOM", category_symbols(rankings, "daily_setups"))
        self.assertNotIn("LATEBOTTOM", category_symbols(rankings, "entry_watchlist"))

    def test_current_output_bottoming_has_zero_regular_setup_overlap(self):
        root = Path(__file__).resolve().parent.parent
        snapshot = json.loads(
            (root / "data" / "output" / "latest.json").read_text(encoding="utf-8")
        )
        categories = snapshot["insight_rankings"]["categories"]

        def symbols(key):
            return {
                item["symbol"]
                for items in categories[key]["items_by_currency"].values()
                for item in items
            }

        bottom = symbols("bottoming_watch")
        self.assertEqual(bottom & symbols("daily_setups"), set())
        self.assertEqual(bottom & symbols("entry_watchlist"), set())

    def test_rankings_are_deterministic_and_tie_safe(self):
        first = base_row("BBB")
        second = base_row("AAA")
        rows = [enrich_row(first), enrich_row(second)]
        ranking_a = build_insight_rankings(rows, enabled=True)
        ranking_b = build_insight_rankings(list(reversed(rows)), enabled=True)
        self.assertEqual(
            category_symbols(ranking_a, "daily_setups"),
            ["AAA", "BBB"],
        )
        self.assertEqual(
            category_symbols(ranking_a, "daily_setups"),
            category_symbols(ranking_b, "daily_setups"),
        )

    def test_category_contract_has_no_actionable_or_probability_claim(self):
        row = enrich_row(base_row())
        rankings = build_insight_rankings([row], enabled=True)
        text = json.dumps(rankings, ensure_ascii=False).lower()
        self.assertNotIn('"actionable": true', text)
        self.assertNotIn("confidence", text)
        self.assertNotIn("probability", text)
        self.assertNotIn("expected return", text)
        self.assertNotIn("scenario", rankings["categories"]["daily_setups"]["formula"])
        generated = json.dumps(
            {
                "rankings": rankings,
                "research_summary": row["research_summary"],
                "research_actions": row["research_actions"],
                "entry_timing_reason": row["entry_timing_reason"],
                "entry_thesis": row["entry_thesis"],
                "longterm_reasons": row["longterm_reasons"],
            },
            ensure_ascii=False,
        ).casefold()
        for phrase in FORBIDDEN_RESEARCH_PHRASES:
            self.assertNotIn(phrase, generated)
        self.assertIn("Analysten: Buy-Konsens", row["research_summary"])

    def test_deep_validator_rejects_missing_category_nested_actionability_and_bad_disable(self):
        row = enrich_row(base_row())
        valid = build_insight_rankings([row], enabled=True)
        validate_insight_contract(valid, [row])

        missing = copy.deepcopy(valid)
        del missing["categories"]["risk_watch"]
        with self.assertRaises(DataContractError):
            validate_insight_contract(missing, [row])

        actionable = copy.deepcopy(valid)
        actionable["categories"]["daily_setups"]["items_by_currency"]["USD"][0][
            "actionable"
        ] = True
        with self.assertRaises(DataContractError):
            validate_insight_contract(actionable, [row])

        disabled = build_insight_rankings([row], enabled=False, blockers=[])
        with self.assertRaises(DataContractError):
            validate_insight_contract(disabled, [row])

        missing_row_field = copy.deepcopy(row)
        del missing_row_field["risk_context"]
        with self.assertRaises(DataContractError):
            validate_insight_contract(valid, [missing_row_field])

        missing_thesis = copy.deepcopy(row)
        del missing_thesis["valuation_thesis"]
        with self.assertRaises(DataContractError):
            validate_insight_contract(valid, [missing_thesis])

        prescriptive = copy.deepcopy(row)
        prescriptive["research_summary"] = "Jetzt kaufen"
        with self.assertRaises(DataContractError):
            validate_insight_contract(valid, [prescriptive])

    def test_provider_free_snapshot_migration_preserves_core_rank(self):
        row = base_row()
        second = base_row("BBB")
        core = {
            "USD": {
                "company_equity": [copy.deepcopy(second), copy.deepcopy(row)]
            }
        }
        snapshot = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": 2,
            "generated_at": "2026-08-13T10:00:00+00:00",
            "data_status": {
                "status": "ok",
                "data_actionable": True,
                "blocking_reasons": [],
            },
            "model_status": {"validation": "unvalidated", "actionable": False},
            "rankings_by_currency_asset": core,
            "all": [row, second],
        }
        enriched = enrich_snapshot(snapshot)
        self.assertEqual(enriched["schema_version"], OUTPUT_SCHEMA_VERSION)
        members = enriched["rankings_by_currency_asset"]["USD"]["company_equity"]
        self.assertEqual([member["symbol"] for member in members], ["BBB", "AAA"])
        for member in members:
            self.assertIn("display_name_full", member)
            self.assertIn("jurisdiction_risk", member)
            self.assertIn("valuation_thesis", member)
            self.assertIn("entry_thesis", member)
        self.assertNotIn(
            "display_name_full",
            core["USD"]["company_equity"][0],
        )
        self.assertIn("insight_rankings", enriched)
        self.assertFalse(enriched["insight_metadata"]["scenario_ranges_used_in_core_ranking"])
        self.assertTrue(enriched["insight_metadata"]["core_ranking_rows_rehydrated"])


if __name__ == "__main__":
    unittest.main()
