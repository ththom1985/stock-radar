import copy
import json
import shutil
import subprocess
import unittest
from datetime import date

from src.automation_guard import recovery_needed
from src.freshness import (
    build_session_freshness, evaluate_session_freshness, parse_timestamp,
    timestamp_failures,
)
from src.market_calendar import _is_projected_us_session
from tests.helpers import ROOT


def payloads(rows, generated="2026-09-05T01:17:37Z"):
    status = {"session_freshness": build_session_freshness(rows), "coverage_sla_pct": 97}
    snapshot = {"generated_at": generated, "all": rows, "data_status": status}
    exported = {"generated_at": generated, "instruments": rows, "data_status": status}
    return snapshot, exported, copy.deepcopy(exported)


CASES = [
    ("Saturday", "AAPL", "2026-09-04", "2026-09-05T12:00:00Z", True),
    ("Sunday", "AAPL", "2026-09-04", "2026-09-06T12:00:00Z", True),
    ("Labor Day", "AAPL", "2026-09-04", "2026-09-07T23:59:00Z", True),
    ("Tuesday before buffer", "AAPL", "2026-09-04", "2026-09-08T21:29:59Z", True),
    ("Tuesday buffer due", "AAPL", "2026-09-04", "2026-09-08T21:30:00Z", False),
    ("missed Friday", "AAPL", "2026-09-03", "2026-09-07T06:15:00Z", False),
    ("stale weekday", "AAPL", "2026-09-01", "2026-09-03T06:15:00Z", False),
    ("future date", "AAPL", "2026-09-08", "2026-09-07T12:00:00Z", False),
    ("same day partial", "AAPL", "2026-09-08", "2026-09-08T20:00:00Z", False),
    ("holiday bar invalid", "AAPL", "2026-09-07", "2026-09-08T08:00:00Z", False),
    ("weekend bar invalid", "AAPL", "2026-09-05", "2026-09-07T08:00:00Z", False),
    ("Good Friday", "AAPL", "2026-04-02", "2026-04-03T23:00:00Z", True),
    ("Easter Monday US open", "AAPL", "2026-04-02", "2026-04-06T21:30:00Z", False),
    ("Christmas weekend", "AAPL", "2026-12-24", "2026-12-27T23:00:00Z", True),
    ("winter DST before buffer", "AAPL", "2026-01-05", "2026-01-06T22:29:59Z", True),
    ("winter DST due", "AAPL", "2026-01-05", "2026-01-06T22:30:00Z", False),
    ("Germany on US holiday before close", "SAP.DE", "2026-09-04", "2026-09-07T11:00:00Z", True),
    ("Germany on US holiday due", "SAP.DE", "2026-09-04", "2026-09-07T17:00:00Z", False),
    ("Tokyo on US holiday due", "7203.T", "2026-09-04", "2026-09-07T08:00:00Z", False),
    ("crypto weekend due", "BTC-USD", "2026-09-04", "2026-09-06T01:30:00Z", False),
    ("crypto weekend current", "BTC-USD", "2026-09-05", "2026-09-06T06:15:00Z", True),
    ("crypto current day incomplete", "BTC-USD", "2026-09-06", "2026-09-06T22:00:00Z", False),
    ("Saudi Sunday due", "2222.SR", "2026-09-03", "2026-09-06T13:50:00Z", False),
    ("unknown market", "ABC.UNKNOWN", "2026-09-04", "2026-09-07T11:00:00Z", False),
    ("non-US index not assumed NYSE", "^N225", "2026-09-04", "2026-09-07T11:00:00Z", False),
    ("missing date", "AAPL", "", "2026-09-07T11:00:00Z", False),
]


class SessionFreshnessTests(unittest.TestCase):
    def test_calendar_cases(self):
        for name, symbol, bar, now, expected in CASES:
            with self.subTest(name=name):
                contract = build_session_freshness([{"symbol": symbol, "bar_date": bar}])
                result = evaluate_session_freshness(contract, now=parse_timestamp(now))
                self.assertEqual(not result["blocking_reasons"], expected, result)

    def test_shared_nyse_calendar_observed_holidays(self):
        for day in ("2026-07-03", "2026-06-19", "2026-09-07", "2025-01-09"):
            self.assertFalse(_is_projected_us_session(date.fromisoformat(day)))
        self.assertTrue(_is_projected_us_session(date(2021, 12, 31)))

    def test_fresh_generated_timestamp_does_not_hide_missing_friday(self):
        rows = [{"symbol": "AAPL", "bar_date": "2026-09-03"}]
        for generated in ("2026-09-05T01:17:37Z", "2026-09-07T06:14:00Z"):
            rebuild, reason = recovery_needed(
                *payloads(rows, generated), now=parse_timestamp("2026-09-07T06:15:00Z")
            )
            self.assertTrue(rebuild)
            self.assertIn("missing completed sessions", reason)

    def test_recovery_skips_complete_friday_on_monday_holiday(self):
        rebuild, reason = recovery_needed(
            *payloads([{"symbol": "AAPL", "bar_date": "2026-09-04"}]),
            now=parse_timestamp("2026-09-07T06:15:00Z"),
        )
        self.assertFalse(rebuild, reason)

    def test_recovery_detects_bar_mismatch_under_same_timestamp(self):
        snapshot, exported, live = payloads([{"symbol": "AAPL", "bar_date": "2026-09-04"}])
        snapshot = copy.deepcopy(snapshot)
        snapshot["all"][0]["bar_date"] = "2026-09-03"
        rebuild, reason = recovery_needed(
            snapshot, exported, live, now=parse_timestamp("2026-09-07T06:15:00Z")
        )
        self.assertTrue(rebuild)
        self.assertIn("evidence differs", reason)

    def test_small_crypto_scope_cannot_hide_behind_us_coverage(self):
        rows = [{"symbol": f"US{i}", "bar_date": "2026-09-04"} for i in range(100)]
        rows.append({"symbol": "BTC-USD", "bar_date": "2026-09-04"})
        result = evaluate_session_freshness(
            build_session_freshness(rows), now=parse_timestamp("2026-09-07T11:00:00Z")
        )
        self.assertGreater(result["fresh_bar_coverage_pct"], 97)
        self.assertTrue(result["blocking_reasons"])
        self.assertEqual(result["markets"]["UTC-24x7"]["fresh_pct"], 0)

    def test_recovery_retries_missing_minority_even_when_dashboard_sla_passes(self):
        rows = [{"symbol": f"US{i}", "bar_date": "2026-09-04"} for i in range(99)]
        rows.append({"symbol": "AAPL", "bar_date": "2026-09-03"})
        now = parse_timestamp("2026-09-07T06:15:00Z")
        current = evaluate_session_freshness(build_session_freshness(rows), now=now)
        self.assertEqual(current["blocking_reasons"], [])
        rebuild, reason = recovery_needed(*payloads(rows), now=now)
        self.assertTrue(rebuild)
        self.assertIn("missing completed sessions for 1 instruments", reason)

    def test_future_generation_is_rejected(self):
        now = parse_timestamp("2026-09-07T06:15:00Z")
        for generated in ("2026-09-07T06:15:01Z", "2026-09-07T06:14:00", "bad"):
            self.assertTrue(timestamp_failures({"generated_at": generated}, now))
        self.assertFalse(timestamp_failures({"generated_at": "2026-09-05T01:17:37Z"}, now))

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute the browser contract")
    def test_static_gate_keeps_safety_checks_and_rejects_bad_freshness_evidence(self):
        row = {"symbol": "AAPL", "bar_date": "2026-09-04"}
        for key in (
            "display_name_full", "short_name", "headquarters_country", "legal_domicile",
            "listing_country", "listing_market", "economic_exposure_country",
            "sector_display", "industry_display", "sweet_spot",
        ):
            row[key] = ""
        row["detail_chunk"] = "details/000.json"
        row["probability_forecast"] = {"status": "withheld", "actionable": False, "forecasts": []}
        _, exported, _ = payloads([row])
        exported["data_status"].update(status="ok", data_actionable=True, blocking_reasons=[])
        exported["model_status"] = {"validation": "unvalidated", "actionable": False}
        exported["insight_rankings"] = {
            "model_status": "heuristic_unvalidated", "actionable": False, "enabled": True,
        }
        exported["instrument_contract"] = {"model_status": "heuristic_unvalidated", "actionable": False}
        code = """
const fs = require('fs');
const {sessionFreshness} = require('./docs/freshness.js');
const html = fs.readFileSync('./docs/index.html', 'utf8');
function source(name, next) {
  return html.slice(html.indexOf('    function ' + name + '('),
    html.indexOf('    function ' + next + '('));
}
const gate = new Function('sessionFreshness',
  source('hasActionableTrue', 'hydrateSweetSpotReasons') +
  source('staticGateFailures', 'categoryItems') + 'return staticGateFailures;')(sessionFreshness);
const base = JSON.parse(fs.readFileSync(0, 'utf8'));
const now = Date.parse('2026-09-07T11:00:00Z');
const changes = [
  d => {},
  d => d.model_status.actionable = true,
  d => d.data_status.data_actionable = false,
  d => d.insight_rankings.actionable = true,
  d => d.generated_at = '2026-09-07T11:00:01Z',
  d => delete d.data_status.session_freshness,
  d => d.instruments[0].bar_date = '2026-09-03',
  d => d.data_status.session_freshness.groups[0].symbols = [],
  d => d.data_status.session_freshness.groups[0].next_completed_after = 'invalid'
];
console.log(JSON.stringify(changes.map(change => {
  const d = structuredClone(base); change(d); return gate(d, now).length > 0;
})));
"""
        result = subprocess.run(
            ["node", "-e", code], input=json.dumps(exported), text=True,
            capture_output=True, cwd=ROOT, check=True,
        )
        self.assertEqual(json.loads(result.stdout), [False] + [True] * 8)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute the browser contract")
    def test_browser_matches_python_for_all_calendar_cases(self):
        fixtures = []
        for name, symbol, bar, now, expected in CASES:
            _, exported, _ = payloads([{"symbol": symbol, "bar_date": bar}], "2025-01-01T00:00:00Z")
            fixtures.append({"name": name, "data": exported, "now": now, "expected": expected})
        code = """
const fs = require('fs');
const {sessionFreshness} = require('./docs/freshness.js');
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(c => ({
  name: c.name, allowed: !sessionFreshness(c.data, Date.parse(c.now)).failures.length
}))));
"""
        result = subprocess.run(
            ["node", "-e", code], input=json.dumps(fixtures), text=True,
            capture_output=True, cwd=ROOT, check=True,
        )
        for case, actual in zip(fixtures, json.loads(result.stdout)):
            self.assertEqual(actual["allowed"], case["expected"], actual["name"])
