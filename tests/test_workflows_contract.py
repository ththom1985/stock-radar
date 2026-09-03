from __future__ import annotations

import re
import unittest
from pathlib import Path

from tests.helpers import ROOT


class WorkflowContractTests(unittest.TestCase):
    def test_primary_schedule_has_conditional_recovery_run(self):
        daily = (ROOT / ".github/workflows/daily.yml").read_text(encoding="utf-8")
        manual = (ROOT / ".github/workflows/intraday.yml").read_text(encoding="utf-8")
        self.assertEqual(
            re.findall(r'cron:\s*"([^"]+)"', daily),
            ["15 23 * * 1-5", "15 6 * * 2-6"],
        )
        self.assertIn("python -m src.automation_guard", daily)
        self.assertIn("needs.freshness.outputs.rebuild == 'true'", daily)
        self.assertNotIn("cron:", manual)
        self.assertNotIn("STOCK_RADAR_INTRADAY", daily + manual)

    def test_publication_is_complete_and_failures_are_not_swallowed(self):
        for name in ("daily.yml", "intraday.yml"):
            text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("data/deep_fundamentals.json", text)
            self.assertIn("data/news_cache.json", text)
            self.assertIn("data/failed_symbols.json", text)
            self.assertIn("data/probability_models.json", text)
            self.assertIn("data/probability_validation.json", text)
            self.assertIn("data/probability_forward_status.json", text)
            self.assertIn("data/finra_short_interest.json", text)
            self.assertIn("data/valuation_anomalies.json", text)
            self.assertIn("data/financial_sector_history.json", text)
            self.assertIn("data/opportunity_history.json", text)
            self.assertIn("gzip -dk data/output/latest.json.gz", text)
            self.assertIn(
                "gzip -n -c -9 data/output/latest.json > data/output/latest.json.gz",
                text,
            )
            self.assertIn("data/output/latest.json.gz", text)
            self.assertIn("git add -u -- data docs", text)
            self.assertIn("docs/details", text)
            self.assertIn("git status --porcelain", text)
            self.assertIn("source_sha=\"$(git rev-parse HEAD)\"", text)
            self.assertIn("git fetch origin main", text)
            self.assertIn("Source changed during analysis", text)
            self.assertNotIn("git pull --rebase", text)
            self.assertIn("All three pull/rebase/push attempts failed", text)
            self.assertIn("Report publication failure", text)
            self.assertIn("python -m src.verify_live", text)
            self.assertIn("FINRA_CLIENT_ID", text)
            self.assertIn("FINRA_CLIENT_SECRET", text)
            self.assertIn('STOCK_RADAR_START_NEW_PAPER: "1"', text)
            self.assertIn(
                "python -m src.probability_forward publish-status --aggregate-only",
                text,
            )
            self.assertNotIn("src.probability_forward capture", text)
            self.assertNotIn("src.probability_forward evaluate", text)
            self.assertIn("persist-credentials: false", text)
            self.assertNotIn("|| true", text)
            self.assertNotIn("push ||", text)
            shas = re.findall(r"uses:\s*actions/[^@]+@([0-9a-f]{40})", text)
            expected_action_count = 3 if name == "daily.yml" else 2
            self.assertEqual(len(shas), expected_action_count)

    def test_daily_tests_are_blocking_and_run_before_analysis(self):
        daily = (ROOT / ".github/workflows/daily.yml").read_text(encoding="utf-8")
        command = "python -m unittest discover -s tests -v"
        self.assertIn(command, daily)
        self.assertLess(daily.index(command), daily.index("python -m src.analyze"))

    def test_live_data_health_runs_after_analysis_and_blocks_publication(self):
        for name in ("daily.yml", "intraday.yml"):
            text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            analysis = text.index("python -m src.analyze")
            health = text.index("python -m src.published_health")
            publication = text.index("Publish complete data transaction")
            self.assertLess(analysis, health)
            self.assertLess(health, publication)
            health_step = text[text.rfind("- name:", 0, health) : health]
            self.assertNotIn("continue-on-error: true", health_step)

    def test_publication_jobs_are_main_branch_only(self):
        for name in ("daily.yml", "intraday.yml"):
            text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("if: github.ref == 'refs/heads/main'", text)
            self.assertIn("ref: main", text)
            self.assertIn("git push origin HEAD:main", text)


if __name__ == "__main__":
    unittest.main()
