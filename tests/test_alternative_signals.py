import unittest
from datetime import datetime, timezone

from src.alternative_signals import build_alternative_signals
from src.sec_insiders import parse_form4, summarize_transactions


FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Example Insider</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector><isOfficer>0</isOfficer>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-08-20</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


class AlternativeSignalTests(unittest.TestCase):
    def test_parses_open_market_form4_purchase(self):
        transactions = parse_form4(FORM4)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["code"], "P")
        self.assertEqual(transactions[0]["value"], 2500)

    def test_cluster_requires_multiple_distinct_buyers(self):
        one = parse_form4(FORM4)[0]
        two = {**one, "owner": "Second Insider"}
        summary = summarize_transactions(
            [one, two],
            today=datetime(2026, 8, 25, tzinfo=timezone.utc).date(),
        )
        self.assertTrue(summary["cluster_purchase"])
        self.assertGreater(summary["score"], 50)

    def test_confluence_reports_missing_signals_and_recency(self):
        insider = {
            "score": 80,
            "last_success_at": "2026-08-24T00:00:00+00:00",
            "source": "SEC EDGAR Form 4",
        }
        result = build_alternative_signals(
            {},
            insider=insider,
            now=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )
        self.assertEqual(result["coverage_count"], 1)
        self.assertIn("congress", result["missing_signals"])
        self.assertGreater(result["confluence_score"], 50)


if __name__ == "__main__":
    unittest.main()
