import io
import unittest
import zipfile
from datetime import date

from src.congress_trades import (
    parse_house_index,
    parse_house_ptr_text,
    parse_senate_report_html,
    senate_source_status,
    summarize_congress_trades,
)


class CongressTradesTests(unittest.TestCase):
    def test_house_index_keeps_only_periodic_transaction_reports(self):
        xml = b"""<?xml version="1.0"?>
        <FinancialDisclosure>
          <Member><First>A</First><Last>B</Last><FilingType>P</FilingType>
          <Year>2026</Year><FilingDate>8/18/2026</FilingDate><DocID>1</DocID></Member>
          <Member><First>C</First><Last>D</Last><FilingType>A</FilingType>
          <Year>2026</Year><FilingDate>8/17/2026</FilingDate><DocID>2</DocID></Member>
        </FinancialDisclosure>"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("2026FD.xml", xml)
        rows = parse_house_index(buffer.getvalue())
        self.assertEqual([row["doc_id"] for row in rows], ["1"])

    def test_congress_signal_uses_ranges_and_transaction_date(self):
        trades = [
            {
                "ticker": "AAPL",
                "transaction_type": "P",
                "transaction_date": "2026-08-01",
                "amount_low": 1001,
            },
            {
                "ticker": "AAPL",
                "transaction_type": "P",
                "transaction_date": "2026-08-10",
                "amount_low": 15001,
            },
        ]
        signal = summarize_congress_trades(
            trades, today=date(2026, 8, 25)
        )["AAPL"]
        self.assertGreater(signal["score"], 50)
        self.assertEqual(signal["latest_transaction_date"], "2026-08-10")
        self.assertIn("45 days", signal["expected_delay"])

    def test_parses_house_ptr_transaction_row(self):
        text = (
            "Apple Inc. - Common Stock (AAPL) [ST] "
            "S (partial) 03/16/2026 03/16/2026 $1,001 - $15,000"
        )
        trades = parse_house_ptr_text(
            text,
            {
                "first": "Mark",
                "last": "Example",
                "filing_date": "2026-03-31",
                "year": 2026,
                "doc_id": "123",
            },
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["ticker"], "AAPL")
        self.assertEqual(trades[0]["transaction_type"], "S")
        self.assertEqual(trades[0]["amount_low"], 1001)

    def test_senate_requires_explicit_acknowledgement(self):
        status = senate_source_status()
        self.assertTrue(status["acknowledged"])
        self.assertTrue(status["local_only"])
        self.assertFalse(status["public_detail"])

    def test_github_actions_never_fetches_local_only_senate_details(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            status = senate_source_status()
        self.assertEqual(status["status"], "local_only_not_run")

    def test_parses_senate_transaction_table(self):
        report_html = """
        <table><tbody><tr><td>1</td><td>08/01/2026</td><td>Self</td>
        <td><a>AAPL</a></td><td>Apple Inc</td><td>Stock</td>
        <td>Purchase</td><td>$1,001 - $15,000</td><td>--</td></tr></tbody></table>
        """
        trades = parse_senate_report_html(
            report_html,
            {
                "member": "Test Senator",
                "report_path": "/search/view/ptr/abc/",
                "filing_date": "2026-08-20",
            },
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["ticker"], "AAPL")
        self.assertEqual(trades[0]["transaction_type"], "P")


if __name__ == "__main__":
    unittest.main()
    parse_senate_report_html,
