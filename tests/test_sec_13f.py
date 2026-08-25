import unittest

from src.sec_13f import (
    compare_holdings,
    map_changes_to_rows,
    parse_information_table,
    summarize_symbol_changes,
)


INFO_XML = """<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip><value>1000</value>
    <shrsOrPrnAmt><sshPrnamt>100</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>ALPHABET INC</nameOfIssuer><titleOfClass>CL A</titleOfClass>
    <cusip>02079K305</cusip><value>2000</value>
    <shrsOrPrnAmt><sshPrnamt>50</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
    <putCall>PUT</putCall>
  </infoTable>
</informationTable>"""


class Sec13FTests(unittest.TestCase):
    def test_parses_namespaced_information_table(self):
        holdings = parse_information_table(INFO_XML)
        self.assertEqual(len(holdings), 2)
        apple = next(item for item in holdings if item["cusip"] == "037833100")
        self.assertEqual(apple["shares"], 100)
        alphabet = next(item for item in holdings if item["cusip"] == "02079K305")
        self.assertEqual(alphabet["put_call"], "PUT")

    def test_compares_new_increased_reduced_and_exited(self):
        current = [
            {"issuer": "A", "cusip": "1", "put_call": None, "shares": 100},
            {"issuer": "B", "cusip": "2", "put_call": None, "shares": 200},
            {"issuer": "D", "cusip": "4", "put_call": None, "shares": 40},
        ]
        previous = [
            {"issuer": "B", "cusip": "2", "put_call": None, "shares": 100},
            {"issuer": "C", "cusip": "3", "put_call": None, "shares": 300},
            {"issuer": "D", "cusip": "4", "put_call": None, "shares": 80},
        ]
        actions = {
            item["cusip"]: item["action"]
            for item in compare_holdings(current, previous)
        }
        self.assertEqual(
            actions, {"1": "new", "2": "increased", "3": "exited", "4": "reduced"}
        )

    def test_compare_handles_mixed_put_call_keys(self):
        changes = compare_holdings(
            [
                {"issuer": "A", "cusip": "1", "put_call": None, "shares": 100},
                {"issuer": "A", "cusip": "1", "put_call": "PUT", "shares": 50},
            ],
            [],
        )
        self.assertEqual(
            {(item["cusip"], item["put_call"]) for item in changes},
            {("1", None), ("1", "PUT")},
        )

    def test_maps_only_unique_conservative_issuer_matches(self):
        changes = [
            {
                "issuer": "APPLE INC",
                "cusip": "1",
                "put_call": None,
                "action": "increased",
            }
        ]
        rows = [
            {
                "symbol": "AAPL",
                "display_name_full": "Apple Inc.",
                "provider_long_name": "Apple Inc.",
                "name": "Apple",
            }
        ]
        self.assertEqual(list(map_changes_to_rows(changes, rows)), ["AAPL"])

    def test_put_new_position_is_negative_and_delay_is_explicit(self):
        signal = summarize_symbol_changes(
            [
                {
                    "manager": "Fund",
                    "action": "new",
                    "put_call": "PUT",
                    "change_pct": None,
                    "shares": 100,
                    "previous_shares": 0,
                    "filing_date": "2026-08-14",
                }
            ],
            "2026-06-30",
        )
        self.assertLess(signal["score"], 50)
        self.assertIn("45 days", signal["expected_delay"])


if __name__ == "__main__":
    unittest.main()
