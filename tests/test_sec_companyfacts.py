import unittest

from src.sec_companyfacts import merge_official_fundamentals, parse_companyfacts


def fact(tag, values):
    return {
        tag: {
            "units": {
                "USD": [
                    {
                        "val": value,
                        "start": f"{year}-01-01",
                        "end": f"{year}-12-31",
                        "filed": f"{year + 1}-02-01",
                        "form": "10-K",
                        "fp": "FY",
                        "fy": year,
                        "accn": f"{year}-test",
                    }
                    for year, value in values
                ]
            }
        }
    }


class SecCompanyfactsTests(unittest.TestCase):
    def test_reduces_annual_facts_and_derives_fcf(self):
        facts = {}
        facts.update(
            fact(
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                [(2024, 1000), (2025, 1200)],
            )
        )
        facts.update(fact("NetIncomeLoss", [(2024, 100), (2025, 144)]))
        facts.update(
            fact(
                "NetCashProvidedByUsedInOperatingActivities",
                [(2024, 180), (2025, 210)],
            )
        )
        facts.update(
            fact(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                [(2024, 50), (2025, 60)],
            )
        )
        payload = {"cik": 123, "entityName": "Example", "facts": {"us-gaap": facts}}
        parsed = parse_companyfacts(payload)
        self.assertEqual(parsed["latest"]["free_cash_flow"], 150)
        self.assertAlmostEqual(parsed["derived"]["profit_margin"], 0.12)
        self.assertAlmostEqual(parsed["derived"]["revenue_growth"], 0.2)

    def test_official_fields_override_accounting_but_keep_market_fields(self):
        yahoo = {"market_cap": 3000, "pe": 20, "profit_margin": 0.05}
        sec = {
            "latest": {"revenue": 1200, "free_cash_flow": 150},
            "derived": {"profit_margin": 0.12, "revenue_growth": 0.2},
        }
        merged = merge_official_fundamentals(yahoo, sec)
        self.assertEqual(merged["pe"], 20)
        self.assertEqual(merged["profit_margin"], 0.12)
        self.assertEqual(merged["ps"], 2.5)
        self.assertEqual(merged["price_to_fcf"], 20)


if __name__ == "__main__":
    unittest.main()
