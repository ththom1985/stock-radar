import unittest
from src.filing_diffs import compare_risk_sections, extract_risk_factors

class FilingDiffTests(unittest.TestCase):
    def test_extracts_risk_section(self):
        risk = ("Cybersecurity incidents could have a material adverse effect on our operations. " * 20)
        html = f"<h2>Item 1A. Risk Factors</h2><p>{risk}</p><h2>Item 1B. Unresolved</h2>"
        self.assertIn("Cybersecurity", extract_risk_factors(html))

    def test_new_intensified_risk_reduces_score(self):
        old = ("Demand may vary substantially and affect results. " * 20)
        new = old + ("New cybersecurity litigation may have a material adverse impact on liquidity. " * 10)
        result = compare_risk_sections(new, old)
        self.assertGreater(result["new_paragraph_count"], 0)
        self.assertGreater(result["intensified_count"], 0)
        self.assertLess(result["score"], 50)

if __name__ == "__main__":
    unittest.main()
