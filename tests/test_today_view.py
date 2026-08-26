import unittest
from src.today_view import build_today_view,local_zone,traffic_light

def row(symbol="AAA",verdict="fair",status="in_zone_confirmed"):
    return {"symbol":symbol,"name":symbol,"display_name_full":symbol,"asset_type":"company_equity","currency":"EUR","price_local":100,"fx_usd":1.2,"entry_timing_score":80,"sweet_spot":{"combined_status":status,"lower":118,"ideal":120,"upper":122,"reliability_score":80},"expert_analysis":{"valuation":{"verdict":verdict,"fair_value_range":{"lower":90,"upper":110}},"risks":{"top_risks":["Test-Risiko"]}},"alternative_signals":{"contributing_groups":[]}}

class TodayViewTests(unittest.TestCase):
    def test_green_is_timing_only_even_when_valuation_is_withheld(self):
        self.assertEqual(traffic_light(row()),"green")
        self.assertEqual(traffic_light(row(verdict="overpriced")),"green")
        self.assertEqual(traffic_light(row(verdict="data_review_required")),"green")
    def test_zone_is_converted_to_listing_currency(self):
        self.assertAlmostEqual(local_zone(row())["lower"],98.333333,places=5)
    def test_no_forced_pick_when_no_green(self):
        view=build_today_view([row(status="approaching")])
        self.assertEqual(view["candidate_count"],0)
        self.assertIn("kein überzeugender",view["headline"])
        self.assertEqual(len(view["candidates"]),1)
if __name__=="__main__":unittest.main()
