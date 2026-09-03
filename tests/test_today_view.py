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
    def test_ideal_candidate_is_ranked_ahead_of_stronger_technical_signals(self):
        rows=[row(f"TECH{index}") for index in range(6)]
        rows[5]["sweet_spot"]["reliability_score"]=1
        question_views={"cheap_with_potential":[{
            "symbol":"TECH5",
            "situation":{"code":"ideal","label":"Idealfall: günstig UND am Einstiegspunkt"},
            "deal_quality":{"label":"Deal-Qualität 4/5"},
        }]}
        view=build_today_view(rows,question_views=question_views)
        self.assertEqual(view["candidates"][0]["symbol"],"TECH5")
        self.assertTrue(view["candidates"][0]["ideal_entry"])
        self.assertEqual(view["ideal_candidate_count"],1)
        self.assertEqual(view["technical_candidate_count"],6)
        self.assertIn("1 Idealfall",view["headline"])
if __name__=="__main__":unittest.main()
