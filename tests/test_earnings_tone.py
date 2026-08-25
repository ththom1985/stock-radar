import unittest
from src.earnings_tone import build_tone_signal

class EarningsToneTests(unittest.TestCase):
    def test_cfo_and_qa_shifts_affect_composite(self):
        result={"current_tone":60,"previous_tone":50,"hedging_shift":2,"qa_evasiveness_shift":None,"cfo_tone_shift":4,"reason":"better"}
        period={"period":"2026-06-30","filing_date":"2026-07-31","source":"SEC","source_url":"x","status":"prepared-only"}
        signal=build_tone_signal(result,period,period)
        self.assertEqual(signal["cfo_weight"],1.5)
        self.assertEqual(signal["transcript_status"],"prepared-only")
        self.assertEqual(signal["qa_status"],"not_available")
        self.assertEqual(signal["tone_shift"],10)

if __name__=="__main__": unittest.main()
