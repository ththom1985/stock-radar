import unittest
from src.earnings_tone import build_tone_signal

class EarningsToneTests(unittest.TestCase):
    def test_cfo_and_qa_shifts_affect_composite(self):
        result={"current_tone":60,"previous_tone":50,"hedging_shift":2,"qa_evasiveness_shift":3,"cfo_tone_shift":4,"reason":"better"}
        signal=build_tone_signal(result,{"year":2026,"quarter":2},{"year":2026,"quarter":1})
        self.assertEqual(signal["cfo_weight"],1.5)
        self.assertEqual(signal["score"],61.0)
        self.assertEqual(signal["tone_shift"],10)

if __name__=="__main__": unittest.main()
