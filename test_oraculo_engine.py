import unittest
from oraculo_engine import rating_key, score_probabilities, market_probabilities_1x2, expected_value

class TestOraculoEngine(unittest.TestCase):
    def test_rating_key_separates_leagues(self):
        self.assertEqual(rating_key(" Liga A ", " Team X"), ("liga a", "team x"))
        self.assertNotEqual(rating_key("Liga A", "Team X"), rating_key("Liga B", "Team X"))

    def test_1x2_sums_to_one(self):
        r = score_probabilities(1.45, 0.95)
        self.assertAlmostEqual(r["p_home"] + r["p_draw"] + r["p_away"], 1.0, places=12)

    def test_dynamic_support_exceeds_nine(self):
        r = score_probabilities(7.0, 1.0)
        self.assertGreater(r["max_home_goals"], 9)
        self.assertGreater(r["captured_mass"], 0.999999999)

    def test_exact_markets_are_bounded(self):
        r = score_probabilities(1.6, 1.2)
        self.assertTrue(0 <= r["p_over_2_5"] <= 1)
        self.assertTrue(0 <= r["p_btts"] <= 1)

    def test_zero_rates(self):
        r = score_probabilities(0, 0)
        self.assertAlmostEqual(r["p_draw"], 1.0, places=12)
        self.assertAlmostEqual(r["p_btts"], 0.0, places=12)

    def test_invalid_lambda(self):
        for value in (-1, float("nan"), float("inf"), 101):
            with self.assertRaises(ValueError):
                score_probabilities(value, 1)

    def test_market_overround_removed(self):
        p = market_probabilities_1x2({"home": 2, "draw": 3.5, "away": 4})
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)

    def test_invalid_odds(self):
        with self.assertRaises(ValueError):
            market_probabilities_1x2({"home": 0, "draw": 3, "away": 4})
        with self.assertRaises(ValueError):
            market_probabilities_1x2({"home": 2, "draw": 3})

    def test_ev(self):
        self.assertAlmostEqual(expected_value(0.5, 2.2), 0.1)
        self.assertLess(expected_value(0.4, 2.0), 0)

if __name__ == "__main__":
    unittest.main()
