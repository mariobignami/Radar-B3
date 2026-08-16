import unittest

from src.predict import StockPredictor


class TestIndicatorNormalization(unittest.TestCase):
    def test_accepts_percent_inputs(self):
        normalized = StockPredictor._normalize_technical_indicators(
            {
                "open_gap_percent": 2.5,
                "volume_rel_20d_percent": 130,
            }
        )

        self.assertAlmostEqual(normalized["open_gap_percent"], 2.5)
        self.assertAlmostEqual(normalized["open_gap_pct"], 0.025)
        self.assertAlmostEqual(normalized["volume_rel_20d_percent"], 130.0)
        self.assertAlmostEqual(normalized["volume_rel_20d"], 0.3)

    def test_accepts_ratio_inputs(self):
        normalized = StockPredictor._normalize_technical_indicators(
            {
                "open_gap_pct": 0.015,
                "volume_rel_20d": 0.2,
            }
        )

        self.assertAlmostEqual(normalized["open_gap_pct"], 0.015)
        self.assertAlmostEqual(normalized["open_gap_percent"], 1.5)
        self.assertAlmostEqual(normalized["volume_rel_20d"], 0.2)
        self.assertAlmostEqual(normalized["volume_rel_20d_percent"], 120.0)


if __name__ == "__main__":
    unittest.main()
