"""
Integration Tests for Review Detector Predictor.
"""

import unittest
from pathlib import Path
from src.predictor import ReviewDetectorPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "detector_model.pkl"

class TestReviewDetectorPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = ReviewDetectorPredictor(model_path=MODEL_PATH)

    def test_predictor_human_review(self):
        text = "Awesome for traveling! Took this on vacation with our 1 year old."
        res = self.predictor.predict(text)
        self.assertEqual(res['prediction'], 0)
        self.assertGreaterEqual(res['or_probability'], 95.0)
        self.assertEqual(res['label'], "OR (Human Review)")

    def test_predictor_ai_review(self):
        text = "Furthermore, this product has demonstrated exceptional suitability for travel purposes."
        res = self.predictor.predict(text)
        self.assertEqual(res['prediction'], 1)
        self.assertLess(res['or_probability'], 95.0)
        self.assertEqual(res['label'], "CG (Computer-Generated / AI-Fake)")

if __name__ == "__main__":
    unittest.main()
