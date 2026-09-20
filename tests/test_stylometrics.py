"""
Unit Tests for Stylometric Feature Extractor.
"""

import unittest
import numpy as np
from src.stylometrics import StylometricFeatureExtractor

class TestStylometricFeatureExtractor(unittest.TestCase):

    def test_stylometric_extractor_shape(self):
        extractor = StylometricFeatureExtractor()
        sample_texts = [
            "I bought this for my daughter and we loved it.",
            "The product performs reliably and offers good value for everyday use."
        ]
        feats = extractor.transform(sample_texts)
        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(feats.shape, (2, 6))

    def test_stylometric_pronoun_density(self):
        extractor = StylometricFeatureExtractor()
        text = "I bought my daughter a toy"
        feats = extractor.transform([text])
        self.assertGreater(feats[0][0], 0.0)

    def test_stylometric_ai_opener_score(self):
        extractor = StylometricFeatureExtractor()
        text = "the product features dependable performance"
        feats = extractor.transform([text])
        self.assertGreater(feats[0][1], 0.0)

if __name__ == "__main__":
    unittest.main()
