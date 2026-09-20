"""
Review Detector Predictor Engine.
Provides high-speed inference for classifying reviews as OR (Human) or CG (AI-Generated).
Enforces the 95.00% OR Probability Threshold Rule.
"""

import sys
import joblib
from pathlib import Path
import numpy as np

# Ensure pickle compatibility for StylometricFeatureExtractor
from .stylometrics import StylometricFeatureExtractor
sys.modules['predict_review'] = sys.modules[__name__]
sys.modules['predict_review'].StylometricFeatureExtractor = StylometricFeatureExtractor

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "detector_model.pkl"

class ReviewDetectorPredictor:
    """Predictor class loading trained model checkpoint for fast inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[X] Model checkpoint not found at '{self.model_path}'. "
                f"Please run 'python scripts/train.py' first to train and generate the model."
            )

        artifacts = joblib.load(self.model_path)
        self.union = artifacts['union']
        self.model = artifacts['model']

    def predict(self, text: str) -> dict:
        """
        Predicts label and confidence based on 95.00% OR Probability threshold rule:
        - OR Probability >= 95.00% => OR (Human Review), Confidence = OR Probability
        - OR Probability < 95.00%  => CG (Computer-Generated / AI-Fake), Confidence = 100 - OR Probability
        """
        X_feat = self.union.transform([text])
        probs = self.model.predict_proba(X_feat)[0]
        
        cg_prob = float(probs[1]) * 100
        or_prob = float(probs[0]) * 100
        
        # 95.00% OR Probability Threshold Rule
        if or_prob >= 95.00:
            pred_class = 0
            label_name = "OR (Human Review)"
            confidence = or_prob
        else:
            pred_class = 1
            label_name = "CG (Computer-Generated / AI-Fake)"
            confidence = 100.0 - or_prob
        
        return {
            'text': text,
            'prediction': pred_class,
            'label': label_name,
            'confidence': confidence,
            'cg_probability': cg_prob,
            'or_probability': or_prob
        }

    def display_prediction(self, text: str):
        """Prints a nicely formatted visual prediction result card."""
        res = self.predict(text)
        
        print("=" * 70)
        print("                 AI REVIEW DETECTION RESULT (95% Rule)")
        print("=" * 70)
        print(f"Review Text: \"{text}\"\n")
        print(f"--> PREDICTED CLASS : {res['label']}")
        print(f"--> CONFIDENCE      : {res['confidence']:.2f}%")
        print(f"--> OR (Human) Probability: {res['or_probability']:.2f}% (Threshold: >= 95.00% for OR)")
        print(f"--> CG (AI) Probability   : {res['cg_probability']:.2f}%")
        
        if res['prediction'] == 0:
            print("--> STATUS          : [GENUINE HUMAN REVIEW]")
        else:
            print("--> STATUS          : [SUSPICIOUS / AI GENERATED]")
        print("=" * 70 + "\n")
