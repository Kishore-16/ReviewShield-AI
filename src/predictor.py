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

# Optional / Pluggable Hugging Face Online Detector Integration
try:
    from .hf_online_detector import HFOnlineDetector
    HF_DETECTOR_AVAILABLE = True
except ImportError:
    HF_DETECTOR_AVAILABLE = False

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "detector_model.pkl"

class ReviewDetectorPredictor:
    """Predictor class loading trained model checkpoint for fast inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, threshold: float = 98.00):
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[X] Model checkpoint not found at '{self.model_path}'. "
                f"Please run 'python scripts/train.py' first to train and generate the model."
            )

        artifacts = joblib.load(self.model_path)
        self.union = artifacts['union']
        self.model = artifacts['model']

        # Pluggable Hugging Face online detector instance
        self.hf_detector = HFOnlineDetector() if HF_DETECTOR_AVAILABLE else None

    def predict(self, text: str, use_online_if_available: bool = False) -> dict:
        """
        Predicts label and confidence based on configurable OR Probability threshold rule:
        - OR Probability >= threshold => OR (Human Review), Confidence = OR Probability
        - OR Probability < threshold  => CG (Computer-Generated / AI-Fake), Confidence = 100 - OR Probability
        """
        X_feat = self.union.transform([text])
        probs = self.model.predict_proba(X_feat)[0]
        
        cg_prob = float(probs[1]) * 100
        or_prob = float(probs[0]) * 100
        
        # Configurable OR Probability Threshold Rule (default 98.00%)
        if or_prob >= self.threshold:
            pred_class = 0
            label_name = "OR (Human Review)"
            confidence = or_prob
        else:
            pred_class = 1
            label_name = "CG (Computer-Generated / AI-Fake)"
            confidence = 100.0 - or_prob

        result = {
            'text': text,
            'prediction': pred_class,
            'label': label_name,
            'confidence': confidence,
            'cg_probability': cg_prob,
            'or_probability': or_prob,
            'threshold': self.threshold,
            'hf_online': {'available': False, 'status': 'disabled'}
        }

        # Pluggable Hugging Face Online API Evaluation (if enabled & token set)
        if use_online_if_available and self.hf_detector and self.hf_detector.is_available():
            hf_res = self.hf_detector.predict_online(text)
            result['hf_online'] = hf_res
            if hf_res.get('status') == 'success':
                # Override decision if Hugging Face model detects AI with high confidence
                if hf_res.get('is_ai'):
                    result['prediction'] = 1
                    result['label'] = "CG (Computer-Generated / AI-Fake) [HF Online Verified]"
                    result['confidence'] = hf_res['ai_confidence']

        return result

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
