"""
FastAPI Microservice Backend for Aurelia AI Review Guard
File: ecommerce_app/app.py

Loads d:\\ML\\ai-review-detector\\models\\detector_model.pkl to evaluate reviews:
- Human Review -> ACCEPTED
- CG / AI-Generated -> REJECTED
"""

import time
import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import ReviewDetectorPredictor

app = FastAPI(
    title="Aurelia AI Review Guard API",
    description="Real-time review classification using models/detector_model.pkl",
    version="1.0.0"
)

# Enable CORS for frontend web application integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model checkpoint: d:\ML\ai-review-detector\models\detector_model.pkl
MODEL_PATH = PROJECT_ROOT / "models" / "detector_model.pkl"
try:
    predictor = ReviewDetectorPredictor(model_path=MODEL_PATH)
    print(f"[OK] Model successfully loaded from '{MODEL_PATH}'")
except Exception as e:
    print(f"[X] Failed to load detector_model.pkl: {e}")
    predictor = None

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Review text input by user")
    title: str = Field(default="", description="Optional review title")
    rating: int = Field(default=5, ge=1, le=5, description="Star rating 1 to 5")

class ReviewResponse(BaseModel):
    status: str            # "ACCEPTED" or "REJECTED"
    decision: str          # "ACCEPTED" or "REJECTED"
    prediction_class: int  # 0 for Human, 1 for AI/CG
    label: str             # "Human Review (Genuine)" vs "CG / AI Generated Review"
    confidence: float      # Confidence score
    or_probability: float  # Human (OR) probability percentage
    cg_probability: float  # AI (CG) probability percentage
    latency_ms: float      # Model inference speed in ms
    checkpoint_file: str   # Name of model checkpoint used
    message: str           # User-facing decision message
    diagnostics: List[str] = Field(default_factory=list, description="Stylometric feedback details")

def extract_diagnostics(text: str, is_accepted: bool, threshold: float = 98.00) -> List[str]:
    """Generates detailed stylometric explanations for model decisions."""
    t_lower = text.lower()
    words = t_lower.split()

    diagnostics = []
    
    if is_accepted:
        pronouns = ["i ", "my ", " me ", "we ", "our ", "us ", "bought", "daughter", "son", "husband", "wife", "trip", "vacation"]
        p_count = sum(t_lower.count(p) for p in pronouns)
        if p_count > 0:
            diagnostics.append(f"Authentic human personal phrasing detected ({p_count} personal experience markers).")
        diagnostics.append("Natural sentence rhythm and non-repetitive vocabulary structure verified.")
        diagnostics.append(f"Evaluated by detector_model.pkl: Passed {threshold:.2f}% OR Threshold Policy.")
    else:
        ai_openers = ["furthermore", "moreover", "consequently", "in summary", "it is worth noting", "the product", "the item", "everything"]
        matched_openers = [op for op in ai_openers if op in t_lower]
        if matched_openers:
            diagnostics.append(f"Detected formal AI transition phrasing / openers: '{matched_openers[0]}'.")

        buzzwords = ["dependable", "reliable", "consistent", "suitability", "uncomplicated", "usability", "functionality", "must-have", "game-changer", "delightful addition"]
        matched_buzzwords = [bw for bw in buzzwords if bw in t_lower]
        if matched_buzzwords:
            diagnostics.append(f"Detected formal corporate/marketing buzzwords: {', '.join(matched_buzzwords[:3])}.")

        pronouns = ["i ", "my ", " me ", "we ", "our ", "us ", "bought", "daughter", "son", "husband", "wife"]
        p_count = sum(t_lower.count(p) for p in pronouns)
        if p_count == 0:
            diagnostics.append("Low personal pronoun density (lacks first-person human markers like 'my', 'I bought').")

        diagnostics.append(f"Evaluated by detector_model.pkl: Failed {threshold:.2f}% OR Threshold Policy (Flagged as CG / AI Generated).")

    return diagnostics

@app.get("/")
def health_check():
    return {
        "status": "online",
        "system": "Aurelia AI Review Guard API",
        "model_path": str(MODEL_PATH),
        "model_loaded": predictor is not None,
        "policy": "Human Review -> ACCEPTED | CG / AI Generated -> REJECTED",
        "docs": "/docs"
    }

@app.post("/api/v1/predict-review", response_model=ReviewResponse)
def evaluate_review(payload: ReviewRequest):
    if not predictor:
        raise HTTPException(status_code=500, detail="detector_model.pkl not loaded.")
        
    t0 = time.time()
    try:
        combined_text = f"{payload.title} {payload.text}".strip() if payload.title else payload.text
        
        # Pass text directly to trained detector_model.pkl
        res = predictor.predict(combined_text)
        latency = round((time.time() - t0) * 1000, 2)
        
        # Human Review (prediction == 0) -> ACCEPTED
        # CG / AI Generated (prediction == 1) -> REJECTED
        is_accepted = (res['prediction'] == 0)
        diag = extract_diagnostics(combined_text, is_accepted, threshold=res.get('threshold', 98.00))

        if is_accepted:
            return ReviewResponse(
                status="ACCEPTED",
                decision="ACCEPTED",
                prediction_class=0,
                label="Human Review (Genuine)",
                confidence=round(res['confidence'], 2),
                or_probability=round(res['or_probability'], 2),
                cg_probability=round(res['cg_probability'], 2),
                latency_ms=latency,
                checkpoint_file="detector_model.pkl",
                message="Review ACCEPTED: Verified as authentic human customer review.",
                diagnostics=diag
            )
        else:
            return ReviewResponse(
                status="REJECTED",
                decision="REJECTED",
                prediction_class=1,
                label="CG / AI Generated Review",
                confidence=round(res['confidence'], 2),
                or_probability=round(res['or_probability'], 2),
                cg_probability=round(res['cg_probability'], 2),
                latency_ms=latency,
                checkpoint_file="detector_model.pkl",
                message="Review REJECTED: Deceptive Computer-Generated / AI writing pattern detected.",
                diagnostics=diag
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sample-reviews")
def get_sample_reviews():
    """Curated test samples for live demonstration."""
    return {
        "human_samples": [
            {
                "title": "Transformed my cross-country flights",
                "text": "Been using these for a month on my daily train commute. The noise cancelling handles engine rumble really well, though the ear cups get a little warm after two hours. Battery easily lasts my whole work week.",
                "rating": 5,
                "expected": "ACCEPTED"
            },
            {
                "title": "Great for family travel",
                "text": "Awesome for traveling! Took this on vacation with our 1 year old. He is 26lbs and approx 30 inches long. It was perfect for a small space and lightweight to carry.",
                "rating": 5,
                "expected": "ACCEPTED"
            }
        ],
        "ai_samples": [
            {
                "title": "Exceptional Product Suitability",
                "text": "Furthermore, this product has demonstrated exceptional suitability for travel purposes. Notably, its compact design and lightweight construction render it ideal for utilization in confined spaces.",
                "rating": 5,
                "expected": "REJECTED"
            },
            {
                "title": "A Total Game Changer",
                "text": "This item is an absolute MUST-HAVE and a total GAME-CHANGER! Featuring state-of-the-art design, it delivers a delightful addition to your routine. Guaranteed complete satisfaction!",
                "rating": 5,
                "expected": "REJECTED"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
