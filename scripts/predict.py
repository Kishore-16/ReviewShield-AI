"""
Interactive & Command-Line Review Prediction Tool.

Usage:
    # Single text evaluation:
    python scripts/predict.py "Awesome for traveling! Took this on vacation with our 1 year old."

    # Interactive prompt mode:
    python scripts/predict.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import ReviewDetectorPredictor

def run_benchmark_samples(predictor: ReviewDetectorPredictor):
    """Runs prediction on a suite of sample human and synthetic AI reviews."""
    sample_reviews = [
        ("Awesome for traveling!!!. Took this on vacation with our 1 year old. He is 26lbs and approx 30 inches long. It was perfect for a small space and lightweight to carry.", "OR (Human Review)"),
        ("Easy setup. This product is so sturdy I tried it out myself and never gave way at 100+ lbs. I love the easy craftsmanship that there is absolutely no way to mess it up.", "OR (Human Review)"),
        ("Furthermore, this product has demonstrated exceptional suitability for travel purposes. Notably, its compact design and lightweight construction render it ideal for utilization in confined spaces.", "CG (AI-Fake)"),
        ("This item is an absolute MUST-HAVE and a total GAME-CHANGER! Featuring state-of-the-art design, it delivers a delightful addition to your routine. Guaranteed complete satisfaction!", "CG (AI-Fake)"),
        ("This product is ABSOLUTELY AMAZING!!! PURE PERFECTION!! I am completely OBSESSED with how well it works!!! Best purchase EVER!!! Highly satisfied!!", "CG (AI-Fake)"),
        ("If you are looking for a reliable option, highly recommend giving this product a try. Look no further if you need a high-quality item for everyday use.", "CG (AI-Fake)")
    ]

    print("\n" + "#" * 70)
    print("         RUNNING PRE-PACKAGED BENCHMARK SAMPLE TEST")
    print("#" * 70 + "\n")

    for i, (text, expected) in enumerate(sample_reviews, 1):
        print(f"[TEST #{i}] Expected Ground Truth: {expected}")
        predictor.display_prediction(text)

def interactive_loop(predictor: ReviewDetectorPredictor):
    """Interactive CLI loop."""
    print("\n" + "=" * 70)
    print(" INTERACTIVE MODE: TYPE OR PASTE ANY REVIEW BELOW (type 'exit' to quit)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter Review Text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n[Exiting prediction engine. Goodbye!]")
                break

            predictor.display_prediction(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting prediction engine.]")
            break

def main():
    model_path = PROJECT_ROOT / "models" / "detector_model.pkl"
    predictor = ReviewDetectorPredictor(model_path=model_path)

    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        predictor.display_prediction(custom_text)
    else:
        run_benchmark_samples(predictor)
        interactive_loop(predictor)

if __name__ == "__main__":
    main()
