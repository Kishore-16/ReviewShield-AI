"""
Benchmark Evaluation Script.
Evaluates 80/20 train-test metrics and Leave-One-Category-Out cross-validation.

Usage:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import evaluate_model_pipeline

def main():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "final_120k_or_cg_dataset.csv"
    evaluate_model_pipeline(dataset_path=dataset_path)

if __name__ == "__main__":
    main()
