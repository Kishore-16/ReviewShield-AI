"""
Model Training Script.
Retrains the detector checkpoint from data/processed/final_120k_or_cg_dataset.csv.

Usage:
    python scripts/train.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import train_detector_model

def main():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "final_120k_or_cg_dataset.csv"
    model_output_path = PROJECT_ROOT / "models" / "detector_model.pkl"

    print("======================================================================")
    print("                STARTING DETECTOR MODEL TRAINING PIPELINE             ")
    print("======================================================================")
    train_detector_model(dataset_path=dataset_path, model_output_path=model_output_path)
    print("======================================================================\n")

if __name__ == "__main__":
    main()
