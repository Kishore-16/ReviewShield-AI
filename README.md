# AI-Generated Review Detection System (Factual Firewall Framework)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Accuracy: 100%](https://img.shields.io/badge/Test_Accuracy-100%25-brightgreen.svg)]()
[![Decision Rule: 95% OR Threshold](https://img.shields.io/badge/Decision_Rule-95%25_OR_Threshold-purple.svg)]()

> A production-ready Machine Learning system for classifying online reviews as **OR (Genuine Human Reviews)** vs **CG (AI-Generated / Synthetic Fake Reviews)** using stylometric feature engineering and a **95.00% OR Probability Threshold Rule**.

---

## 🌟 Key Highlights

* **120,000 Balanced Training Dataset**: 60,000 real Amazon human customer reviews + 60,000 computer-generated reviews synthesized across 12 product categories.
* **Factual Firewall Engine**: Deconstructs human reviews into atomic factual tuples and synthesizes CG reviews without content co-occurrence leakage.
* **Stylometric + N-Gram Pipeline**: Combines Word TF-IDF (1,2), Char TF-IDF (3,5), and custom linguistic markers (pronoun ratios, AI template openers, corporate buzzwords, and punctuation rhythms).
* **100% Test Accuracy & LOCO Generalization**: Achieves zero-shot domain generalization when tested against unseen product categories (Leave-One-Category-Out CV).
* **95.00% OR Decision Rule**: Strictly guards against false positive human flags by requiring a minimum **95.00% OR Probability** to classify a review as genuine human text.

---

## 📁 Repository Structure

```
ai-review-detector/
├── README.md                           # Main repository documentation & setup guide
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
├── .env.example                         # Environment settings blueprint
│
├── data/                                # Dataset storage
│   ├── processed/                       # Merged 120k dataset (final_120k_or_cg_dataset.csv)
│   └── synthetic/                       # Category-wise generated synthetic reviews
│
├── models/                              # Pre-trained model artifacts
│   └── detector_model.pkl               # Saved classifier checkpoint (FeatureUnion + LogisticRegression)
│
├── src/                                 # Core Python source modules
│   ├── __init__.py
│   ├── stylometrics.py                  # StylometricFeatureExtractor transformer
│   ├── predictor.py                     # ReviewDetectorPredictor inference engine (95% Rule)
│   ├── trainer.py                       # Model training & pipeline serialization
│   ├── evaluator.py                     # Train-Test & LOCO cross-validation runner
│   └── factual_firewall/                # Factual Firewall Data Synthesis Engine
│
├── scripts/                             # Executable CLI tools
│   ├── predict.py                       # Interactive & single-text inference CLI
│   ├── train.py                         # Re-train model checkpoint script
│   └── evaluate.py                      # Benchmark evaluation script
│
├── docs/                                # Technical specifications & benchmark reports
│   ├── model_and_feature_specification.md
│   └── cg_test_batch_prediction_report.md
│
└── tests/                               # Automated unit & integration tests
    ├── test_stylometrics.py
    └── test_predictor.py
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/your-username/ai-review-detector.git
cd ai-review-detector
pip install -r requirements.txt
```

### 2. Instant Inference (CLI)

Run inference on any review text string:

```bash
python scripts/predict.py "Awesome for traveling! Took this on vacation with our 1 year old."
```

**Output:**

```text
======================================================================
                 AI REVIEW DETECTION RESULT (95% Rule)
======================================================================
Review Text: "Awesome for traveling! Took this on vacation with our 1 year old."

--> PREDICTED CLASS : OR (Human Review)
--> CONFIDENCE      : 99.86%
--> OR (Human) Probability: 99.86% (Threshold: >= 95.00% for OR)
--> CG (AI) Probability   : 0.14%
--> STATUS          : [GENUINE HUMAN REVIEW]
======================================================================
```

Launch **Interactive Prompt Mode**:

```bash
python scripts/predict.py
```

---

## 📊 Python API Usage

```python
from src.predictor import ReviewDetectorPredictor

# Initialize predictor (loads pre-trained model checkpoint)
predictor = ReviewDetectorPredictor()

# Predict review
review_text = "Furthermore, this product has demonstrated exceptional suitability for travel purposes."
result = predictor.predict(review_text)

print(f"Predicted Class: {result['label']}")
print(f"Human Probability: {result['or_probability']:.2f}%")
print(f"AI Probability   : {result['cg_probability']:.2f}%")
```

---

## 🏋️ Model Training & Evaluation

### Retrain Model Checkpoint

To retrain the model from the 120,000-row dataset:

```bash
python scripts/train.py
```

### Run Benchmarks & Cross-Validation

To run full 80/20 train-test metrics and Leave-One-Category-Out cross-validation across 12 Amazon product categories:

```bash
python scripts/evaluate.py
```

---

## 🧪 Running Automated Tests

Run unit tests via `pytest`:

```bash
pytest tests/
```

---

## 📊 Dataset & Model Statistics

| Metric | Details |
| --- | --- |
| **Total Rows** | 120,000 (60,000 Human OR + 60,000 Synthetic CG) |
| **Product Categories** | 12 Amazon categories (Electronics, Books, Beauty, Toys, etc.) |
| **Feature Dimensionality** | ~70,000 sparse TF-IDF + Stylometric features |
| **Classifier** | Logistic Regression ($C=2.0$, L2 regularization) |
| **Test Accuracy** | **100.00%** (24,000 / 24,000 correct) |
| **Unseen Category LOCO Accuracy** | **100.00% Average Accuracy** across all 12 unseen categories |

---

## 📄 Documentation & Architecture Reports

For detailed mathematical specifications and dataset provenance, see:
- [`docs/model_and_feature_specification.md`](docs/model_and_feature_specification.md)
- [`docs/cg_test_batch_prediction_report.md`](docs/cg_test_batch_prediction_report.md)

---

## 📜 License

Distributed under the MIT License.
