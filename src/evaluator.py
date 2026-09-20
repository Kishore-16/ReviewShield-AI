"""
Model Evaluator & Benchmark Module.
Executes Stratified 80/20 Train-Test benchmarks and Leave-One-Category-Out (LOCO) CV.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "final_120k_or_cg_dataset.csv"

def evaluate_model_pipeline(dataset_path: Path = DEFAULT_DATASET_PATH):
    """Executes standard train-test evaluation and LOCO-CV benchmark."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"[X] Dataset not found at '{dataset_path}'")

    print(f"[*] Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    df['text_'] = df['text_'].fillna("")
    df['target'] = (df['label'] == 'CG').astype(int)

    print("\n" + "="*60)
    print("STAGE 1: TF-IDF FEATURE EXTRACTION (Word + Char N-Grams)")
    print("="*60)
    
    start_time = time.time()
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
    X_word = word_vec.fit_transform(df['text_'])
    
    char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=35000, sublinear_tf=True, analyzer='char')
    X_char = char_vec.fit_transform(df['text_'])
    
    X = hstack([X_word, X_char]).tocsr()
    y = df['target'].values
    
    print(f"[OK] Feature extraction complete in {time.time() - start_time:.2f}s. Matrix shape: {X.shape}")

    print("\n" + "="*60)
    print("STAGE 2: STRATIFIED 80/20 TRAIN-TEST EVALUATION")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    classifiers = {
        "Logistic Regression (L2)": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear Support Vector Machine (LinearSVC)": LinearSVC(C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1)
    }

    for name, clf in classifiers.items():
        print(f"\n---> Training {name} on {X_train.shape[0]} samples...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        print(f"[OK] Trained in {train_time:.2f}s | Test Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=["OR (Human)", "CG (AI/Fake)"], digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    if 'category' in df.columns:
        print("\n" + "="*60)
        print("STAGE 3: UNSEEN DOMAIN GENERALIZATION (Leave-One-Category-Out CV)")
        print("="*60)

        categories = df['category'].unique()
        eval_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        loco_results = []
        
        for held_out_cat in categories:
            train_mask = (df['category'] != held_out_cat)
            test_mask = (df['category'] == held_out_cat)
            
            X_tr, y_tr = X[train_mask.values], y[train_mask.values]
            X_te, y_te = X[test_mask.values], y[test_mask.values]
            
            eval_clf.fit(X_tr, y_tr)
            te_preds = eval_clf.predict(X_te)
            
            cat_acc = accuracy_score(y_te, te_preds)
            cat_f1 = f1_score(y_te, te_preds)
            loco_results.append((held_out_cat, cat_acc, cat_f1))
            
            print(f"  - Held-out Category: '{held_out_cat:<35}' -> Accuracy: {cat_acc*100:.2f}% | F1-Score: {cat_f1:.4f}")

        mean_acc = np.mean([r[1] for r in loco_results])
        mean_f1 = np.mean([r[2] for r in loco_results])
        print("\n" + "="*60)
        print(f"FINAL UNSEEN CATEGORY GENERALIZATION RESULTS:")
        print(f"  - Average Accuracy Across Unseen Categories: {mean_acc*100:.2f}%")
        print(f"  - Average F1-Score Across Unseen Categories: {mean_f1:.4f}")
        print("="*60)

if __name__ == "__main__":
    evaluate_model_pipeline()
