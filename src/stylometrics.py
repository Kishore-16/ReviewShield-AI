"""
Stylometric Feature Extractor Module.
Extracts non-semantic structural, lexical, and stylistic features from review text.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts high-level stylometric indicators:
    1. Personal Pronoun Density (Human Indicator)
    2. Generic AI Formal Template Openers (AI Indicator)
    3. Formal Corporate Review Buzzwords (AI Indicator)
    4. Exclamation & Punctuation Rhythm
    5. Average Word Length and Total Word Count
    """

    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for text in X:
            t_lower = str(text).lower()
            words = t_lower.split()
            n_words = max(len(words), 1)
            n_chars = max(len(str(text)), 1)
            
            # 1. Personal Pronoun Density (Human Indicator)
            pronouns = ["i ", "my ", " me ", "we ", "our ", "us ", "bought", "daughter", "son", "husband", "wife"]
            pronoun_count = sum(t_lower.count(p) for p in pronouns)
            pronoun_ratio = pronoun_count / n_words
            
            # 2. Generic AI Formal Template Openers (AI Indicator)
            ai_openers = [
                "the product ", "the item ", "the quality ", "the design ", 
                "everything ", "setup was ", "reliable performance", 
                "dependable performance", "consistent results", "practical functionality"
            ]
            opener_score = sum(1.0 if t_lower.startswith(p) or p in t_lower else 0.0 for p in ai_openers)
            
            # 3. Formal Corporate Review Buzzwords (AI Indicator)
            ai_buzzwords = [
                "dependable", "reliable", "consistent", "straightforward", 
                "uncomplicated", "usability", "functionality", "routine", 
                "satisfactory", "durability", "everyday use", "regular use"
            ]
            buzzword_count = sum(t_lower.count(b) for b in ai_buzzwords)
            buzzword_ratio = buzzword_count / n_words
            
            # 4. Exclamation & Punctuation Rhythm
            excl_count = text.count("!") / n_chars
            avg_word_len = n_chars / n_words
            
            features.append([
                pronoun_ratio,
                opener_score,
                buzzword_ratio,
                excl_count,
                avg_word_len,
                n_words
            ])
            
        return np.array(features)
