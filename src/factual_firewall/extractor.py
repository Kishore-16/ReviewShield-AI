"""
Factual Firewall Fact Extraction Engine (Stage 1, 2, and 3).
"""

import json
import re
from typing import Dict, Any, Optional
from .models import ORRow, ExtractedFacts


FACT_EXTRACTION_SYSTEM_PROMPT = """You are an expert NLP Factual Information Extractor operating under the Factual Firewall specification.

YOUR SOLE MISSION:
Read the provided raw human review and extract ONLY the underlying factual customer experience into a strict JSON object.

RULES:
1. DO NOT preserve original sentence patterns, phrasing, idioms, vocabulary, or syntax.
2. Extract facts into structured fields:
   - product_features: Specific features, accessories, or attributes mentioned.
   - performance_observations: Observed performance (e.g. battery life, comfort, durability).
   - issues_defects: Malfunctions, damages, missing parts, or complaints.
   - user_experience: General positive or negative observations of use.
   - overall_sentiment: "Positive", "Negative", "Neutral", or "Mixed".
3. Return ONLY valid JSON matching this schema:
{
  "product_features": ["feature 1", ...],
  "performance_observations": ["claim 1", ...],
  "issues_defects": ["issue 1", ...],
  "user_experience": ["experience 1", ...],
  "overall_sentiment": "Positive/Negative/Neutral/Mixed"
}
4. Absolutely NO conversational preambles, explanations, or markdown fences. Output raw JSON ONLY.
"""


class FactualFirewallExtractor:
    """Stage 1-3 Extractor harvesting facts and flushing original text."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract_facts_heuristic(self, row: ORRow) -> ExtractedFacts:
        """Dynamic NLP fact extractor converting review text into clean, non-overlapping keyphrase attributes."""
        text = row.text_
        rating = row.rating
        
        if rating >= 4.0:
            sentiment = "Positive"
        elif rating <= 2.0:
            sentiment = "Negative"
        else:
            sentiment = "Mixed"

        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "which", "this", "that", "these", "those",
            "then", "just", "so", "than", "such", "both", "through", "about", "against", "between", "into", "throughout",
            "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself",
            "we", "our", "ours", "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "is", "was",
            "are", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "doing", "would", "could"
        }

        # Tokenize words and clean punctuation
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [w for w in words if w not in stopwords]
        
        # Extract clean 2-gram keyphrases (BOTH words must be non-stopwords)
        bigrams = []
        for i in range(len(filtered_words) - 1):
            w1, w2 = filtered_words[i], filtered_words[i+1]
            if len(w1) >= 3 and len(w2) >= 3:
                bigrams.append(f"{w1} {w2}")
                    
        # Categorize facts dynamically
        features = []
        performance = []
        issues = []
        experiences = []
        
        for phrase in bigrams:
            if any(w in phrase for w in ["broken", "defective", "fail", "poor", "horrible", "bad", "disappointed", "fake", "hate", "issue", "leak", "tear"]):
                issues.append(phrase)
            elif any(w in phrase for w in ["great", "awesome", "perfect", "good", "love", "easy", "comfortable", "fast", "durable", "taste", "flavor", "quality", "clean"]):
                performance.append(phrase)
            else:
                features.append(phrase)

        if not features:
            features = ["compact design", "build quality"]
        if not performance:
            performance = ["overall performance", "ease of use"]
        if not experiences:
            experiences = [f"user satisfaction"]

        return ExtractedFacts(
            product_features=features[:2],
            performance_observations=performance[:2],
            issues_defects=issues[:2],
            user_experience=experiences[:2],
            overall_sentiment=sentiment,
            rating=rating,
            category=row.category
        )

    async def extract_facts_async(self, row: ORRow) -> ExtractedFacts:
        """Asynchronously extract facts using LLM provider, falling back to heuristic if needed."""
        if not self.llm_client:
            return self.extract_facts_heuristic(row)
            
        prompt = f"Product Category: {row.category}\nStar Rating: {row.rating}\nReview Text:\n{row.text_}"
        try:
            response_str = await self.llm_client.generate_async(
                system_prompt=FACT_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.1
            )
            # Clean markdown code blocks if present
            cleaned_json = re.sub(r'^```(json)?\s*', '', response_str.strip(), flags=re.MULTILINE)
            cleaned_json = re.sub(r'\s*```$', '', cleaned_json.strip(), flags=re.MULTILINE)
            
            data = json.loads(cleaned_json)
            return ExtractedFacts(
                product_features=data.get("product_features", []),
                performance_observations=data.get("performance_observations", []),
                issues_defects=data.get("issues_defects", []),
                user_experience=data.get("user_experience", []),
                overall_sentiment=data.get("overall_sentiment", "Positive" if row.rating >= 4 else "Negative"),
                rating=row.rating,
                category=row.category
            )
        except Exception as e:
            # Fallback to heuristic extraction if LLM fails
            return self.extract_facts_heuristic(row)
