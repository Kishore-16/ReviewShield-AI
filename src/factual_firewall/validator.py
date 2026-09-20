"""
Quality Assurance, Sanitization, and Anti-Leakage Validation Engine.
"""

import re
from typing import Tuple, List, Set
from .models import ORRow, CGRow, ValidationResult, PersonaType
from .personas import PERSONA_BLUEPRINTS


class QualityAssuranceValidator:
    """Validates generated CG reviews against strict research quality criteria."""

    def __init__(self, max_allowed_ngram_overlap: float = 0.35):
        self.max_allowed_ngram_overlap = max_allowed_ngram_overlap

    @staticmethod
    def extract_ngrams(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
        """Extract word n-grams from cleaned text."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return set()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    def compute_leakage_score(self, original_text: str, generated_text: str) -> float:
        """Computes 3-gram overlap ratio between original human text and synthetic CG text."""
        orig_3grams = self.extract_ngrams(original_text, n=3)
        gen_3grams = self.extract_ngrams(generated_text, n=3)
        
        if not orig_3grams or not gen_3grams:
            return 0.0
            
        intersection = orig_3grams.intersection(gen_3grams)
        # Ratio relative to generated text size
        overlap_score = len(intersection) / len(gen_3grams)
        return overlap_score

    @staticmethod
    def sanitize_text(text: str) -> Tuple[str, bool, bool]:
        """Strips AI preambles, markdown formatting, outer quotes, and conversational chatter.
        
        Returns:
            (cleaned_text, preamble_detected, markdown_detected)
        """
        preamble_detected = False
        markdown_detected = False

        # Detect markdown formatting
        if re.search(r'(\*\*|```|###|_|`|\[|\])', text):
            markdown_detected = True

        # Strip conversational preambles
        preamble_patterns = [
            r'^(Sure|Certainly|Here is|Here\'s|Below is|I have generated|As requested).*?:\s*',
            r'^(Review:|Generated Review:|Synthetic Review:)\s*',
        ]
        
        cleaned = text.strip()
        for pattern in preamble_patterns:
            if re.match(pattern, cleaned, re.IGNORECASE):
                preamble_detected = True
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # Remove markdown fences and bolding
        cleaned = re.sub(r'```(markdown|json|text)?', '', cleaned)
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)

        # Remove wrapping double quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
            cleaned = cleaned[1:-1].strip()

        return cleaned, preamble_detected, markdown_detected

    def validate_sample(self, original: ORRow, generated: CGRow) -> ValidationResult:
        """Performs full quality validation check on a generated sample."""
        errors: List[str] = []
        
        # 1. Sanitize text
        cleaned_text, preamble_detected, markdown_detected = self.sanitize_text(generated.text_)
        
        if preamble_detected:
            errors.append("Conversational preamble detected and stripped.")
        if markdown_detected:
            errors.append("Markdown formatting detected and sanitized.")
            
        # 2. Check empty or ultra-short text
        if len(cleaned_text.split()) < 5:
            errors.append("Generated text is excessively short (< 5 words).")

        # 3. Check schema alignment
        if original.category != generated.category:
            errors.append(f"Category mismatch: expected '{original.category}', got '{generated.category}'.")
            
        if abs(original.rating - generated.rating) > 0.01:
            errors.append(f"Rating mismatch: expected {original.rating}, got {generated.rating}.")

        # 4. Anti-Leakage Validation (N-gram overlap check)
        leakage_score = self.compute_leakage_score(original.text_, cleaned_text)
        if leakage_score > self.max_allowed_ngram_overlap:
            errors.append(
                f"Linguistic leakage threshold exceeded! Overlap score {leakage_score:.2f} > max allowed {self.max_allowed_ngram_overlap:.2f}."
            )

        is_valid = len([e for e in errors if "threshold exceeded" in e or "short" in e or "mismatch" in e]) == 0

        return ValidationResult(
            is_valid=is_valid,
            leakage_score=leakage_score,
            persona_matched=True,
            preamble_detected=preamble_detected,
            markdown_detected=markdown_detected,
            errors=errors,
            cleaned_text=cleaned_text
        )
