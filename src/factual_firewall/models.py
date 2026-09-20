"""
Data Schemas and Models for Factual Firewall Dataset Engine.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PersonaType(str, Enum):
    GENERIC_LOW_INFO = "Generic / Low Information"
    AI_POLISHED = "AI Polished"
    RECOMMENDATION_FOCUSED = "Recommendation Focused"
    MARKETING_ORIENTED = "Marketing Oriented"
    EMOTIONALLY_PERSUASIVE = "Emotionally Persuasive"
    MILD_EXAGGERATION = "Mild Exaggeration"
    COMPARATIVE_COMPETITOR = "Comparative / Competitor"


class ORRow(BaseModel):
    """Original Human Review Row Schema."""
    category: str = Field(..., description="Product category")
    rating: float = Field(..., description="Star rating")
    label: str = Field(..., description="Label, typically 'OR' or '0'")
    text_: str = Field(..., description="Original human review text")


class ExtractedFacts(BaseModel):
    """Facts extracted from original review (Stage 2). All wording discarded."""
    product_features: List[str] = Field(default_factory=list, description="Product features/specs")
    performance_observations: List[str] = Field(default_factory=list, description="Performance claims or observations")
    issues_defects: List[str] = Field(default_factory=list, description="Defects, problems, complaints")
    user_experience: List[str] = Field(default_factory=list, description="Positive or negative experiences")
    overall_sentiment: str = Field(..., description="Overall sentiment (Positive, Negative, Neutral, Mixed)")
    rating: float = Field(..., description="Original star rating")
    category: str = Field(..., description="Original category")


class CGRow(BaseModel):
    """Computer-Generated Synthetic Review Row Schema."""
    category: str = Field(..., description="Product category")
    rating: float = Field(..., description="Star rating")
    label: str = Field(default="CG", description="Label, strictly 'CG' or '1'")
    text_: str = Field(..., description="Synthesized review text inside double quotes")


class ValidationResult(BaseModel):
    """Quality Assurance and Validation Result."""
    is_valid: bool = Field(..., description="True if review passes all QA checks")
    leakage_score: float = Field(..., description="N-gram overlap / similarity ratio with original text (0.0 to 1.0)")
    persona_matched: bool = Field(default=True, description="True if review adheres to selected persona markers")
    preamble_detected: bool = Field(default=False, description="True if AI intro or conversational chatter detected")
    markdown_detected: bool = Field(default=False, description="True if markdown tags detected")
    errors: List[str] = Field(default_factory=list, description="List of validation errors or warnings")
    cleaned_text: Optional[str] = Field(default=None, description="Sanitized text after stripping chatter/quotes")
