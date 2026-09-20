"""
Configuration Module for Factual Firewall Dataset Engine.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration for input datasets and output paths."""
    input_dir: Path = Field(default=Path("."), description="Directory containing input OR CSV files")
    output_dir: Path = Field(default=Path("./cg_outputs"), description="Directory to store generated CG CSV files")
    merged_output_path: Path = Field(default=Path("./final_120k_or_cg_dataset.csv"), description="Path for final merged dataset")
    checkpoint_dir: Path = Field(default=Path("./checkpoints"), description="Directory for saving intermediate progress")


class PipelineConfig(BaseModel):
    """Pipeline execution and concurrency configurations."""
    batch_size: int = Field(default=50, description="Batch size for pipeline processing")
    max_concurrency: int = Field(default=10, description="Maximum concurrent LLM requests")
    max_retries: int = Field(default=3, description="Maximum retries per sample on failure or validation error")
    retry_delay_seconds: float = Field(default=2.0, description="Initial backoff delay in seconds")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    label_format: str = Field(default="CG", description="Label format for generated reviews: 'CG' or '1'")
    or_label_format: str = Field(default="OR", description="Label format for original human reviews: 'OR' or '0'")


class LLMConfig(BaseModel):
    """LLM provider settings."""
    provider: str = Field(default="openai", description="LLM provider: 'openai', 'gemini', 'anthropic', 'ollama', 'groq'")
    model_name: str = Field(default="gpt-4o-mini", description="Model name to use for generation")
    api_key: Optional[str] = Field(default=None, description="API Key for provider (reads from ENV if None)")
    temperature: float = Field(default=0.7, description="Sampling temperature for synthesis")
    top_p: float = Field(default=0.9, description="Top-P nucleus sampling")
    max_tokens: int = Field(default=500, description="Max tokens for review generation")


class EngineConfig(BaseModel):
    """Master Engine Configuration."""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
