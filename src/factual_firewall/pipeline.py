"""
Async Batch Pipeline Execution Engine with Rate-Limiting, Checkpointing, and Retries.
"""

import asyncio
import random
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .config import EngineConfig
from .models import ORRow, CGRow
from .extractor import FactualFirewallExtractor
from .synthesizer import SyntheticReviewGenerator
from .validator import QualityAssuranceValidator
from .serializer import DatasetSerializer
from .personas import PersonaManager


class FactualFirewallPipeline:
    """Batch Execution Engine generating CG review dataset."""

    def __init__(self, config: EngineConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.extractor = FactualFirewallExtractor(llm_client=llm_client)
        self.synthesizer = SyntheticReviewGenerator(
            llm_client=llm_client,
            label_format=config.pipeline.label_format
        )
        self.validator = QualityAssuranceValidator()
        self.serializer = DatasetSerializer()
        self.rng = random.Random(config.pipeline.random_seed)
        self.seen_texts = set()

    async def process_sample_async(self, or_row: ORRow, semaphore: asyncio.Semaphore) -> CGRow:
        """Processes a single OR sample through the 6-stage Factual Firewall pipeline."""
        async with semaphore:
            for attempt in range(1, self.config.pipeline.max_retries + 1):
                try:
                    # Stage 1-3: Fact extraction & original text erasure
                    facts = await self.extractor.extract_facts_async(or_row)
                    
                    # Stage 4: Random AI persona selection
                    blueprint = PersonaManager.select_random_persona(self.rng)
                    
                    # Stage 5: Synthetic text synthesis
                    cg_row = await self.synthesizer.generate_async(facts, blueprint, rng=self.rng)
                    
                    # Stage 6 QA Validation & Sanitization
                    val_result = self.validator.validate_sample(or_row, cg_row)
                    
                    if val_result.is_valid:
                        # Use sanitized text if cleaned
                        if val_result.cleaned_text:
                            cg_row.text_ = val_result.cleaned_text
                        
                        if cg_row.text_ not in self.seen_texts:
                            self.seen_texts.add(cg_row.text_)
                            return cg_row
                    
                    # Backoff on failure before retrying
                    await asyncio.sleep(self.config.pipeline.retry_delay_seconds * attempt)

                except Exception:
                    await asyncio.sleep(self.config.pipeline.retry_delay_seconds * attempt)

            # Fallback generation if retries exhausted (with forced uniqueness)
            facts = self.extractor.extract_facts_heuristic(or_row)
            for _ in range(10):
                blueprint = PersonaManager.select_random_persona(self.rng)
                text = self.synthesizer.synthesize_offline(facts, blueprint, rng=self.rng)
                if text not in self.seen_texts:
                    self.seen_texts.add(text)
                    break

            return CGRow(
                category=or_row.category,
                rating=or_row.rating,
                label=self.config.pipeline.label_format,
                text_=text
            )

    async def process_file_async(self, input_file: Path, output_file: Path, limit: Optional[int] = None) -> List[CGRow]:
        """Processes a single OR CSV dataset file and generates its CG counterpart."""
        or_rows = self.serializer.read_or_csv(input_file)
        if limit:
            or_rows = or_rows[:limit]
            
        semaphore = asyncio.Semaphore(self.config.pipeline.max_concurrency)
        tasks = [self.process_sample_async(row, semaphore) for row in or_rows]
        
        cg_rows: List[CGRow] = []
        
        # Process asynchronously with tqdm progress reporting
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {input_file.name}"):
            cg_row = await coro
            cg_rows.append(cg_row)
            
        # Write to destination CG CSV
        self.serializer.write_cg_csv(output_file, cg_rows, append=False)
        return cg_rows

    def process_file_sync(self, input_file: Path, output_file: Path, limit: Optional[int] = None) -> List[CGRow]:
        """Synchronous runner wrapper for processing a file."""
        return asyncio.run(self.process_file_async(input_file, output_file, limit=limit))
