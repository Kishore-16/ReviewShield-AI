"""
Unit and Integration Tests for Factual Firewall Dataset Engine.
"""

import pytest
from pathlib import Path
from factual_firewall_engine.models import ORRow, CGRow, ExtractedFacts, PersonaType
from factual_firewall_engine.personas import PersonaManager, PERSONA_BLUEPRINTS
from factual_firewall_engine.extractor import FactualFirewallExtractor
from factual_firewall_engine.synthesizer import SyntheticReviewGenerator
from factual_firewall_engine.validator import QualityAssuranceValidator
from factual_firewall_engine.serializer import DatasetSerializer
from factual_firewall_engine.config import EngineConfig
from factual_firewall_engine.pipeline import FactualFirewallPipeline


def test_persona_manager():
    """Verify all 7 personas exist and random selection works."""
    personas = PersonaManager.get_all_personas()
    assert len(personas) == 7
    
    blueprint = PersonaManager.select_random_persona()
    assert blueprint.persona_type in PersonaType
    assert len(blueprint.required_markers) > 0


def test_fact_extractor_heuristic():
    """Test stage 1-3 fact extraction heuristic."""
    or_sample = ORRow(
        category="Baby_Products_cleaned",
        rating=5.0,
        label="OR",
        text_="Awesome for traveling!!! Took this on vacation with our 1 year old. It was perfect for a small space and lightweight."
    )
    extractor = FactualFirewallExtractor()
    facts = extractor.extract_facts_heuristic(or_sample)
    
    assert facts.rating == 5.0
    assert facts.category == "Baby_Products_cleaned"
    assert facts.overall_sentiment == "Positive"
    assert len(facts.product_features) > 0 or len(facts.performance_observations) > 0


def test_synthetic_generator_offline():
    """Test stage 4-5 synthesis across all 7 personas offline."""
    facts = ExtractedFacts(
        product_features=["compact design", "lightweight frame"],
        performance_observations=["fits in carry-on bag", "easy to carry"],
        issues_defects=[],
        user_experience=["great for vacations"],
        overall_sentiment="Positive",
        rating=5.0,
        category="Baby_Products_cleaned"
    )
    
    generator = SyntheticReviewGenerator()
    
    for persona_type in PersonaType:
        blueprint = PERSONA_BLUEPRINTS[persona_type]
        text = generator.synthesize_offline(facts, blueprint)
        assert len(text) > 10
        assert facts.category.replace("_cleaned", "").replace("_", " ") in text.lower() or "product" in text.lower() or "item" in text.lower() or "option" in text.lower() or "decals" in text.lower() or "this" in text.lower()


def test_quality_assurance_validator():
    """Test QA validation, anti-preamble, and anti-leakage checks."""
    validator = QualityAssuranceValidator()
    
    or_sample = ORRow(
        category="Baby_Products_cleaned",
        rating=5.0,
        label="OR",
        text_="Awesome for traveling!!! Took this on vacation with our 1 year old."
    )
    
    cg_sample = CGRow(
        category="Baby_Products_cleaned",
        rating=5.0,
        label="CG",
        text_="Here is your review: Furthermore, this product is ideal for travel purposes."
    )
    
    val = validator.validate_sample(or_sample, cg_sample)
    assert val.preamble_detected is True
    assert val.cleaned_text.startswith("Furthermore")
    assert val.is_valid is True


def test_serializer_and_pipeline(tmp_path: Path):
    """Test reading OR CSV, pipeline generation, and writing CG CSV."""
    # Write mock OR CSV file
    mock_or_csv = tmp_path / "test_OR.csv"
    with open(mock_or_csv, "w", encoding="utf-8") as f:
        f.write("category,rating,label,text_\n")
        f.write('Baby_Products_cleaned,5.0,OR,"Awesome item, very lightweight and portable."\n')
        f.write('Baby_Products_cleaned,1.0,OR,"Defective zipper broke immediately."\n')

    # Run Pipeline
    config = EngineConfig()
    output_cg_csv = tmp_path / "test_CG.csv"
    
    pipeline = FactualFirewallPipeline(config)
    cg_rows = pipeline.process_file_sync(mock_or_csv, output_cg_csv)
    
    assert len(cg_rows) == 2
    assert output_cg_csv.exists()
    
    # Read back generated CG CSV
    read_cg = DatasetSerializer.read_or_csv(output_cg_csv)
    assert len(read_cg) == 2
    assert read_cg[0].label == "CG"
