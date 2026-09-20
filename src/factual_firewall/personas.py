"""
AI Writing Personas Definitions and Blueprints for Factual Firewall Dataset Engine.
"""

import random
from typing import Dict, Any
from .models import PersonaType


class PersonaBlueprint:
    """Blueprint detailing linguistic instructions and stylistic markers for a persona."""
    def __init__(
        self,
        persona_type: PersonaType,
        name: str,
        description: str,
        style_instructions: str,
        required_markers: list[str],
        example_phrasings: list[str]
    ):
        self.persona_type = persona_type
        self.name = name
        self.description = description
        self.style_instructions = style_instructions
        self.required_markers = required_markers
        self.example_phrasings = example_phrasings


PERSONA_BLUEPRINTS: Dict[PersonaType, PersonaBlueprint] = {
    PersonaType.GENERIC_LOW_INFO: PersonaBlueprint(
        persona_type=PersonaType.GENERIC_LOW_INFO,
        name="Generic / Low Information",
        description="Simple, generic wording with low detail density and repetitive phrasing.",
        style_instructions=(
            "Write a simple, generic review. Use plain, repetitive vocabulary and broad statements. "
            "Keep details minimal while maintaining the core rating and experience. Avoid complex vocabulary or deep analysis."
        ),
        required_markers=["overall", "good", "okay", "fine", "item", "product"],
        example_phrasings=[
            "This product is fine for what it is.",
            "It does the job okay.",
            "Overall it is pretty good and works fine."
        ]
    ),
    PersonaType.AI_POLISHED: PersonaBlueprint(
        persona_type=PersonaType.AI_POLISHED,
        name="AI Polished",
        description="Highly structured, clinical, uniform sentence lengths, and formal transition adverbs.",
        style_instructions=(
            "Write in an extremely polished, formal, and clinical AI style. "
            "Use formal starting transition words at sentence starts such as 'Furthermore,', 'Moreover,', 'Consequently,', 'In summary,', or 'It is worth noting that'. "
            "Maintain uniform sentence lengths and immaculate grammar without emotional slang."
        ),
        required_markers=["Furthermore,", "Moreover,", "Consequently,", "In summary,", "It is worth noting that"],
        example_phrasings=[
            "Furthermore, this product has demonstrated exceptional suitability for travel purposes.",
            "Moreover, its construction ensures consistent utility.",
            "Consequently, it is worth noting that its overall design is highly satisfactory."
        ]
    ),
    PersonaType.RECOMMENDATION_FOCUSED: PersonaBlueprint(
        persona_type=PersonaType.RECOMMENDATION_FOCUSED,
        name="Recommendation Focused",
        description="Structured directly as buying advice or guide targeted directly at the reader.",
        style_instructions=(
            "Write strictly as buyer advice targeted directly at the reader. "
            "Use direct recommendation language such as 'If you are looking for...', 'Highly recommend this for anyone who...', or 'Look no further if you need...'."
        ),
        required_markers=["If you are looking for", "Highly recommend", "Look no further", "Great option for"],
        example_phrasings=[
            "If you are looking for a reliable travel option, this is a great choice.",
            "Highly recommend this for anyone who needs a compact solution.",
            "Look no further if you want a quality item."
        ]
    ),
    PersonaType.MARKETING_ORIENTED: PersonaBlueprint(
        persona_type=PersonaType.MARKETING_ORIENTED,
        name="Marketing Oriented",
        description="Promotional, sales-driven, commercial vocabulary and buzzword emphasis.",
        style_instructions=(
            "Write in a hyper-enthusiastic, marketing-oriented tone. "
            "Use commercial buzzwords such as 'must-have', 'game-changer', 'revolutionary', 'premium quality', and 'delightful addition'."
        ),
        required_markers=["must-have", "game-changer", "revolutionary", "premium quality", "delightful addition"],
        example_phrasings=[
            "This item is a total game-changer and a must-have for parents!",
            "Delivers revolutionary convenience and premium quality.",
            "A delightful addition to your daily routine!"
        ]
    ),
    PersonaType.EMOTIONALLY_PERSUASIVE: PersonaBlueprint(
        persona_type=PersonaType.EMOTIONALLY_PERSUASIVE,
        name="Emotionally Persuasive",
        description="Extreme emotional hyperbole, dramatic tone, and structural punctuation shifts.",
        style_instructions=(
            "Write with intense emotional exaggeration and dramatic hyperbole. "
            "Use multiple exclamation marks (!!!), ALL-CAPS emphasis on key words, and extreme emotional terms like 'ABSOLUTELY AMAZING!!!', 'PURE PERFECTION!!', or 'COMPLETE GARBAGE!!!'."
        ),
        required_markers=["!!!", "ABSOLUTELY", "PURE PERFECTION", "COMPLETE", "OBSESSED"],
        example_phrasings=[
            "This is ABSOLUTELY AMAZING!!! I am completely OBSESSED!!!",
            "PURE PERFECTION!! Best purchase I have EVER made!!!",
            "COMPLETE GARBAGE!!! Totally disappointed with this!!!"
        ]
    ),
    PersonaType.MILD_EXAGGERATION: PersonaBlueprint(
        persona_type=PersonaType.MILD_EXAGGERATION,
        name="Mild Exaggeration",
        description="Elevated certainty, confidence, and strong backing without altering underlying facts.",
        style_instructions=(
            "Write with high certainty and strong confidence while keeping facts exact. "
            "Use strong assurance adverbs such as 'definitely', 'without a doubt', 'guaranteed', and 'certainly'."
        ),
        required_markers=["definitely", "without a doubt", "guaranteed", "certainly"],
        example_phrasings=[
            "This is definitely worth considering for travel.",
            "Without a doubt, it performs better than expected.",
            "Guaranteed to deliver solid results every time."
        ]
    ),
    PersonaType.COMPARATIVE_COMPETITOR: PersonaBlueprint(
        persona_type=PersonaType.COMPARATIVE_COMPETITOR,
        name="Comparative / Competitor",
        description="Framed through product comparisons, alternative references, and competitive positioning.",
        style_instructions=(
            "Write by framing the experience through product comparison and competitive positioning. "
            "Use comparative expressions such as 'Better than...', 'Compared with...', 'Outperforms...', or 'Superior to...'."
        ),
        required_markers=["Better than", "Compared with", "Outperforms", "Superior to"],
        example_phrasings=[
            "Better than most alternatives in this price range.",
            "Compared with other models, this is far easier to use.",
            "Outperforms rival brands in durability and comfort."
        ]
    ),
}


class PersonaManager:
    """Manages persona selection and blueprint retrieval."""
    
    @staticmethod
    def get_all_personas() -> list[PersonaType]:
        return list(PersonaType)

    @staticmethod
    def select_random_persona(rng: random.Random = None) -> PersonaBlueprint:
        """Randomly select exactly one persona blueprint."""
        if rng is None:
            rng = random.Random()
        persona_type = rng.choice(list(PersonaType))
        return PERSONA_BLUEPRINTS[persona_type]

    @staticmethod
    def get_blueprint(persona_type: PersonaType) -> PersonaBlueprint:
        return PERSONA_BLUEPRINTS[persona_type]
