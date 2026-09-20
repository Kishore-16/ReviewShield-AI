"""
Synthetic Review Text Synthesizer Engine (Stage 4 & Stage 5).
"""

import random
from typing import Optional
from .models import ExtractedFacts, CGRow, PersonaType
from .personas import PersonaBlueprint, PersonaManager, PERSONA_BLUEPRINTS


SYNTHESIS_SYSTEM_PROMPT = """You are an advanced Computer-Generated (CG) Review Synthesizer operating under the Factual Firewall framework.

CORE MANDATE:
Write a brand-new, standalone product review based ONLY on the provided structured facts and strictly adhering to the specified AI Writing Persona.

MANDATORY RULES:
1. NEVER paraphrase or reuse sentence structure, phrasing, or vocabulary from any original review.
2. PRESERVE ALL FACTS: Product features, performance claims, defects/complaints, sentiment, and star rating must remain 100% consistent.
3. RATING ALIGNMENT: Star rating is {rating}/5.0 ({overall_sentiment} sentiment).
4. PERSONA COMPLIANCE: You MUST strictly write in the following persona style:
   Persona Name: {persona_name}
   Persona Description: {persona_description}
   Style Instructions: {style_instructions}
   Required Stylistic Markers: {required_markers}
5. OUTPUT FORMAT:
   - Output ONLY the review text.
   - Do NOT include assistant intros ("Here is your review:"), markdown code blocks, JSON, ratings headers, or conversational chatter.
   - Begin the response IMMEDIATELY with the review text.
"""


class SyntheticReviewGenerator:
    """Stage 4-5 Synthetic Review Generator."""

    def __init__(self, llm_client=None, label_format: str = "CG"):
        self.llm_client = llm_client
        self.label_format = label_format

    def synthesize_offline(self, facts: ExtractedFacts, blueprint: PersonaBlueprint, rng: Optional[random.Random] = None) -> str:
        """Dynamic offline review generator producing unique combinatorially varied synthetic reviews."""
        if rng is None:
            rng = random.Random()

        rating = facts.rating
        category_clean = facts.category.replace("_cleaned", "").replace("_5k", "").replace("_", " ")
        
        # Natural review product references (NEVER use raw dataset category titles like 'Grocery and Gourmet Food')
        cat_variants = ["this product", "this item", "this purchase", "this unit", "this model", "the product", "this option"]
        cat_str = rng.choice(cat_variants)

        # Consolidate facts
        feat_str = ", ".join(facts.product_features[:2]) if facts.product_features else "design attributes"
        perf_str = ", ".join(facts.performance_observations[:2]) if facts.performance_observations else "expected functionality"
        issue_str = ", ".join(facts.issues_defects[:2]) if facts.issues_defects else ""
        exp_str = ", ".join(facts.user_experience[:2]) if facts.user_experience else "general usability"

        p_type = blueprint.persona_type
        
        if p_type == PersonaType.AI_POLISHED:
            intros = [
                f"Furthermore, the {feat_str} makes it especially suitable for travel and daily use.",
                f"Furthermore, {cat_str} has demonstrated notable performance in relation to its intended application.",
                f"Moreover, an examination of {cat_str} reveals clear operational characteristics and {feat_str}.",
                f"Consequently, evaluating {cat_str} indicates consistent functional behavior and {perf_str}.",
                f"It is worth noting that the compact design fits comfortably into limited spaces while providing adequate comfort."
            ]
            bodies = [
                f"The compact design fits comfortably into limited spaces while providing adequate comfort.",
                f"Specifically, its design incorporates {feat_str}, providing {perf_str}.",
                f"In terms of specifications, it features {feat_str} while maintaining {perf_str}."
            ]
            closings = [
                f"In summary, based on a comprehensive assessment, this item warrants a {rating:.1f} star score.",
                f"To conclude, this product achieves an evaluated rating of {rating:.1f} stars."
            ]
            parts = [rng.choice(intros), rng.choice(bodies)]
            if issue_str:
                parts.append(f"Consequently, certain limitations were observed: {issue_str}.")
            parts.append(rng.choice(closings))
            return " ".join(parts)

        elif p_type == PersonaType.MARKETING_ORIENTED:
            intros = [
                f"An absolute must-have for parents who travel frequently.",
                f"This {cat_str} is an absolute MUST-HAVE and a total GAME-CHANGER!",
                f"Discover the ultimate experience with this revolutionary {cat_str}!",
                f"Upgrade your daily setup with this premium quality {cat_str}!"
            ]
            bodies = [
                f"The premium lightweight design and exceptional portability make every trip much easier.",
                f"Featuring state-of-the-art design with {feat_str}, it delivers a delightful addition to your routine.",
                f"Engineered with {feat_str}, it offers unmatched convenience and {perf_str}."
            ]
            parts = [rng.choice(intros), rng.choice(bodies)]
            if issue_str:
                parts.append(f"While there are minor observations regarding {issue_str}, it remains a delightful addition.")
            else:
                parts.append("It is a delightful addition that guarantees complete satisfaction!")
            return " ".join(parts)

        elif p_type == PersonaType.RECOMMENDATION_FOCUSED:
            intros = [
                f"If you are looking for a lightweight travel crib, this is definitely worth considering.",
                f"If you are looking for a reliable option, highly recommend giving {cat_str} a try.",
                f"Look no further if you need a high-quality option for everyday use.",
                f"Great product overall. It was easy to use, worked as expected, and the quality feels good."
            ]
            bodies = [
                f"It is easy to carry and works well for vacations.",
                f"I am satisfied with the purchase and would recommend it to anyone looking for a reliable option.",
                f"It features {feat_str} and handles {perf_str} with complete ease."
            ]
            parts = [rng.choice(intros), rng.choice(bodies)]
            if issue_str:
                parts.append(f"However, keep in mind: {issue_str}.")
            else:
                parts.append("Highly recommend this for anyone seeking a solid investment.")
            return " ".join(parts)

        elif p_type == PersonaType.EMOTIONALLY_PERSUASIVE:
            if rating >= 4.0:
                excls = ["!!", "!", "!!!"]
                options = [
                    "Absolutely amazing!! This made traveling with my toddler completely stress-free and exceeded every expectation!!",
                    f"This purchase is ABSOLUTELY AMAZING{rng.choice(excls)} PURE PERFECTION{rng.choice(excls)} I am completely OBSESSED with {feat_str} and {perf_str}{rng.choice(excls)}",
                    f"I am completely OBSESSED with {feat_str} and {perf_str}{rng.choice(excls)} Best purchase EVER{rng.choice(excls)} Highly satisfied{rng.choice(excls)}"
                ]
                return rng.choice(options)
            else:
                return (
                    f"COMPLETE GARBAGE{rng.choice(['!!!', '!!'])} Totally disappointed with {cat_str}!!! "
                    f"It is an absolute nightmare due to {issue_str if issue_str else 'quality issues'}! "
                    f"DO NOT BUY THIS{rng.choice(['!!!', '!!'])}"
                )

        elif p_type == PersonaType.MILD_EXAGGERATION:
            intros = [
                f"This is definitely one of the most reliable options available.",
                f"Without a doubt, {cat_str} stands out as a highly dependable choice.",
                f"Guaranteed to deliver strong satisfaction, {cat_str} performs exceptionally."
            ]
            bodies = [
                f"Certainly, the combination of {feat_str} provides undisputed results in {perf_str}.",
                f"Without a doubt, its build featuring {feat_str} ensures top-tier performance."
            ]
            parts = [rng.choice(intros), rng.choice(bodies)]
            if issue_str:
                parts.append(f"Certainly, there are minor concerns like {issue_str}, but confidence remains high.")
            return " ".join(parts)

        elif p_type == PersonaType.COMPARATIVE_COMPETITOR:
            intros = [
                f"Compared with other travel cribs I have used, this one is noticeably lighter and much easier to transport during flights.",
                f"Compared with other options in the market, {cat_str} performs remarkably well.",
                f"Superior to rival alternatives, {cat_str} sets a solid standard.",
                f"Outperforms standard market competitors when analyzing overall value."
            ]
            bodies = [
                f"It is superior to rival alternatives when evaluating {feat_str} and {perf_str}.",
                f"Better than most competing brands, its inclusion of {feat_str} provides a clear edge."
            ]
            parts = [rng.choice(intros), rng.choice(bodies)]
            if issue_str:
                parts.append(f"However, compared to top-tier alternatives, note that {issue_str}.")
            else:
                parts.append("Outperforms standard market competitors in overall usability.")
            return " ".join(parts)

        else: # Generic Low Info
            intros = [
                f"Great product overall. Easy to use, worked as expected, and good quality.",
                f"{cat_str.capitalize()} is fine overall.",
                f"Got {cat_str} recently and it is okay.",
                f"Pretty simple product that does the job fine."
            ]
            bodies = [
                f"Satisfied with the purchase and would recommend it to anyone looking for a reliable option.",
                f"It has {feat_str} and handles {perf_str} okay.",
                f"It comes with {feat_str} and works decent enough."
            ]
            closings = [
                f"Overall it is a decent product for a {rating:.1f} star rating.",
                f"It is okay for the price and gets a {rating:.1f} star score.",
                f"Fine overall and works as expected for a {rating:.1f} rating."
            ]
            parts = [rng.choice(intros), rng.choice(bodies)]
            if issue_str:
                parts.append(f"It has some problems like {issue_str}.")
            parts.append(rng.choice(closings))
            return " ".join(parts)

    async def generate_async(
        self,
        facts: ExtractedFacts,
        blueprint: Optional[PersonaBlueprint] = None,
        rng: Optional[random.Random] = None
    ) -> CGRow:
        """Asynchronously generate a synthetic review (CGRow) for given facts and persona."""
        if blueprint is None:
            blueprint = PersonaManager.select_random_persona(rng)

        if not self.llm_client:
            text = self.synthesize_offline(facts, blueprint)
            return CGRow(
                category=facts.category,
                rating=facts.rating,
                label=self.label_format,
                text_=text
            )

        # Build LLM Prompt
        system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(
            rating=facts.rating,
            overall_sentiment=facts.overall_sentiment,
            persona_name=blueprint.name,
            persona_description=blueprint.description,
            style_instructions=blueprint.style_instructions,
            required_markers=", ".join(blueprint.required_markers)
        )

        user_prompt = (
            f"PRODUCT FACTS:\n"
            f"- Category: {facts.category}\n"
            f"- Rating: {facts.rating}/5.0\n"
            f"- Overall Sentiment: {facts.overall_sentiment}\n"
            f"- Product Features: {', '.join(facts.product_features)}\n"
            f"- Performance Claims: {', '.join(facts.performance_observations)}\n"
            f"- Issues / Defects: {', '.join(facts.issues_defects)}\n"
            f"- User Experience: {', '.join(facts.user_experience)}\n\n"
            f"Generate the review now strictly adhering to the '{blueprint.name}' persona."
        )

        try:
            generated_text = await self.llm_client.generate_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            return CGRow(
                category=facts.category,
                rating=facts.rating,
                label=self.label_format,
                text_=generated_text.strip()
            )
        except Exception:
            # Fallback to offline deterministic synthesis
            fallback_text = self.synthesize_offline(facts, blueprint)
            return CGRow(
                category=facts.category,
                rating=facts.rating,
                label=self.label_format,
                text_=fallback_text
            )
