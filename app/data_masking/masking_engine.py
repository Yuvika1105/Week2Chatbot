# app/data_masking/masking_engine.py
import re
from dataclasses import dataclass
from typing import Dict
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.custom_recognizers import (
    MGBrandRecognizer,
    ModelLineRecognizer,
    MaterialCodeRecognizer,
    CustomPhoneRecognizer,
    CustomNameRecognizer,
    abbreviate_models
)

# Regex patterns for direct pre-processing (faster & more reliable than Presidio for these)
_MG_BRAND_PATTERN = re.compile(
    r"\bMG[_\s]+Motors?\b|\bMG[_\s]+[A-Z0-9\-]+\b|\bMG\d*\b|\bMG_\b|\bMG\b",
    re.IGNORECASE
)

@dataclass
class MaskingResult:
    masked_text: str
    entity_count: int
    entities_found: Dict[str, int]

class MaskingEngine:
    def __init__(self, policy: MaskingPolicy):
        self.policy = policy

        # Initialize Presidio for PII (PERSON, EMAIL, PHONE, MATERIAL)
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(MaterialCodeRecognizer())
        registry.add_recognizer(CustomPhoneRecognizer())
        registry.add_recognizer(CustomNameRecognizer())

        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()

        # Operators for remaining Presidio entities only
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": token})
            for entity, token in self.policy.replacement_map.items()
            if entity not in ("MG_BRAND", "MG_MODEL")  # handled by pre-processing
        }

        # Which entities Presidio should scan (exclude MG_BRAND/MG_MODEL — done pre-process)
        self.presidio_entities = [
            e for e in self.policy.entity_rules
            if e not in ("MG_BRAND", "MG_MODEL")
        ]

    def mask_text(self, text: str) -> MaskingResult:
        if not isinstance(text, str):
            text = str(text)
        if not text or not text.strip():
            return MaskingResult(text, 0, {})

        entities_found = {}

        # Step 1: Replace MG model names with first+last letter abbreviation
        original = text
        text = abbreviate_models(text)
        if text != original:
            entities_found["MG_MODEL"] = entities_found.get("MG_MODEL", 0) + 1

        # Step 2: Remove MG brand references entirely
        brand_token = self.policy.replacement_map.get("MG_BRAND", "")
        pre_brand = text
        text = _MG_BRAND_PATTERN.sub(brand_token, text)
        if text != pre_brand:
            entities_found["MG_BRAND"] = entities_found.get("MG_BRAND", 0) + 1
            # Clean up stray separators left by empty brand removal
            if brand_token == "":
                text = re.sub(r",\s*,", ",", text)
                text = re.sub(r"^\s*,\s*", "", text)
                text = re.sub(r",\s*$", "", text)
                text = re.sub(r"  +", " ", text).strip()

        # Step 3: Presidio handles remaining PII (PERSON, EMAIL, PHONE, MATERIAL)
        if self.presidio_entities:
            analyzer_results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=self.presidio_entities,
                score_threshold=0.4
            )
            if analyzer_results:
                anonymized = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=self.operators
                )
                text = anonymized.text
                for res in analyzer_results:
                    entities_found[res.entity_type] = entities_found.get(res.entity_type, 0) + 1

        # Step 4: Safety net regex fallbacks for highly critical patterns (PHONE_NUMBER, CREDIT_CARD, US_SSN)
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        if ssn_pattern.search(text):
            text = ssn_pattern.sub(self.policy.replacement_map.get("US_SSN", "<US_SSN>"), text)
            entities_found["US_SSN"] = entities_found.get("US_SSN", 0) + 1

        cc_pattern = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
        if cc_pattern.search(text):
            text = cc_pattern.sub(self.policy.replacement_map.get("CREDIT_CARD", "<CREDIT_CARD>"), text)
            entities_found["CREDIT_CARD"] = entities_found.get("CREDIT_CARD", 0) + 1

        phone_pattern = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        if phone_pattern.search(text):
            text = phone_pattern.sub(self.policy.replacement_map.get("PHONE_NUMBER", "<PHONE_NUMBER>"), text)
            entities_found["PHONE_NUMBER"] = entities_found.get("PHONE_NUMBER", 0) + 1

        return MaskingResult(text, sum(entities_found.values()), entities_found)

    def mask_value(self, column_name: str, value: str) -> str:
        if not value or not isinstance(value, str):
            return str(value)
        # For model columns, abbreviate directly
        target_entity = self.policy.column_rules.get(column_name)
        if target_entity == "MG_MODEL":
            return abbreviate_models(value)
        if target_entity == "MG_BRAND":
            # If explicitly mapped to MG_BRAND column, completely replace with the brand token (empty string)
            return self.policy.replacement_map.get("MG_BRAND", "")
        if target_entity and target_entity in self.policy.replacement_map:
            return self.policy.replacement_map[target_entity]
        return self.mask_text(value).masked_text