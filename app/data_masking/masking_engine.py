# app/data_masking/masking_engine.py
import re
from dataclasses import dataclass
from typing import Dict
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.custom_recognizers import (
    CustomPhoneRecognizer,
    CustomNameRecognizer
)

@dataclass
class MaskingResult:
    masked_text: str
    entity_count: int
    entities_found: Dict[str, int]

class MaskingEngine:
    def __init__(self, policy: MaskingPolicy):
        self.policy = policy

        # Initialize Presidio for PII (PERSON, EMAIL, PHONE)
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(CustomPhoneRecognizer())
        registry.add_recognizer(CustomNameRecognizer())

        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()

        # Operators for Presidio entities
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": token})
            for entity, token in self.policy.replacement_map.items()
        }

        # Which entities Presidio should scan
        self.presidio_entities = list(self.policy.entity_rules)

    def mask_text(self, text: str) -> MaskingResult:
        if not isinstance(text, str):
            text = str(text)
        if not text or not text.strip():
            return MaskingResult(text, 0, {})

        entities_found = {}

        # Step 1: Presidio handles PII (PERSON, EMAIL, PHONE)
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

        # Step 2: Safety net regex fallbacks for highly critical patterns (PHONE_NUMBER, CREDIT_CARD, US_SSN)
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
        target_entity = self.policy.column_rules.get(column_name)
        if target_entity and target_entity in self.policy.replacement_map:
            return self.policy.replacement_map[target_entity]
        return self.mask_text(value).masked_text