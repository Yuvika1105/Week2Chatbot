# app/data_masking/masking_engine.py
from dataclasses import dataclass
from typing import Dict
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.custom_recognizers import (
    MGBrandRecognizer,
    ModelLineRecognizer,
    MaterialCodeRecognizer
)

@dataclass
class MaskingResult:
    masked_text: str
    entity_count: int
    entities_found: Dict[str, int]

class MaskingEngine:
    def __init__(self, policy: MaskingPolicy):
        self.policy = policy
        
        # Initialize Presidio's underlying registry manager
        registry = RecognizerRegistry()
        
        # Load all default built-in PII detectors (Email, Phones, etc.)
        registry.load_predefined_recognizers()
        
        # Safely register our custom MG Motors brand, model, and material scanners
        registry.add_recognizer(MGBrandRecognizer())
        registry.add_recognizer(ModelLineRecognizer())
        registry.add_recognizer(MaterialCodeRecognizer())
        
        # Wire the populated registry configuration right into the Analyzer Engine
        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()
        
        # Map dynamic entity masks from the YAML policy
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": token})
            for entity, token in self.policy.replacement_map.items()
        }

    def mask_text(self, text: str) -> MaskingResult:
        if not text or not text.strip():
            return MaskingResult(text, 0, {})
        
        analyzer_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.policy.entity_rules,
            score_threshold=0.4
        )
        if not analyzer_results:
            return MaskingResult(text, 0, {})

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=self.operators
        )
        
        entities_found = {}
        for res in analyzer_results:
            entities_found[res.entity_type] = entities_found.get(res.entity_type, 0) + 1

        return MaskingResult(anonymized.text, len(analyzer_results), entities_found)

    def mask_value(self, column_name: str, value: str) -> str:
        if not value or not isinstance(value, str):
            return str(value)
        target_entity = self.policy.column_rules.get(column_name)
        if target_entity and target_entity in self.policy.replacement_map:
            return self.policy.replacement_map[target_entity]
        return self.mask_text(value).masked_text