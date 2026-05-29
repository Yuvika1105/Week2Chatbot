# app/guardrails/pii_detector.py
#
# PII detection and masking using Microsoft Presidio.
# Applied to the LLM output BEFORE it is returned to the user.
# This ensures that even if the model inadvertently echoes PII from
# its training data or from the prompt, the user never sees it raw.
#
# Supported entity types (subset — Presidio detects 50+ by default):
#   PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD,
#   IBAN_CODE, IP_ADDRESS, URL, LOCATION, DATE_TIME, NRP, MEDICAL_LICENSE
#
# Each detected entity is replaced with a placeholder token, e.g.:
#   "Call John on 555-1234, SSN 123-45-6789"
#   → "Call <PERSON> on <PHONE_NUMBER>, SSN <US_SSN>"
#
# OWASP LLM Top 10: LLM06 – Sensitive Information Disclosure
# NIST AI RMF:      Govern 1.6 – data privacy and protection
# app/guardrails/pii_detector.py
import logging
import re
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from app.data_masking.custom_recognizers import CustomPhoneRecognizer, CustomNameRecognizer

logger = logging.getLogger(__name__)

# List of tracking entities to identify and scrub
ENTITIES_TO_DETECT = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
]

class PIIDetector:
    def __init__(self) -> None:
        # Initialize Presidio's underlying scanning and redacting engines
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(CustomPhoneRecognizer())
        registry.add_recognizer(CustomNameRecognizer())
        self._analyzer = AnalyzerEngine(registry=registry)
        self._anonymizer = AnonymizerEngine()

        # Build structural replacement tokens dynamically: e.g., <EMAIL_ADDRESS>
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in ENTITIES_TO_DETECT
        }
        self._operators["DEFAULT"] = OperatorConfig("replace", {"new_value": "<REDACTED>"})

    def mask(self, text: str) -> dict:
        # Handle blank
        if not text or not text.strip():
            return {"masked_text": text, "pii_detected": False, "entity_count": 0, "entities_found": []}

        # Scan the text to locate PII positions
        try:
            analyzer_results = self._analyzer.analyze(
                text=text,
                language="en",
                entities=ENTITIES_TO_DETECT,
                score_threshold=0.3,
            )
        except Exception as exc:
            logger.error(f"Presidio analyzer error: {exc} — returning original text.")
            return {"masked_text": text, "pii_detected": False, "entity_count": 0, "entities_found": []}

        # If no personal data targets are discovered, run a lightweight
        # regex fallback to catch common formats that Presidio may miss.
        if not analyzer_results:
            masked_text = text

            ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
            ssn_matches = len(ssn_pattern.findall(masked_text))
            if ssn_matches:
                masked_text = ssn_pattern.sub("<US_SSN>", masked_text)

            cc_pattern = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
            cc_matches = len(cc_pattern.findall(masked_text))
            if cc_matches:
                masked_text = cc_pattern.sub("<CREDIT_CARD>", masked_text)

            phone_pattern = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
            phone_matches = len(phone_pattern.findall(masked_text))
            if phone_matches:
                masked_text = phone_pattern.sub("<PHONE_NUMBER>", masked_text)

            if ssn_matches or cc_matches or phone_matches:
                entities_found = []
                if ssn_matches:
                    entities_found.append("US_SSN")
                if cc_matches:
                    entities_found.append("CREDIT_CARD")
                if phone_matches:
                    entities_found.append("PHONE_NUMBER")
                entity_count = ssn_matches + cc_matches + phone_matches
                return {
                    "masked_text": masked_text,
                    "pii_detected": True,
                    "entity_count": entity_count,
                    "entities_found": entities_found,
                }

            return {"masked_text": text, "pii_detected": False, "entity_count": 0, "entities_found": []}

        # Step 2: Swap the identified secret words with our placeholder brackets
        try:
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=self._operators,
            )
            masked_text = anonymized.text
        except Exception as exc:
            logger.error(f"Presidio anonymizer error: {exc} — returning original text.")
            masked_text = text

        # Map findings to unique text string tags . set {} so that no dupliction
        entities_found = list({r.entity_type for r in analyzer_results})
        entity_count = len(analyzer_results)

        # NOTE: In some environments Presidio recognizers may be limited.
        # As a safety net, post-process for common patterns such as
        # US SSN and long credit-card numbers that Presidio may miss.
        # Work on the already-anonymized text to avoid double-masking.

        # Mask raw SSN patterns that Presidio didn't catch
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        ssn_matches = len(ssn_pattern.findall(masked_text))
        if ssn_matches:
            masked_text = ssn_pattern.sub("<US_SSN>", masked_text)
            if "US_SSN" not in entities_found:
                entities_found.append("US_SSN")
            entity_count += ssn_matches

        # Mask long numeric sequences that look like credit cards (13-16 digits)
        cc_pattern = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
        cc_matches = len(cc_pattern.findall(masked_text))
        if cc_matches:
            masked_text = cc_pattern.sub("<CREDIT_CARD>", masked_text)
            if "CREDIT_CARD" not in entities_found:
                entities_found.append("CREDIT_CARD")
            entity_count += cc_matches

        # Mask raw phone number patterns that Presidio didn't catch (e.g., 10-digit numbers or standard formats)
        phone_pattern = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        phone_matches = len(phone_pattern.findall(masked_text))
        if phone_matches:
            masked_text = phone_pattern.sub("<PHONE_NUMBER>", masked_text)
            if "PHONE_NUMBER" not in entities_found:
                entities_found.append("PHONE_NUMBER")
            entity_count += phone_matches

        if entities_found:
            logger.info(f"PII masked | entities={entities_found} | original_len={len(text)} | masked_len={len(masked_text)}")

        return {
            "masked_text": masked_text,
            "pii_detected": bool(entities_found),
            "entity_count": entity_count,
            "entities_found": entities_found,
        }


    def detect(self, text: str) -> list[dict]:
        # Diagnostic inspection mode — detects coordinates without changing text
        if not text or not text.strip():
            return []

        try:
            results = self._analyzer.analyze(
                text=text,
                language="en",
                entities=ENTITIES_TO_DETECT,
                score_threshold=0.5,
            )
        except Exception as exc:
            logger.error(f"Presidio detect error: {exc}")
            return []

        return [
            {
                "entity_type": r.entity_type,
                "score": round(r.score, 3),
                "start": r.start,
                "end": r.end,
            }
            for r in results
        ]