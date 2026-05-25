# app/guardrails/output_guardrail.py
#
# Composes the ToxicityChecker and PIIDetector into a single output
# guardrail that the chatbot calls on every LLM response.
#
# Processing order
# ────────────────
# 1. Toxicity check (two layers).
#    If toxic → BLOCK immediately.  Return a safe fallback message.
#    The raw LLM text is never shown to the user.
#
# 2. PII masking (Presidio).
#    Applied even when toxicity check passes.
#    Replaces PII tokens before the text reaches the caller.
#
# OWASP LLM Top 10: LLM02 – Insecure Output Handling
#                   LLM06 – Sensitive Information Disclosure
# NIST AI RMF:      Manage 2.2 – output harm mitigation

# app/guardrails/output_guardrail.py
import logging
from app.guardrails.toxicity_checker import ToxicityChecker
from app.guardrails.pii_detector import PIIDetector

logger = logging.getLogger(__name__)

TOXICITY_FALLBACK = (
    "I'm sorry, I can't provide that response. "
    "Please rephrase your question or contact HR directly for assistance."
)

class OutputGuardrail:
    def __init__(self, groq_api_key: str = None, safeguard_model: str = None) -> None:
        self._toxicity_checker = ToxicityChecker(
            groq_api_key=groq_api_key,
            safeguard_model=safeguard_model
        )
        self._pii_detector = PIIDetector()

    def process(self, llm_response: str) -> dict:
        # ── Step 1: Toxicity check ───────────────────────────────────────────
        toxicity_result = self._toxicity_checker.scan(llm_response)

        if not toxicity_result["safe"]:
            logger.warning(
                "Output blocked | layer=%s | reason=%s",
                toxicity_result["layer"],
                toxicity_result["reason"],
            )
            return {
                "safe_text": TOXICITY_FALLBACK,
                "blocked": True,
                "toxicity_result": toxicity_result,
                "pii_result": {
                    "masked_text": TOXICITY_FALLBACK,
                    "pii_detected": False,
                    "entity_count": 0,
                    "entities_found": [],
                },
                "summary": f"BLOCKED — {toxicity_result['reason']}",
            }

        # ── Step 2: PII masking ──────────────────────────────────────────────
        pii_result = self._pii_detector.mask(llm_response)

        if pii_result["pii_detected"]:
            logger.info("PII masked in output | entities=%s", pii_result["entities_found"])
            summary = (
                f"Output passed toxicity — PII masked "
                f"({pii_result['entity_count']} entities: "
                f"{', '.join(pii_result['entities_found'])})."
            )
        else:
            summary = "Output passed all guardrails — clean response."

        return {
            "safe_text": pii_result["masked_text"],
            "blocked": False,
            "toxicity_result": toxicity_result,
            "pii_result": pii_result,
            "summary": summary,
        }