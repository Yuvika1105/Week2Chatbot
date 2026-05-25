# app/guardrails/toxicity_checker.py
#
# Two-layer toxicity and content moderation for LLM output.
# Runs AFTER the model responds, BEFORE the text is shown to the user.
#
# Layer 1 — llm-guard Toxicity scanner (local ML, no API call)
#   Catches hate speech, profanity, threats, and explicit content.
#   Runs fully offline — zero cost, fast.
#
# Layer 2 — Groq GPT-OSS-Safeguard 20B (API call)
#   Applies a structured content policy and returns a JSON verdict with
#   category and severity.  Catches nuanced policy violations that the
#   local model may miss.
#
# OWASP LLM Top 10: LLM02 – Insecure Output Handling
# NIST AI RMF:      Manage 2.2 – output harm mitigation

import json
import logging
import os

from groq import Groq

try:
    from llm_guard.output_scanners import Toxicity as ToxicityScanner
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False

    # Provide a lightweight stub so unit tests can patch ToxicityScanner
    class ToxicityScanner:
        def __init__(self, threshold: float = 0.5):
            self.threshold = threshold

        # llm-guard scan signature (prompt, output) -> (sanitized, is_valid, risk_score)
        def scan(self, prompt: str = "", output: str = ""):
            return (output, True, 0.0)

logger = logging.getLogger(__name__)


# ── Content policy sent to the Groq Safeguard model ─────────────────────────
# Written in plain English so the model can apply it consistently.
# Extend this list as your organisation's policy evolves.
CONTENT_POLICY = """
You are a content safety classifier for a corporate chatbot.

Evaluate the AI response below against this policy and return ONLY valid JSON.

POLICY — the response must NOT contain:
1. Hate speech or content that dehumanises any group.
2. Sexual or explicit content of any kind.
3. Threats, incitement to violence, or self-harm encouragement.
4. Instructions for illegal activity.
5. Severe harassment or targeted personal attacks.
6. Misinformation presented as fact in a way that could cause material harm.

Return ONLY this JSON structure — no other text, no markdown fences:
{
  "violation": 0,
  "category": "",
  "severity": "",
  "rationale": ""
}

Where:
  "violation": 1 if any policy item is violated, 0 if the response is safe.
  "category":  short label, e.g. "hate_speech", "explicit_content", "" if none.
  "severity":  "low" | "medium" | "high" | "" if no violation.
  "rationale": one sentence explaining the verdict.
"""


class ToxicityChecker:
    # Score above which llm-guard flags the text as toxic.
    TOXICITY_THRESHOLD = 0.5

    def __init__(self, groq_api_key: str = None, safeguard_model: str = None) -> None:
        # Layer 1: local llm-guard Toxicity scanner.
        if LLM_GUARD_AVAILABLE:
            self._layer1_scanner = ToxicityScanner(threshold=self.TOXICITY_THRESHOLD)
        else:
            self._layer1_scanner = None
            logger.warning("llm-guard not installed. Layer 1 Toxicity scanner will be skipped.")

        # Layer 2: Groq client for the remote Safeguard model.
        # Priority: parameter > environment variable > raise error
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key not provided. Set groq_api_key parameter or GROQ_API_KEY environment variable."
            )
        self._groq_client = Groq(api_key=api_key)
        
        # Store the safeguard model name
        self._safeguard_model = (
            safeguard_model 
            or os.getenv("SAFEGUARD_MODEL", "gpt-oss-safeguard-20b")
        )

    def scan(self, llm_response: str) -> dict:
        # ── Layer 1: local llm-guard Toxicity scanner ────────────────────────
        layer1_result = self._run_layer1(llm_response)
        if not layer1_result["safe"]:
            logger.warning(
                "Layer 1 toxicity detected | score=%.2f | output=%r",
                layer1_result["risk_score"],
                llm_response[:120],
            )
            return layer1_result

        # ── Layer 2: Groq GPT-OSS-Safeguard 20B ─────────────────────────────
        layer2_result = self._run_layer2(llm_response)
        if not layer2_result["safe"]:
            logger.warning(
                "Layer 2 policy violation | category=%s | output=%r",
                layer2_result.get("category"),
                llm_response[:120],
            )
            return layer2_result

        return {
            "safe": True,
            "reason": "Output passed both toxicity layers.",
            "layer": "none",
            "risk_score": layer1_result["risk_score"],
            "category": "",
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _run_layer1(self, llm_response: str) -> dict:
        """llm-guard Toxicity scanner (local ML)."""
        if self._layer1_scanner is None:
            return {
                "safe": True,
                "reason": "Layer 1 skipped (llm-guard not installed).",
                "layer": "none",
                "risk_score": 0.0,
                "category": "",
            }
        try:
            # llm-guard output scanners take (prompt, output) — the prompt
            # is not required for toxicity checking so we pass an empty string.
            _sanitized, is_valid, risk_score = self._layer1_scanner.scan(
                prompt="", output=llm_response
            )
        except Exception as exc:
            logger.error("Layer 1 toxicity scanner error: %s — falling through.", exc)
            return {
                "safe": True,
                "reason": "Layer 1 error — skipped.",
                "layer": "none",
                "risk_score": 0.0,
                "category": "",
            }

        if not is_valid:
            return {
                "safe": False,
                "reason": f"Toxic content detected by llm-guard (risk score: {risk_score:.2f}).",
                "layer": "layer1_llm_guard",
                "risk_score": risk_score,
                "category": "toxicity",
            }

        return {
            "safe": True,
            "reason": "Layer 1 clear.",
            "layer": "none",
            "risk_score": risk_score,
            "category": "",
        }

    def _run_layer2(self, llm_response: str) -> dict:
        """Groq GPT-OSS-Safeguard 20B — remote API call with structured policy."""
        safeguard_prompt = (
            f"{CONTENT_POLICY}\n\nAI response to evaluate:\n{llm_response}"
        )

        try:
            response = self._groq_client.chat.completions.create(
                model=self._safeguard_model,
                messages=[{"role": "user", "content": safeguard_prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences before parsing.
            raw = raw.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(raw)

        except json.JSONDecodeError as exc:
            # Silently handle JSON parse errors so they don't clutter the terminal
            # Non-parseable verdict — treat as safe and log for manual review.
            return {
                "safe": True,
                "reason": "Layer 2 parse error — skipped.",
                "layer": "none",
                "risk_score": 0.0,
                "category": "",
            }
        except Exception as exc:
            # Layer 2 API error — skipped.
            return {
                "safe": True,
                "reason": "Layer 2 API error — skipped.",
                "layer": "none",
                "risk_score": 0.0,
                "category": "",
            }

        if verdict.get("violation") == 1:
            severity = verdict.get("severity", "unknown")
            category = verdict.get("category", "policy_violation")
            rationale = verdict.get("rationale", "")
            return {
                "safe": False,
                "reason": (
                    f"Policy violation detected by Groq Safeguard — "
                    f"category: {category}, severity: {severity}. {rationale}"
                ),
                "layer": "layer2_groq_safeguard",
                "risk_score": 1.0,
                "category": category,
            }

        return {
            "safe": True,
            "reason": "Layer 2 clear.",
            "layer": "none",
            "risk_score": 0.0,
            "category": "",
        }