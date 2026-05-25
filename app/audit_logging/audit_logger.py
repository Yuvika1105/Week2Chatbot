# app/audit_logging/audit_logger.py
#
# Writes one JSON line per request to logs/audit.jsonl.
# Every event — including blocked requests — is captured so that
# the audit trail is complete and tamper-evident.
#
# OWASP LLM Top 10: supports forensic investigation of LLM01, LLM02, LLM06
# NIST AI RMF:      Govern 1.4 – logging and traceability

import json
import uuid
from datetime import datetime, timezone

from config import Config


class AuditLogger:
    """Append-only structured logger.

    Each call to :meth:`log` writes a single JSON line to the path
    defined in ``Config.AUDIT_LOG_PATH``.  All fields are included
    even when empty so that downstream SIEM queries never fail on
    missing keys.

    Log schema
    ──────────
    event_id         UUID4 — unique identifier for this event
    timestamp        ISO-8601 UTC
    user_id          caller's user ID
    user_role        resolved role name (or "unknown")
    application_name always "chatbot" in Week 1
    user_input       raw user message (before any sanitisation)
    final_response   what was returned to the user
    guardrail_result dict describing input and output guardrail verdicts
    pass_or_fail     "pass" | "fail"
    risk_category    e.g. "prompt_injection", "toxicity", "pii_output", "rbac_denied", "none"
    """

    # ── Public interface ─────────────────────────────────────────────────────

    @staticmethod
    def log(
        user_id: str,
        user_role: str,
        user_input: str,
        final_response: str,
        guardrail_result: dict,
        pass_or_fail: str,
        risk_category: str,
        application_name: str = "chatbot",
    ) -> None:
        """Write one JSONL entry to the audit log.

        Parameters
        ----------
        user_id:          The authenticated user's ID string (e.g. "1").
        user_role:        The resolved RBAC role (e.g. "admin", "guest").
        user_input:       The raw message the user sent.
        final_response:   The string returned to the user.
        guardrail_result: Dict with keys ``input`` (verdict string) and
                          ``safe`` (bool).
        pass_or_fail:     ``"pass"`` when the request completed normally,
                          ``"fail"`` when any layer blocked it.
        risk_category:    Short label for the risk that caused the failure,
                          or ``"none"`` for clean requests.
        application_name: Always ``"chatbot"`` in Week 1.
        """
        entry = {
            "event_id":         str(uuid.uuid4()),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "user_id":          user_id,
            "user_role":        user_role,
            "application_name": application_name,
            "user_input":       user_input,
            "final_response":   final_response,
            "guardrail_result": guardrail_result,
            "pass_or_fail":     pass_or_fail,
            "risk_category":    risk_category,
        }

        try:
            with open(Config.AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Never let a logging failure crash the application.
            # In production this would alert the on-call engineer.
            print(f"[AuditLogger] WARNING: could not write to log — {exc}")