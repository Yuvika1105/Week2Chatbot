import uuid
from langchain_groq import ChatGroq
from config import Config
from app.auth.rbac import RBAC
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.masking_engine import MaskingEngine
from app.context_engineering.context_manager import ConversationContextManager
from app.context_engineering.context_engineer import ContextEngineer
from app.audit_logging.audit_logger import AuditLogger

class SecureRAGPipeline:
    def __init__(self):
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.mask_engine = MaskingEngine(policy=MaskingPolicy.from_yaml("data/masking_policies/mg_policy.yaml"))
        self.history = ConversationContextManager()
        self.llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MAIN_MODEL, temperature=0.2)

    def execute_query(self, user_id: str, query: str) -> dict:
        user_role = RBAC.get_role(user_id) or "guest"
        event_id = str(uuid.uuid4())

        if not self.input_guard.scan(query)["safe"]:
            return self._abort(event_id, user_id, user_role, query, "Blocked: Injection attempt detected.", "prompt_injection")

        masked_query = self.mask_engine.mask_text(query).masked_text

        if user_role not in ["admin", "finance_user", "it_user"] and "mg_data" in query.lower():
            return self._abort(event_id, user_id, user_role, query, "Access Denied: Category restriction.", "rbac_denied")

        chunks = ["Sample system reference for <MODEL_LINE> variant distributions."]
        sources = ["mg_policy_doc.txt"]

        engineer = ContextEngineer(role=user_role)
        prompt_tmpl, _ = engineer.build_rag_context(masked_query, chunks, sources, self.history)

        try:
            res = self.llm.invoke(prompt_tmpl.format_messages(query=masked_query))
            final_text = self.output_guard.process(res.content)["safe_text"]
        except Exception as e:
            return self._abort(event_id, user_id, user_role, query, f"System Inference Fault: {e}", "llm_error")

        self.history.add_turn(masked_query, final_text)
        AuditLogger.log(user_id, user_role, query, final_text, {"status": "passed"}, "pass", "none")
        return {"response": final_text, "sources": sources, "masked_query": masked_query, "blocked": False, "event_id": event_id}

    def _abort(self, eid, uid, role, inp, msg, risk) -> dict:
        AuditLogger.log(uid, role, inp, msg, {"status": "blocked"}, "fail", risk)
        return {"response": msg, "sources": [], "masked_query": inp, "blocked": True, "event_id": eid}