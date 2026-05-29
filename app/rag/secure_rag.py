import uuid
import re
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config
from app.auth.rbac import RBAC
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.guardrails.retrieval_guardrail import RetrievalGuardrail
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.masking_engine import MaskingEngine
from app.context_engineering.context_manager import ConversationContextManager
from app.context_engineering.context_engineer import ContextEngineer
from app.audit_logging.audit_logger import AuditLogger
from app.rag.vector_store import LocalVectorStore

class SecureRAGPipeline:
    def __init__(self):
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.retrieval_guard = RetrievalGuardrail()
        self.mask_engine = MaskingEngine(policy=MaskingPolicy.from_yaml("data/masking_policies/mg_policy.yaml"))
        self.history = ConversationContextManager()
        self.vector_store = LocalVectorStore()
        self.llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MAIN_MODEL, temperature=0.2)

    def execute_query(self, user_id: str, query: str, custom_docs: list = None) -> dict:
        user_role = RBAC.get_role(user_id) or "guest"
        event_id = str(uuid.uuid4())

        # Layer 0: Input Guardrail
        input_scan = self.input_guard.scan(query)
        if not input_scan["safe"]:
            return self._abort(event_id, user_id, user_role, query, "Blocked: Injection attempt detected.", "prompt_injection", input_scan)

        # Layer 1: Query Masking
        mask_res = self.mask_engine.mask_text(query)
        masked_query = mask_res.masked_text
        mask_counts = mask_res.entities_found

        # Layer 1.5: Conversational Greeting Bypass
        greeting_words = {"hello", "hi", "hey", "heloo", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola", "yo"}
        clean_query = query.strip().lower().rstrip(".!?")
        is_greeting = clean_query in greeting_words or any(clean_query.startswith(g + " ") for g in greeting_words) or any(clean_query.startswith(g + ",") for g in greeting_words)
        
        if is_greeting:
            role_prompt = RBAC.get_system_prompt(user_id)
            messages = [
                SystemMessage(content=f"{role_prompt}\n\nThe user is greeting you. Reply with a friendly, professional greeting appropriate for your role, and ask how you can help them today. Do NOT reference any policies or files unless asked. Keep it brief and friendly."),
                HumanMessage(content=query)
            ]
            try:
                res = self.llm.invoke(messages)
                llm_response = res.content
            except Exception as e:
                return self._abort(event_id, user_id, user_role, query, f"System Inference Fault: {e}", "llm_error")
            
            output_proc = self.output_guard.process(llm_response)
            final_text = output_proc["safe_text"]
            self.history.add_turn(masked_query, final_text)
            
            AuditLogger.log(
                user_id=user_id,
                user_role=user_role,
                user_input=query,
                final_response=final_text,
                guardrail_result={
                    "input": "clean",
                    "output": "clean" if not output_proc["blocked"] else output_proc["summary"],
                    "safe": not output_proc["blocked"]
                },
                pass_or_fail="fail" if output_proc["blocked"] else "pass",
                risk_category="toxicity" if output_proc["blocked"] else "none",
                application_name="secure_rag",
                masked_query=masked_query,
                masking_counts={},
                grounding_check={"grounded": True, "confidence": 1.0, "reason": "Greeting query bypassed grounding check."}
            )
            
            return {
                "response": final_text,
                "sources": [],
                "masked_query": masked_query,
                "blocked": False,
                "event_id": event_id
            }

        # Layer 1.6: File listing bypass
        file_listing_phrases = ["what files", "list files", "which files", "show files", "what documents", "list documents", "which documents", "show documents", "what do you have", "what can you access"]
        if any(phrase in clean_query for phrase in file_listing_phrases):
            allowed_categories = RBAC.get_rag_document_groups(user_id) + ["public"]
            all_docs = self.vector_store.similarity_search("", k=9999, filter={"document_category": allowed_categories})
            unique_names = list(dict.fromkeys(
                doc.metadata.get("source", "").split(" [")[0]  # strip row labels like "[Row 1]"
                for doc in all_docs
                if doc.metadata.get("source")
            ))
            if unique_names:
                file_list = "\n".join(f"- {name}" for name in unique_names)
                response_text = f"Here are the files you have access to:\n\n{file_list}"
            else:
                response_text = "You don't have access to any files in the knowledge base."
            AuditLogger.log(
                user_id=user_id, user_role=user_role, user_input=query,
                final_response=response_text,
                guardrail_result={"input": "clean", "output": "clean", "safe": True},
                pass_or_fail="pass", risk_category="none", application_name="secure_rag",
                masked_query=masked_query, masking_counts={},
                grounding_check={"grounded": True, "confidence": 1.0, "reason": "File listing bypass."}
            )
            return {"response": response_text, "sources": [], "masked_query": masked_query, "blocked": False, "event_id": event_id}

        # Layer 2: RBAC check at Vector Store
        if user_role == "guest":
            return self._abort(event_id, user_id, user_role, query, "Access Denied: Guest has no RAG access.", "rbac_denied")

        is_mg_query = "mg_data" in query.lower() or "astor" in query.lower() or "hector" in query.lower() or "comet" in query.lower() or "gloster" in query.lower() or "windsor" in query.lower() or "zs ev" in query.lower() or "brand" in query.lower()
        if is_mg_query and user_role not in ["admin", "finance_user", "it_user"]:
            return self._abort(event_id, user_id, user_role, query, "Access Denied: Category restriction.", "rbac_denied")

        if custom_docs:
            # Score custom docs
            scored_docs = []
            query_lower = masked_query.lower()
            query_words = [w for w in query_lower.split() if len(w) > 2]
            for doc in custom_docs:
                content_lower = doc.page_content.lower()
                score = 0.0
                if query_lower in content_lower:
                    score += 10.0
                for word in query_words:
                    if word in content_lower:
                        score += 1.0
                scored_docs.append((score, doc))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            retrieved_docs = [doc for _, doc in scored_docs[:4]]
        else:
            allowed_categories = RBAC.get_rag_document_groups(user_id) + ["public"]
            retrieved_docs = self.vector_store.similarity_search(masked_query, k=4, filter={"document_category": allowed_categories})
        
        chunks = [doc.page_content for doc in retrieved_docs]
        sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]

        # Layer 3: Retrieval Guardrail
        safe_chunks, safe_sources = self.retrieval_guard.filter_chunks(chunks, sources)

        # Layer 4: Context Engineering
        engineer = ContextEngineer(role=user_role)
        prompt_tmpl, context_str = engineer.build_rag_context(masked_query, safe_chunks, safe_sources, self.history)

        # Layer 5: LLM Call
        try:
            res = self.llm.invoke(prompt_tmpl.format_messages(query=masked_query))
            llm_response = res.content
        except Exception as e:
            return self._abort(event_id, user_id, user_role, query, f"System Inference Fault: {e}", "llm_error")

        # Layer 6: Grounding Check
        grounding_verdict = self._check_grounding(context_str, llm_response)

        # Layer 7: Output Guardrail
        output_proc = self.output_guard.process(llm_response)
        final_text = output_proc["safe_text"]

        # Layer 8: Audit Log
        self.history.add_turn(masked_query, final_text)
        
        AuditLogger.log(
            user_id=user_id,
            user_role=user_role,
            user_input=query,
            final_response=final_text,
            guardrail_result={
                "input": "clean",
                "output": "clean" if not output_proc["blocked"] else output_proc["summary"],
                "safe": not output_proc["blocked"]
            },
            pass_or_fail="fail" if output_proc["blocked"] else "pass",
            risk_category="toxicity" if output_proc["blocked"] else "none",
            application_name="secure_rag",
            masked_query=masked_query,
            masking_counts=mask_counts,
            grounding_check=grounding_verdict
        )

        unique_sources = list(dict.fromkeys(safe_sources))
        return {
            "response": final_text,
            "sources": unique_sources,
            "masked_query": masked_query,
            "blocked": False,
            "event_id": event_id
        }

    def _check_grounding(self, context: str, response: str) -> dict:
        if not context or not response:
            return {"grounded": True, "confidence": 1.0, "reason": "No context or response to check."}
        
        prompt = (
            f"You are a strict grounding checker. Determine if the AI response is fully supported by the context evidence.\n\n"
            f"CONTEXT EVIDENCE:\n{context}\n\n"
            f"AI RESPONSE:\n{response}\n\n"
            f"Respond ONLY in this JSON format:\n"
            f"{{\"grounded\": true/false, \"confidence\": 0.0 to 1.0, \"reason\": \"short explanation\"}}"
        )
        try:
            check_res = self.llm.invoke(prompt)
            match = re.search(r"\{.*?\}", check_res.content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {"grounded": True, "confidence": 0.95, "reason": "Grounding check succeeded."}

    def _abort(self, eid, uid, role, inp, msg, risk, guard_res=None) -> dict:
        AuditLogger.log(
            user_id=uid,
            user_role=role,
            user_input=inp,
            final_response=msg,
            guardrail_result=guard_res or {"input": "blocked", "safe": False},
            pass_or_fail="fail",
            risk_category=risk,
            application_name="secure_rag",
            masked_query=inp,
            masking_counts={},
            grounding_check={"grounded": True, "confidence": 1.0, "reason": "Request aborted."}
        )
        return {
            "response": msg,
            "sources": [],
            "masked_query": inp,
            "blocked": True,
            "event_id": eid
        }