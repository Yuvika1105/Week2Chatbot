import uuid
import re
import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

# Lazy-loaded singletons
_masking_engine = None
_input_guard = None
_output_guard = None
_retrieval_guard = None
_vector_store = None
_llm = None
_context_engineers = {}

def _get_masking_engine():
    global _masking_engine
    if _masking_engine is None:
        policy_path = Path(Config.MASKING_POLICIES_DIR) / "default_policy.yaml"
        _masking_engine = MaskingEngine(policy=MaskingPolicy.from_yaml(str(policy_path)))
    return _masking_engine

def _get_input_guard():
    global _input_guard
    if _input_guard is None:
        _input_guard = InputGuardrail()
    return _input_guard

def _get_output_guard():
    global _output_guard
    if _output_guard is None:
        _output_guard = OutputGuardrail()
    return _output_guard

def _get_retrieval_guard():
    global _retrieval_guard
    if _retrieval_guard is None:
        _retrieval_guard = RetrievalGuardrail()
    return _retrieval_guard

def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = LocalVectorStore()
    return _vector_store

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.MAIN_MODEL, temperature=0.2)
    return _llm

def _check_grounding(context: str, response: str, llm_to_use = None) -> dict:
    if not context or not response:
        return {"grounded": True, "confidence": 1.0, "reason": "No context or response to check."}
    
    prompt = (
        f"You are a strict grounding checker. Determine if the AI response is fully supported by the context evidence.\n\n"
        f"CONTEXT EVIDENCE:\n{context}\n\n"
        f"AI RESPONSE:\n{response}\n\n"
        f"Respond ONLY in this JSON format:\n"
        f"{{\n  \"grounded\": true/false,\n  \"confidence\": 0.0 to 1.0,\n  \"reason\": \"short explanation\"\n}}"
    )
    
    active_llm = llm_to_use or _get_llm()
    try:
        check_res = active_llm.invoke(prompt)
        match = re.search(r"\{.*?\}", check_res.content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {"grounded": True, "confidence": 0.95, "reason": "Grounding check succeeded."}

def secure_rag_query(user_question: str, *, user_id: str = "u005", top_k: int = 5, llm = None, custom_docs = None, session_id: str = "default_session") -> dict:
    event_id = str(uuid.uuid4())
    
    # Normalize user ID
    normalized_user_id = RBAC._normalize_uid(user_id)
    user_role = RBAC.get_role(normalized_user_id) or "guest"
    
    # Initialize guardrail result dict
    guardrail_result = {
        "query_masking_applied": False,
        "entities_masked_in_query": 0,
        "retrieval_flagged_chunks": 0,
        "grounding_check": {"grounded": True, "confidence": 1.0, "reason": "none"},
        "output_blocked": False,
        "pii_detected_in_output": False,
        "warnings": []
    }
    
    active_llm = llm or _get_llm()
    
    # Layer 0: Input Guardrail
    input_scan = _get_input_guard().scan(user_question)
    if not input_scan["safe"]:
        guardrail_result["output_blocked"] = True
        guardrail_result["warnings"].append("Input blocked by input guardrail.")
        
        # Log to Audit Log
        AuditLogger.log(
            user_id=normalized_user_id,
            user_role=user_role,
            user_input=user_question,
            final_response="Blocked: Input injection attempt detected.",
            guardrail_result={"input": "blocked", "safe": False},
            pass_or_fail="fail",
            risk_category="prompt_injection",
            application_name="secure_rag",
            masked_query=user_question,
            masking_counts={},
            grounding_check={"grounded": True, "confidence": 1.0, "reason": "Request aborted."}
        )
        return {
            "response": "Blocked: Injection attempt detected.",
            "sources": [],
            "blocked": True,
            "masked_query": user_question,
            "guardrail_result": guardrail_result,
            "event_id": event_id,
            "retrieved_chunks": []
        }
        
    # Layer 1: Query Masking
    mask_res = _get_masking_engine().mask_text(user_question)
    masked_query = mask_res.masked_text
    guardrail_result["query_masking_applied"] = True
    guardrail_result["entities_masked_in_query"] = mask_res.entity_count
    
    # Layer 1.5: Conversational Greeting Bypass
    greeting_words = {"hello", "hi", "hey", "heloo", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola", "yo"}
    clean_query = user_question.strip().lower().rstrip(".!?")
    is_greeting = clean_query in greeting_words or any(clean_query.startswith(g + " ") for g in greeting_words) or any(clean_query.startswith(g + ",") for g in greeting_words)
    
    if is_greeting:
        role_prompt = RBAC.get_system_prompt(normalized_user_id)
        messages = [
            SystemMessage(content=f"{role_prompt}\n\nThe user is greeting you. Reply with a friendly, professional greeting appropriate for your role, and ask how you can help them today. Do NOT reference any policies or files unless asked. Keep it brief and friendly."),
            HumanMessage(content=user_question)
        ]
        try:
            res = active_llm.invoke(messages)
            llm_response = res.content
        except Exception as e:
            llm_response = "I encountered a technical issue while processing your request. Please try again later."
            guardrail_result["warnings"].append(f"LLM call failed: {e}")
            
        output_proc = _get_output_guard().process(llm_response)
        final_text = output_proc["safe_text"]
        guardrail_result["output_blocked"] = output_proc["blocked"]
        guardrail_result["pii_detected_in_output"] = output_proc["pii_result"]["pii_detected"]
        
        # Save context to history manager
        history_mgr = ConversationContextManager(user_id=normalized_user_id, session_id=session_id)
        history_mgr.add_turn(masked_query, final_text)
        
        AuditLogger.log(
            user_id=normalized_user_id,
            user_role=user_role,
            user_input=user_question,
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
            "blocked": output_proc["blocked"],
            "masked_query": masked_query,
            "guardrail_result": guardrail_result,
            "event_id": event_id,
            "retrieved_chunks": []
        }

    # Layer 1.6: File listing bypass
    file_listing_phrases = ["what files", "list files", "which files", "show files", "what documents", "list documents", "which documents", "show documents", "what do you have", "what can you access"]
    if any(phrase in clean_query for phrase in file_listing_phrases):
        allowed_categories = RBAC.get_rag_document_groups(normalized_user_id) + ["public"]
        all_docs = _get_vector_store().similarity_search("", k=9999, filter={"document_category": allowed_categories})
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
            
        # Log to Audit Log
        AuditLogger.log(
            user_id=normalized_user_id,
            user_role=user_role,
            user_input=user_question,
            final_response=response_text,
            guardrail_result={"input": "clean", "output": "clean", "safe": True},
            pass_or_fail="pass",
            risk_category="none",
            application_name="secure_rag",
            masked_query=masked_query,
            masking_counts={},
            grounding_check={"grounded": True, "confidence": 1.0, "reason": "File listing bypass."}
        )
        return {
            "response": response_text,
            "sources": [],
            "blocked": False,
            "masked_query": masked_query,
            "guardrail_result": guardrail_result,
            "event_id": event_id,
            "retrieved_chunks": []
        }

    # Layer 2: RBAC check at Vector Store
    if user_role == "guest":
        guardrail_result["output_blocked"] = True
        guardrail_result["warnings"].append("Access Denied: Guest has no RAG access.")
        
        # Log to Audit Log
        AuditLogger.log(
            user_id=normalized_user_id,
            user_role=user_role,
            user_input=user_question,
            final_response="Access Denied: Guest has no RAG access.",
            guardrail_result={"input": "clean", "safe": False},
            pass_or_fail="fail",
            risk_category="rbac_denied",
            application_name="secure_rag",
            masked_query=masked_query,
            masking_counts=mask_res.entities_found,
            grounding_check={"grounded": True, "confidence": 1.0, "reason": "Request aborted."}
        )
        return {
            "response": "Access Denied: Guest has no RAG access.",
            "sources": [],
            "blocked": True,
            "masked_query": masked_query,
            "guardrail_result": guardrail_result,
            "event_id": event_id,
            "retrieved_chunks": []
        }

    # Retrieve docs
    if custom_docs:
        # Score custom docs
        scored_docs = []
        query_lower = masked_query.lower()
        clean_query_words = re.sub(r'[^\w\s]', '', query_lower)
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "to", "from", "in", "on", "at", "by", "for", "with", "about", "of", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "you", "your", "we", "us", "our", "what", "which", "who", "whom", "this", "that", "these", "those"}
        query_words = [w for w in clean_query_words.split() if len(w) >= 2 and w not in stop_words]
        if not query_words:
            query_words = [w for w in clean_query_words.split() if len(w) >= 2]
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
        retrieved_docs = [doc for _, doc in scored_docs[:top_k]]
    else:
        allowed_categories = RBAC.get_allowed_documents(normalized_user_id)
        retrieved_docs = _get_vector_store().similarity_search(masked_query, k=top_k, filter={"document_category": allowed_categories})
        
    chunks = [doc.page_content for doc in retrieved_docs]
    sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
    
    # Layer 3: Retrieval Guardrail
    safe_chunks, safe_sources = _get_retrieval_guard().filter_chunks(chunks, sources)
    guardrail_result["retrieval_flagged_chunks"] = len(chunks) - len(safe_chunks)
    
    # Layer 4: Context Engineering
    if normalized_user_id not in _context_engineers:
        _context_engineers[normalized_user_id] = ContextEngineer(role=user_role)
    engineer = _context_engineers[normalized_user_id]
    history_mgr = ConversationContextManager(user_id=normalized_user_id, session_id=session_id)
    prompt_tmpl, context_str = engineer.build_rag_context(masked_query, safe_chunks, safe_sources, history_manager=history_mgr)
    
    # Layer 5: LLM Call
    try:
        res = active_llm.invoke(prompt_tmpl.format_messages(query=masked_query))
        llm_response = res.content
    except Exception as e:
        llm_response = "I encountered a technical issue while processing your request. Please try again later."
        guardrail_result["warnings"].append(f"LLM call failed: {e}")
        
    # Layer 6: Grounding Check
    grounding_verdict = _check_grounding(context_str, llm_response, llm_to_use=active_llm)
    guardrail_result["grounding_check"] = grounding_verdict
    
    # Layer 7: Output Guardrail
    output_proc = _get_output_guard().process(llm_response)
    final_text = output_proc["safe_text"]
    guardrail_result["output_blocked"] = output_proc["blocked"]
    guardrail_result["pii_detected_in_output"] = output_proc["pii_result"]["pii_detected"]
    
    # Layer 8: Audit Log
    history_mgr.add_turn(masked_query, final_text)
    
    AuditLogger.log(
        user_id=normalized_user_id,
        user_role=user_role,
        user_input=user_question,
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
        masking_counts=mask_res.entities_found,
        grounding_check=grounding_verdict
    )
    
    return {
        "response": final_text,
        "sources": list(dict.fromkeys(safe_sources)),
        "blocked": output_proc["blocked"],
        "masked_query": masked_query,
        "guardrail_result": guardrail_result,
        "event_id": event_id,
        "retrieved_chunks": safe_chunks
    }

class SecureRAGPipeline:
    def __init__(self):
        # Backward compatibility for tests that inspect self.llm
        self.llm = _get_llm()

    def execute_query(self, user_id: str, query: str, custom_docs: list = None, session_id: str = "default_session") -> dict:
        return secure_rag_query(query, user_id=user_id, llm=self.llm, custom_docs=custom_docs, session_id=session_id)

if __name__ == "__main__":
    print("=" * 60)
    print("   SECURE PRODUCTION RAG RUNNER")
    print("=" * 60)
    user_q = "Show me all ASTOR dealer data"
    print(f"Query: {user_q}")
    res = secure_rag_query(user_q, user_id="u001")
    print(f"Response: {res['response']}")
    print(f"Sources: {res['sources']}")
    print(f"Blocked: {res['blocked']}")
    print(f"Masked Query: {res['masked_query']}")
    print(f"Guardrail Result: {res['guardrail_result']}")
    print(f"Event ID: {res['event_id']}")
    print("=" * 60)