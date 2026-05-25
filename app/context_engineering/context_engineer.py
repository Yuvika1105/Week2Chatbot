from langchain_core.prompts import ChatPromptTemplate
from app.context_engineering.prompt_templates import ROLE_SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES

class ContextEngineer:
    def __init__(self, role: str):
        self.role = role if role in ROLE_SYSTEM_PROMPTS else "guest"

    def build_rag_context(self, query: str, chunks: list, sources: list, history_manager) -> tuple:
        system_base = ROLE_SYSTEM_PROMPTS[self.role]
        
        formatted_chunks = [
            f"--- CHUNK {idx+1} [Source: {src}] ---\n{chk}"
            for idx, (chk, src) in enumerate(zip(chunks, sources))
        ]
        context_str = "\n\n".join(formatted_chunks)

        history_str = "".join([f"User: {t['user']}\nAI: {t['ai']}\n" for t in history_manager.get_window()])

        full_prompt = (
            f"{system_base}\n\n"
            f"EXAMPLES:\n{FEW_SHOT_EXAMPLES}\n"
            f"HISTORY:\n{history_str or 'None'}\n\n"
            f"CONTEXT EVIDENCE:\n{context_str or 'None'}\n\n"
            f"Answer using ONLY context evidence."
        )
        return ChatPromptTemplate.from_messages([("system", full_prompt), ("human", "{query}")]), full_prompt