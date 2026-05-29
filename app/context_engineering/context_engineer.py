from langchain_core.prompts import ChatPromptTemplate
from app.context_engineering.prompt_templates import ROLE_SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES
from app.context_engineering.context_manager import ConversationContextManager

class ContextEngineer:
    def __init__(self, role: str):
        self.role = role if role in ROLE_SYSTEM_PROMPTS else "guest"
        self.history = ConversationContextManager()

    def add_to_history(self, user_msg: str, ai_res: str) -> None:
        self.history.add_turn(user_msg, ai_res)

    def build_rag_context(self, query: str, chunks: list, sources: list, history_manager=None) -> tuple:
        system_base = ROLE_SYSTEM_PROMPTS[self.role]
        
        # Format chunks internally — labels are for context only, NOT for the LLM to repeat
        formatted_chunks = [
            f"[{src}]:\n{chk}"
            for idx, (chk, src) in enumerate(zip(chunks, sources))
        ]
        context_str = "\n\n".join(formatted_chunks)

        mgr = history_manager or self.history
        history_str = "".join([f"User: {t['user']}\nAI: {t['ai']}\n" for t in mgr.get_window()])

        full_prompt = (
            f"{system_base}\n\n"
            f"EXAMPLES:\n{FEW_SHOT_EXAMPLES}\n"
            f"CONVERSATION HISTORY:\n{history_str or 'None'}\n\n"
            f"CONTEXT:\n{context_str or 'None'}\n\n"
            f"Answer using ONLY the context above. "
            f"Be clear and concise. Do NOT repeat file names, chunk labels, or internal formatting in your answer."
        )
        return ChatPromptTemplate.from_messages([("system", full_prompt), ("human", "{query}")]), full_prompt