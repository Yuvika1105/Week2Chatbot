class ConversationContextManager:
    def __init__(self):
        self.history = []

    def add_turn(self, user_msg: str, ai_res: str) -> None:
        self.history.append({"user": user_msg, "ai": ai_res})

    def get_window(self, max_tokens: int = 800) -> list:
        max_chars = max_tokens * 4
        current_chars = 0
        selected = []
        for turn in reversed(self.history):
            turn_str = f"User: {turn['user']}\nAI: {turn['ai']}\n"
            if current_chars + len(turn_str) > max_chars: break
            selected.insert(0, turn)
            current_chars += len(turn_str)
        return selected