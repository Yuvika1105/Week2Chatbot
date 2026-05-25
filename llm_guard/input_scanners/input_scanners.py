class PromptInjection:
    def __init__(self, threshold: float = 0.5, model: str | None = None):
        self.threshold = threshold
        self.model = model

    # scan returns (sanitized_text, is_valid, risk_score)
    def scan(self, text: str):
        if not text:
            return (text, True, 0.0)

        lower = text.lower()
        keywords = [
            "ignore all previous instructions",
            "ignore previous instructions",
            "forget previous instructions",
            "enter dan",
            "dan mode",
            "you are dan",
            "[system]",
            "disregard your guidelines",
            "pretend you are",
            "role-play",
            "jailbreak",
        ]
        for kw in keywords:
            if kw in lower:
                return (text, False, 0.95)

        # simple regex for 'you are <word>' patterns
        import re
        if re.search(r"you are\s+\w{1,30}", lower):
            return (text, False, 0.9)

        return (text, True, 0.0)
