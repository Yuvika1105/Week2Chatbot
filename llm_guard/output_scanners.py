class Toxicity:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    # scan(prompt, output) -> (sanitized_output, is_valid, risk_score)
    def scan(self, prompt: str = "", output: str = ""):
        return (output, True, 0.0)
