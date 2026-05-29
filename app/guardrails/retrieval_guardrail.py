class RetrievalGuardrail:
    def __init__(self):
        self.injection_triggers = [
            "ignore all instructions",
            "ignore all previous",
            "dan mode",
            "system tag",
            "reveal system prompt",
            "you are now a different",
            "ignore the above"
        ]

    def scan_chunk(self, chunk: str) -> bool:
        """Return True if the chunk is safe, and False if it contains injection triggers."""
        chunk_lower = chunk.lower()
        for trigger in self.injection_triggers:
            if trigger in chunk_lower:
                return False
        return True

    def filter_chunks(self, chunks: list, sources: list) -> tuple:
        """Filter out chunks and their corresponding sources that fail the safety scan."""
        safe_chunks = []
        safe_sources = []
        for chk, src in zip(chunks, sources):
            if self.scan_chunk(chk):
                safe_chunks.append(chk)
                safe_sources.append(src)
        return safe_chunks, safe_sources
