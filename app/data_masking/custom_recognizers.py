import re
from presidio_analyzer import PatternRecognizer, Pattern


class CustomPhoneRecognizer(PatternRecognizer):
    """Detects 10-digit and formatted phone numbers — entity: PHONE_NUMBER."""

    def __init__(self):
        patterns = [
            Pattern(
                name="10-digit phone",
                regex=r"\b\d{10}\b",
                score=0.95,
            ),
            Pattern(
                name="formatted phone",
                regex=r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                score=0.95,
            )
        ]
        context = ["phone", "number", "mobile", "contact", "call"]
        super().__init__(supported_entity="PHONE_NUMBER", patterns=patterns, context=context)


class CustomNameRecognizer(PatternRecognizer):
    """Detects names introduced by standard name phrases — entity: PERSON."""

    def __init__(self):
        patterns = [
            Pattern(
                name="my name is",
                regex=r"(?<=my name is )[A-Za-z]+\b",
                score=0.95,
            ),
            Pattern(
                name="i am",
                regex=r"(?<=i am )[A-Z][a-z]+\b",
                score=0.95,
            )
        ]
        super().__init__(supported_entity="PERSON", patterns=patterns)


