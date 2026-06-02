import re
from presidio_analyzer import PatternRecognizer, Pattern


class MGBrandRecognizer(PatternRecognizer):
    """Detects 'MG', 'MG DIMAPUR', 'MG Motors' — entity: MG_BRAND."""

    def __init__(self):
        patterns = [
            Pattern(
                name="MG Motors",
                regex=r"\bMG\s+Motors?\b",
                score=0.95,
            ),
            Pattern(
                name="MG brand + outlet",
                regex=r"\bMG\s+[A-Z0-9\-]+\b",
                score=0.92,
            ),
            Pattern(
                name="MG standalone",
                regex=r"\bMG\b",
                score=0.72,
            ),
        ]
        context = ["dealer", "branch", "zone", "outlet", "showroom", "client", "motors"]
        super().__init__(supported_entity="MG_BRAND", patterns=patterns, context=context)


class ModelLineRecognizer(PatternRecognizer):
    """Detects ASTOR, HECTOR, ZS EV, COMET EV, etc. — entity: MG_MODEL"""
    MODEL_KEYWORDS = ["ASTOR", "HECTOR PLUS", "HECTOR", "GLOSTER", "ZS EV", "COMET EV", "WINDSOR EV", "CLOUD EV", "MAJESTOR", "CYBERSTER", "M9"]

    # Pre-built abbreviation map: first letter + last letter of base model word
    # Multi-word models (e.g. "ZS EV") use first+last of first word only
    ABBREV_MAP = {
        model: model.split()[0][0].upper() + model.split()[0][-1].upper()
        for model in MODEL_KEYWORDS
    }

    # Comprehensive map of base models to abbreviation and their matching regex patterns
    MODEL_PATTERNS = {
        "HECTOR": (r"\b(?:MG[_\s]+)?HECTOR(?:[_\s]+PLUS)?(?:\s+\d+S)?\b", "HR"),
        "ASTOR": (r"\b(?:MG[_\s]+)?ASTOR\w*\b", "AR"),
        "GLOSTER": (r"\b(?:MG[_\s]+)?GLOSTER\w*\b", "GR"),
        "MAJESTOR": (r"\b(?:MG[_\s]+)?MAJESTOR\w*\b", "MR"),
        "COMET": (r"\b(?:MG[_\s]+)?COMET(?:[_\s]+EV\w*)?\b", "CT"),
        "CYBERSTER": (r"\b(?:MG[_\s]+)?CYBERSTER\w*\b", "CR"),
        "M9": (r"\b(?:MG[_\s]+)?M9\b", "M9"),
        "ZS": (r"\b(?:MG[_\s]+)?ZS(?:[_\s]+EV\w*)?\b", "ZS"),
        "WINDSOR": (r"\b(?:MG[_\s]+)?WINDSOR(?:[_\s]+(?:EV|PRO)\w*)?\b", "WR"),
        "CLOUD": (r"\b(?:MG[_\s]+)?CLOUD(?:[_\s]+EV\w*)?\b", "CD")
    }

    def __init__(self):
        patterns = []
        for model_name, (pattern, abbrev) in self.MODEL_PATTERNS.items():
            patterns.append(Pattern(name=f"MG Model {model_name}", regex=pattern, score=0.95))
        super().__init__(supported_entity="MG_MODEL", patterns=patterns)

class MaterialCodeRecognizer(PatternRecognizer):
    """Detects SAP/ERP codes like 2298GFP — entity: MG_MATERIAL"""
    def __init__(self):
        patterns = [Pattern(name="SAP Material Code", regex=r"\b\d{4}[A-Z]{3}\b", score=0.95)]
        super().__init__(supported_entity="MG_MATERIAL", patterns=patterns)


def abbreviate_models(text: str) -> str:
    """Replace each MG model name with its first+last letter abbreviation."""
    # Sort by key length to process larger/longer model patterns first
    for model_name, (pattern, abbrev) in sorted(ModelLineRecognizer.MODEL_PATTERNS.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(pattern, abbrev, text, flags=re.IGNORECASE)
    return text


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

