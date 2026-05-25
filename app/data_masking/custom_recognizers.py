import re
from presidio_analyzer import PatternRecognizer, Pattern

class MGBrandRecognizer(PatternRecognizer):
    """Detects 'MG', 'MG DIMAPUR', 'MG Motors' — entity: MG_BRAND"""
    def __init__(self):
        patterns = [
            Pattern(name="MG Motors company", regex=r"\bMG\s+Motors?\b", score=0.95),
            Pattern(name="MG dealer prefix", regex=r"\bMG\s+[A-Z][A-Z\s,\-]+", score=0.90),
            Pattern(name="MG standalone", regex=r"\bMG\b", score=0.75),
        ]
        context = ["dealer", "branch", "zone", "outlet", "showroom", "client", "motors"]
        super().__init__(supported_entity="MG_BRAND", patterns=patterns, context=context)

class ModelLineRecognizer(PatternRecognizer):
    """Detects ASTOR, HECTOR, ZS EV, COMET EV, etc. — entity: MG_MODEL"""
    MODEL_KEYWORDS = ["ASTOR", "HECTOR", "HECTOR PLUS", "GLOSTER", "ZS EV", "COMET EV", "WINDSOR EV", "CLOUD EV"]

    def __init__(self):
        regex_str = r"\b(" + "|".join([re.escape(m) for m in self.MODEL_KEYWORDS]) + r")\b"
        patterns = [Pattern(name="MG Model Line Match", regex=regex_str, score=0.95)]
        super().__init__(supported_entity="MG_MODEL", patterns=patterns)

class MaterialCodeRecognizer(PatternRecognizer):
    """Detects SAP/ERP codes like 2298GFP — entity: MG_MATERIAL"""
    def __init__(self):
        patterns = [Pattern(name="SAP Material Code", regex=r"\b\d{4}[A-Z]{3}\b", score=0.95)]
        super().__init__(supported_entity="MG_MATERIAL", patterns=patterns)