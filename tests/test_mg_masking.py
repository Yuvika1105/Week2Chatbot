from app.data_masking.custom_recognizers import (
    MGBrandRecognizer,
    ModelLineRecognizer,
    MaterialCodeRecognizer,
)
from app.data_masking.masking_engine import MaskingEngine
from app.data_masking.masking_policy import MaskingPolicy


def test_custom_recognizers_detect_expected_strings():
    # Smoke-test that recognizers are wired and regexes match expected MG values.
    brand = MGBrandRecognizer()
    model = ModelLineRecognizer()
    material = MaterialCodeRecognizer()

    text = "MG DIMAPUR and MG Motors ASTOR outlet 2298GFP"

    import re
    brand_hits = any(re.search(p.regex, text) for p in brand.patterns)
    model_hits = any(re.search(p.regex, text) for p in model.patterns)
    material_hits = any(re.search(p.regex, text) for p in material.patterns)

    assert brand_hits is True
    assert model_hits is True
    assert material_hits is True


def test_masking_engine_replaces_mg_entities_in_free_text():
    policy = MaskingPolicy.from_yaml("data/masking_policies/mg_policy.yaml")
    engine = MaskingEngine(policy=policy)

    res = engine.mask_text(
        "Contact MG DIMAPUR to book an ASTOR test drive. Code 2298GFP"
    )

    # Original sensitive tokens must be gone
    assert "DIMAPUR" not in res.masked_text
    assert "ASTOR" not in res.masked_text
    assert "2298GFP" not in res.masked_text

    # New behavior: MG brand removed (empty), model abbreviated, material code placeholder
    assert "<CLIENT>" not in res.masked_text          # brand is now removed, not replaced
    assert "AR" in res.masked_text                     # ASTOR -> AR
    assert "<MATERIAL_CODE>" in res.masked_text

    assert res.entity_count >= 2


def test_masking_engine_column_rule_mapping():
    # Avoid pandas dependency in unit tests.
    policy = MaskingPolicy.from_yaml("data/masking_policies/mg_policy.yaml")
    engine = MaskingEngine(policy=policy)

    # Model columns: abbreviated to first+last letter
    assert engine.mask_value("Model_Line1", "ASTOR") == "AR"
    # Material column: still uses placeholder
    assert engine.mask_value("Material", "2298GFP") == "<MATERIAL_CODE>"
    # Brand/client columns: removed (empty string)
    assert engine.mask_value("Client", "MG DELHI") == ""
    assert engine.mask_value("Name_1", "MG DIMAPUR") == ""
    
    # Verify new model forms and codes
    assert engine.mask_value("Model_Line1", "MG_COMET_EV") == "CT"
    assert engine.mask_value("Model_Line2", "WINDSOR_PRO") == "WR"
    assert engine.mask_value("Model_Line1", "MAJESTOR") == "MR"
    
    # Verify brand/branch codes in brand columns
    assert engine.mask_value("Branch", "MG01") == ""
    assert engine.mask_value("Zone", "MG02") == ""


