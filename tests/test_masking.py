from app.chatbot.chatbot import _mask_confidential_sections


def test_masking_redacts_for_guest():
    txt = "Public info\n[[CONFIDENTIAL:roles=admin,it_user]]SecretP@ss[[/CONFIDENTIAL]]\nEnd"
    out = _mask_confidential_sections(txt, user_role="guest")
    assert "[REDACTED]" in out
    assert "SecretP@ss" not in out


def test_masking_allows_for_admin():
    txt = "Start\n[[CONFIDENTIAL:roles=admin]]TopSecret[[/CONFIDENTIAL]]\nDone"
    out = _mask_confidential_sections(txt, user_role="admin")
    assert "TopSecret" in out
    assert "[REDACTED]" not in out
