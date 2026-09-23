"""Tests for the rule-based humanizer."""

from src.humanizer import Humanizer

H = Humanizer()

AI_EMAIL = """Dear Professor Smith,

I hope this email finds you well. I am writing to express my interest in your esteemed lab — a pivotal
place for research. This is not just a position, but a unique opportunity to delve into the evolving
landscape of AI, showcasing my skills. Let that sink in.

Best regards,
[Your Name]"""

CLEAN_EMAIL = """Dear Professor Smith,

I'd like to apply for the PhD position in graph learning advertised on your group's page. I'm finishing
an M.Sc. in Computer Science, and my thesis used convolutional networks to detect tuberculosis in chest
X-rays, reaching 97% accuracy on a public dataset.

I've attached my CV. Could we arrange a short call?

Best regards,
Test User"""


def ids(text, kind="email"):
    return {i.id for i in H.detect(text, kind)}


def test_detects_the_main_tells():
    found = ids(AI_EMAIL)
    for expected in ["email_cliche", "not_x_but_y", "closer", "inflation", "sales",
                     "ai_words", "placeholder", "dashes", "ing_riders"]:
        assert expected in found, expected
    assert H.needs_rewrite(H.detect(AI_EMAIL))


def test_clean_professional_email_passes():
    issues = H.detect(CLEAN_EMAIL)
    assert issues == [], [i.describe() for i in issues]
    assert not H.needs_rewrite(issues)
    assert H.score(issues) == 1.0


def test_clean_fixes_mechanical_tells_without_touching_facts():
    text = "My work — on TB — hit 97% in 2024–2025. “Good” **results** \U0001F680\n\nI hope this helps!"
    cleaned = H.clean(text)
    assert "—" not in cleaned and "“" not in cleaned and "**" not in cleaned
    assert "2024-2025" in cleaned and "97%" in cleaned
    assert "I hope this helps" not in cleaned
    assert "\U0001F680" not in cleaned


def test_clean_removes_stock_opener_sentence():
    cleaned = H.clean("Dear Dr. Rahman,\n\nI hope this email finds you well. I'd like to apply.")
    assert cleaned == "Dear Dr. Rahman,\n\nI'd like to apply."


def test_citation_needed_allowed_only_in_proposals():
    text = "Graph models help diagnosis [citation needed]."
    assert "placeholder" in ids(text, "email")
    assert "placeholder" not in ids(text, "proposal")
    assert "placeholder" not in ids("As shown in [3], it works.", "proposal")


def test_rewrite_prompt_lists_found_issues_and_keeps_facts_rule():
    prompt = H.rewrite_prompt(AI_EMAIL, H.detect(AI_EMAIL), "email")
    assert "Stock email opener" in prompt
    assert "Don't add any new ones" in prompt


def test_repeated_openings_flagged():
    text = "I built a model. I trained it on X-rays. I published the results."
    assert "openings" in ids(text)


def test_unicode_hyphens_are_normalised_and_detected():
    text = "It reached state‑of‑the‑art accuracy with deep‑learning."
    assert "sales" in ids(text)
    assert H.clean(text) == "It reached state-of-the-art accuracy with deep-learning."
