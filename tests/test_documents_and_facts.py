"""Fact-check parsing, Word export and CV phone extraction."""

from docx import Document

from src.cv_extractor import CVExtractor
from src.documents import save_document
from src.hybrid_llm import HybridLLM, LLMConfig
from src.writer import GeneratedDocument


def test_fact_check_parses_model_json(monkeypatch):
    llm = HybridLLM(LLMConfig(groq_api_key="x"))
    monkeypatch.setattr(llm, "_call_groq", lambda *a, **k: 'Sure: {"problems": [{"claim": "master\'s thesis", "issue": "it was the undergraduate thesis"}]}')
    problems = llm.fact_check("My master's thesis...", "B.Sc. thesis: chest disease detection")
    assert problems == [{"claim": "master's thesis", "issue": "it was the undergraduate thesis"}]


def test_fact_check_tolerates_bad_output(monkeypatch):
    llm = HybridLLM(LLMConfig(groq_api_key="x"))
    monkeypatch.setattr(llm, "_call_groq", lambda *a, **k: "no json here")
    assert llm.fact_check("text", "sources") == []


def test_hybrid_generate_fixes_flagged_facts(monkeypatch):
    llm = HybridLLM(LLMConfig(groq_api_key="x"))
    monkeypatch.setattr(llm, "_check_local_available", lambda: False)
    calls = []

    def fake_groq(prompt, system="", max_tokens=2048, reasoning="low"):
        calls.append(prompt)
        if prompt.startswith("You are fact-checking"):
            if "master's thesis" in prompt.split("DOCUMENT:")[1]:
                return '{"problems": [{"claim": "master\'s thesis", "issue": "undergraduate thesis"}]}'
            return '{"problems": []}'
        if prompt.startswith("Correct these factual problems"):
            return "My undergraduate thesis used deep learning to detect chest disease in X-ray images."
        return "My master's thesis used deep learning to detect chest disease in X-ray images."

    monkeypatch.setattr(llm, "_call_groq", fake_groq)
    result = llm.hybrid_generate("Write an SOP", "B.Sc. thesis: chest disease", kind="sop")
    assert "undergraduate thesis" in result["humanized"]
    assert result["fact_fixes"] and result["fact_problems"] == []


def test_docx_export_builds_headings_lists_and_tables(tmp_path):
    text = ("# Graph learning for X-rays\n\n## Work plan\n\n- Year 1: baselines\n\n"
            "| Year | Milestone |\n|------|-----------|\n| 1 | Baselines |\n| 2 | Graph models |\n")
    doc = GeneratedDocument("proposal", "Research Proposal", text, post_title="PhD in ML")
    paths = save_document(doc, author="Test User", output_dir=tmp_path)
    word = Document(paths["docx"])
    assert [p.text for p in word.paragraphs if p.style.name.startswith("Heading")] == \
        ["Graph learning for X-rays", "Work plan"]
    assert len(word.tables) == 1 and word.tables[0].cell(2, 1).text == "Graph models"
    assert paths["md"].read_text().startswith("# Graph learning")


def test_phone_ignores_address_numbers():
    cv = "RATUL SAHA\n240/12 President Road, Dhaka 1400\n+880 1910 076810 | me@example.com\n2020 - 2024"
    assert CVExtractor()._extract_phone(cv) == "+880 1910 076810"


def test_fact_check_drops_claims_not_in_document(monkeypatch):
    llm = HybridLLM(LLMConfig(groq_api_key="x"))
    monkeypatch.setattr(llm, "_call_groq", lambda *a, **k: '{"problems": [{"claim": "CGPA: 4.00 / 4.00", "issue": "x"}, {"claim": "first author of LW-ALDDNet", "issue": "Prito is first author"}]}')
    problems = llm.fact_check("I was first author of LW-ALDDNet.", "sources")
    assert [p["claim"] for p in problems] == ["first author of LW-ALDDNet"]


def test_context_includes_whole_cv():
    from src.resource_loader import UserResources
    r = UserResources(name="A", cv_text="x" * 5000 + " MICCAI benchmark")
    assert "MICCAI benchmark" in r.to_context_string()
