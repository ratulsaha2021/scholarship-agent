"""Tests for chat routing and the send/edit flow (no LLM, SMTP or network needed)."""

import pytest

from src import chat_agent as ca
from src import email_sender as es
from src import digest, documents, rag_store
from src.humanizer import HumanizationResult
from src.rag_store import PostProcessor
from src.writer import GeneratedEmail

POST = """PhD Position in Machine Learning
Department of Computer Science, Example University

We are looking for a motivated PhD student to join our lab working on
graph neural networks. Applicants should have a master's degree.
Deadline: 30 November 2026
Send your CV to prof.smith@example.edu
Email subject: PhD Application - Your Name
"""


class FakeLLM:
    """Stands in for HybridLLM: canned text, but the real humanizer clean/detect runs."""

    OUTPUTS = {
        "email": "Dear Professor Smith,\n\nI'd like to apply for the PhD position.\n\nBest regards,\nTest User",
        "sop": "My research focuses on graph neural networks for medical imaging. " * 20,
        "proposal": "# Graph learning for chest X-rays\n\n## Background and motivation\n\nPrior work exists [citation needed].\n\n## Methodology\n\nWe train models.",
    }

    def __init__(self):
        from src.humanizer import Humanizer
        self.prompts = []
        self.humanizer = Humanizer()

    def generate(self, prompt, system="", use_groq=False, max_tokens=2048):
        self.prompts.append(prompt)
        return "Edited email body"

    def polish(self, text, kind="email", context="", max_tokens=2048, result=None):
        result = result if result is not None else {"local_used": False, "groq_used": True}
        text = self.humanizer.clean(text, kind)
        issues = self.humanizer.detect(text, kind)
        result.update({"humanized": text, "issues": issues, "rewrites": 0,
                       "ai_analysis": {"ai_score": 0, "patterns_found": [], "suggestions": []}})
        result.setdefault("draft", text)
        return result

    def hybrid_generate(self, prompt, context="", kind="email", max_tokens=2048):
        self.prompts.append(prompt)
        return self.polish(self.OUTPUTS[kind], kind, context,
                           result={"draft": self.OUTPUTS[kind], "local_used": False, "groq_used": True,
                                   "fallback": False, "error": ""})

    def get_status(self):
        return {"local_available": False, "groq_configured": False,
                "local_model": "", "groq_model": ""}


class FakeSender:
    def __init__(self, configured=True):
        self.configured = configured
        self.sent = []

    def is_configured(self):
        return self.configured

    def send_email(self, **kwargs):
        self.sent.append(kwargs)
        return {"success": True, "to": kwargs["to_email"]}


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "RESOURCES_DIR", tmp_path / "resources")
    monkeypatch.setattr(rag_store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(digest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(documents, "OUTPUT_DIR", tmp_path / "saved_responses")
    monkeypatch.setattr(es, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(ca, "HybridLLM", FakeLLM)
    a = ca.ChatAgent()
    a.email_sender = FakeSender()
    return a


def draft(to_email="prof@example.edu"):
    return GeneratedEmail(
        to_email=to_email, subject="Application", body="Dear Professor,\n\nHello.",
        humanization_result=HumanizationResult("", "", [], [], 1.0),
    )


def test_has_word_matches_whole_words_only():
    assert ca.has_word("hi there", ["hi"])
    assert not ca.has_word("make this shorter", ["hi"])
    assert not ca.has_word("not good", ["go"])


def test_edit_instruction_is_not_mistaken_for_greeting(agent):
    agent.current_post = {"title": "PhD", "institution": ""}
    resp = agent.chat("make this shorter")
    assert not resp.startswith("Hello")


def test_long_message_with_go_edits_instead_of_sending(agent):
    agent.pending_email = draft()
    agent.flow_state = "waiting_send"
    resp = agent.chat("make it longer and go into more detail about my thesis")
    assert agent.email_sender.sent == []
    assert "Updated email" in resp
    assert agent.flow_state == "waiting_send"


def test_edit_keyword_keeps_draft_for_next_instruction(agent):
    agent.pending_email = draft()
    agent.flow_state = "waiting_send"
    agent.chat("edit")
    assert agent.flow_state == "waiting_send"
    agent.chat("make it shorter")
    assert agent.pending_email.body == "Edited email body"
    assert agent.flow_state == "waiting_send"


def test_send_confirmation_sends(agent):
    agent.pending_email = draft()
    agent.flow_state = "waiting_send"
    resp = agent.chat("yes send it")
    assert len(agent.email_sender.sent) == 1
    assert "sent to prof@example.edu" in resp
    assert agent.pending_email is None


def test_dont_send_does_not_send(agent):
    agent.pending_email = draft()
    agent.flow_state = "waiting_send"
    agent.chat("no don't send")
    assert agent.email_sender.sent == []


def test_send_without_recipient_asks_for_one(agent):
    agent.pending_email = draft(to_email="")
    agent.flow_state = "waiting_send"
    resp = agent.chat("send")
    assert agent.email_sender.sent == []
    assert "to:" in resp
    agent.chat("to: someone@uni.edu")
    agent.chat("send")
    assert agent.email_sender.sent[0]["to_email"] == "someone@uni.edu"


def test_cv_attached_when_present(agent):
    ca.RESOURCES_DIR.mkdir(parents=True)
    (ca.RESOURCES_DIR / "cv.pdf").write_bytes(b"%PDF")
    agent.pending_email = draft()
    agent.flow_state = "waiting_send"
    agent.chat("send")
    assert agent.email_sender.sent[0]["attachments"] == [(ca.RESOURCES_DIR / "cv.pdf", "Applicant_CV.pdf")]


def test_email_credentials_are_saved(agent, monkeypatch):
    monkeypatch.setattr(es.EmailSender, "test_connection",
                        lambda self: {"success": True})
    agent.chat("setup email gmail")
    resp = agent.chat("email: me@gmail.com password: abcd efgh ijkl mnop")
    assert "configured" in resp.lower()
    saved = es.EmailConfig.load()
    assert saved.smtp_server == "smtp.gmail.com"
    assert saved.password == "abcdefghijklmnop"


def test_post_is_detected_and_parsed(agent):
    agent.chat(POST)
    assert agent.current_post["title"] == "PhD Position in Machine Learning"
    assert agent.current_post["metadata"]["professor_email"] == "prof.smith@example.edu"


def test_post_processor_extracts_subject_and_deadline():
    post = PostProcessor().process_text(POST)
    assert post.metadata["subject_format"] == "PhD Application - Your Name"
    assert post.deadline == "30 November 2026"


def test_subject_to_contract_is_not_a_subject_line():
    post = PostProcessor().process_text("PhD Studentship\nThe offer is subject to contract. Full details later.\n")
    assert post.metadata["subject_format"] == ""
    post = PostProcessor().process_text("PhD Studentship\nPlease use the subject line: GNN-PhD-2026\n")
    assert post.metadata["subject_format"] == "GNN-PhD-2026"


def load_post(agent):
    agent.resources.name = "Test User"
    agent.resources.email = "test@example.com"
    agent.chat(POST + "\nPlease include a statement of purpose and a research proposal.")


def test_post_asking_for_documents_suggests_commands(agent):
    agent.resources.name, agent.resources.email = "Test User", "t@example.com"
    resp = agent.chat(POST + "\nApplicants must submit a statement of purpose.")
    assert "write sop" in resp


def test_sop_flow_asks_for_notes_then_saves_documents(agent):
    load_post(agent)
    resp = agent.chat("write sop")
    assert agent.flow_state == "waiting_sop_notes" and "skip" in resp
    resp = agent.chat("I got into ML through a TB X-ray project. I want to work on GNNs. Goal: research career.")
    assert agent.flow_state == "reviewing_doc"
    assert (ca.RESOURCES_DIR / "sop_notes.txt").read_text().startswith("I got into ML")
    paths = agent.post_documents["sop"]
    assert paths["docx"].exists() and paths["md"].exists()
    assert "TB X-ray project" in agent.llm.prompts[-1]  # notes reach the prompt
    assert "Statement of Purpose" in resp
    # saved notes are reused next time without asking
    agent.chat("done")
    agent.chat("write sop")
    assert agent.flow_state == "reviewing_doc"


def test_doc_edit_revises_and_done_returns_to_idle(agent):
    load_post(agent)
    agent.chat("write sop")
    agent.chat("skip")
    resp = agent.chat("make the opening more concrete")
    assert "Updated statement of purpose" in resp
    assert agent.current_doc.text == "Edited email body"  # FakeLLM.generate output
    assert "Saved" in agent.chat("done")
    assert agent.flow_state == "idle"


def test_proposal_flow_marks_citations(agent):
    load_post(agent)
    agent.chat("write proposal")
    assert agent.flow_state == "waiting_proposal_topic"
    resp = agent.chat("skip")
    assert "citation needed" in resp.lower()
    assert "proposal" in agent.post_documents


def test_email_mentions_and_attaches_generated_documents(agent):
    load_post(agent)
    ca.RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    (ca.RESOURCES_DIR / "cv.pdf").write_bytes(b"%PDF")
    agent.chat("write sop"); agent.chat("skip"); agent.chat("done")
    resp = agent.chat("write")
    assert "CV, statement of purpose" in resp
    assert "CV and statement of purpose" in agent.llm.prompts[-1]
    agent.chat("send")
    names = [name for _, name in agent.email_sender.sent[0]["attachments"]]
    assert names == ["Test_User_CV.pdf", "Test_User_Statement_of_Purpose.docx"]


def test_send_from_doc_review_sends_pending_email(agent):
    load_post(agent)
    agent.chat("write")
    agent.chat("write proposal")
    agent.chat("skip")
    agent.chat("send")
    assert len(agent.email_sender.sent) == 1
