"""Tests for chat routing and the send/edit flow (no LLM, SMTP or network needed)."""

import pytest

from src import chat_agent as ca
from src import email_sender as es
from src import rag_store
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
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system="", use_groq=False):
        self.prompts.append(prompt)
        return "Edited email body"

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
    assert agent.email_sender.sent[0]["attachments"] == [ca.RESOURCES_DIR / "cv.pdf"]


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
