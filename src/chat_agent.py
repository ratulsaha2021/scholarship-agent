"""Chat agent - post-first flow with persistent storage."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .config import AgentConfig
from .resource_loader import UserResources
from .humanizer import Humanizer
from .writer import EmailWriter, GeneratedEmail, GeneratedDocument, DOC_NAMES
from .documents import save_document
from .cv_extractor import CVExtractor, ExtractedCV
from .rag_store import RAGStore, ApplicationPost, PostProcessor
from .ocr_processor import OCRProcessor
from .email_sender import EmailSender, EmailConfig
from .hybrid_llm import HybridLLM, LLMConfig, NO_FABRICATION_RULE
from .web_fetch import PoliteFetcher, FetchError, extract_main_text
from .sources import Listing, find_opportunities, profile_text
from .digest import WatchList, run_digest

RESOURCES_DIR = Path(__file__).parent.parent / "resources"
CV_EXTENSIONS = ["pdf", "docx", "txt"]
URL_RE = r"https?://[^\s<>\"')\]]+"
FIND_RE = r"^\s*(?:find|search(?:\s+for)?|look\s+for)\s+(.+?)\s*$"
SOP_WORDS = ["sop", "statement of purpose", "personal statement", "motivation letter",
             "letter of motivation", "research statement"]
PROPOSAL_WORDS = ["proposal", "research proposal", "research plan"]
DONE_WORDS = ["done", "ok", "okay", "looks good", "good", "great", "perfect", "fine", "thanks", "thank you"]
PICK_RE = r"^\s*(?:apply|open|pick|use)\s*(?:to|for)?\s*(?:#|no\.?|number)?\s*(\d+)\s*$"
EMAIL_RE = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
SMTP_PROVIDERS = {
    "gmail": ("smtp.gmail.com", 587),
    "outlook": ("smtp-mail.outlook.com", 587),
    "yahoo": ("smtp.mail.yahoo.com", 587),
}


def has_word(text, words):
    """True if any word/phrase appears in text as a whole word (not a substring)."""
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)


class ChatAgent:
    def __init__(self):
        self.config = AgentConfig.load()
        self.humanizer = Humanizer(level="high")
        self.llm = HybridLLM()
        self.cv_extractor = CVExtractor()
        self.rag_store = RAGStore()
        self.post_processor = PostProcessor()
        self.ocr = OCRProcessor()
        self.email_sender = EmailSender()
        self.resources = self._load_persistent_resources()
        self.writer = EmailWriter(self.resources, self.humanizer, self.llm)
        self.conversation_history = []
        self.current_post = None
        self.pending_email = None
        self.flow_state = "idle"
        self.context = {}
        self.fetcher = PoliteFetcher(delay=self.config.discovery.delay_between_requests)
        self.watch = WatchList()
        self.search_results = []
        self.current_doc = None
        self.post_documents = {}   # kind -> saved .docx/.md paths for the current post
        self.saved_documents = []  # everything generated this session, for downloads

    def _load_persistent_resources(self):
        return UserResources.load(RESOURCES_DIR)

    def _save_resources(self):
        user_data = {
            "name": self.resources.name,
            "email": self.resources.email,
            "phone": self.resources.phone,
            "research_interests": self.resources.research_interests,
            "skills": self.resources.skills,
            "education": self.resources.education,
            "experience": self.resources.experience,
            "publications": self.resources.publications,
            "awards": self.resources.awards,
            "target_universities": self.resources.target_universities,
        }
        RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESOURCES_DIR / "user_data.json", "w") as f:
            json.dump(user_data, f, indent=2)

    def chat(self, user_message):
        self.conversation_history.append({"role": "user", "content": user_message})

        if self.flow_state != "idle" and (self._is_web_command(user_message) or self._doc_request(user_message)):
            # A link, search, pick or document request always starts fresh, whatever we were waiting for
            self.flow_state = "idle"

        if self.flow_state == "waiting_sop_notes":
            response = self._handle_sop_notes(user_message)
        elif self.flow_state == "waiting_proposal_topic":
            response = self._handle_proposal_topic(user_message)
        elif self.flow_state == "reviewing_doc":
            response = self._handle_doc_response(user_message)
        elif self.flow_state == "waiting_cv":
            response = self._handle_cv_response(user_message)
        elif self.flow_state == "waiting_send":
            response = self._handle_send_response(user_message)
        elif self.flow_state == "waiting_clarification":
            response = self._handle_clarification_response(user_message)
        else:
            response = self._handle_new_message(user_message)

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def handle_file_upload(self, filename, file_bytes, file_type):
        if file_type in ["pdf", "docx", "txt"]:
            return self._process_cv_upload(filename, file_bytes, file_type)
        elif file_type in ["png", "jpg", "jpeg"]:
            return self._process_image_upload(filename, file_bytes)
        return f"Unsupported file type: {file_type}"

    def _is_web_command(self, message):
        return bool(
            (re.search(URL_RE, message) and len(message.split()) <= 3)
            or re.match(FIND_RE, message, re.IGNORECASE)
            or re.match(PICK_RE, message, re.IGNORECASE)
            or re.match(r"^\s*(?:un)?watch\b", message, re.IGNORECASE)
            or message.lower().strip() in ["digest", "what's new", "whats new", "watches", "watching"]
        )

    def _handle_web_command(self, message):
        ml = message.lower().strip()
        url = re.search(URL_RE, message)
        if url and len(message.split()) <= 3:
            return self._process_url(url.group(0))

        m = re.match(FIND_RE, message, re.IGNORECASE)
        if m:
            return self._handle_find(m.group(1))

        m = re.match(PICK_RE, message, re.IGNORECASE)
        if m:
            return self._handle_pick(int(m.group(1)))

        m = re.match(r"^\s*unwatch\s+(.+?)\s*$", message, re.IGNORECASE)
        if m:
            if self.watch.remove_query(m.group(1)):
                return f"Stopped watching **{m.group(1)}**."
            return f"I wasn't watching '{m.group(1)}'. Type `watches` to see the list."

        m = re.match(r"^\s*watch\s+(.+?)\s*$", message, re.IGNORECASE)
        if m:
            query = m.group(1)
            added = self.watch.add_query(query)
            return ((f"Watching **{query}**. " if added else f"Already watching **{query}**. ")
                    + "Type `digest` any time to see new matches, or schedule "
                    "`uv run python -m src.digest` to collect them daily.")

        if ml in ["watches", "watching"]:
            if not self.watch.queries:
                return "No watched searches. Type e.g. `watch machine learning phd`."
            return "**Watching:**\n" + "\n".join(f"- {q}" for q in self.watch.queries)

        return self._handle_digest()

    def _process_url(self, url, listing=None):
        try:
            title, text = extract_main_text(self.fetcher.fetch(url))
        except FetchError as e:
            return (f"**Couldn't read that page:** {e}\n\n"
                    "Instead, copy the post text and paste it here, or upload a screenshot of it "
                    "in the sidebar (LinkedIn, FindAPhD and similar sites don't allow automated access).")
        if len(text.split()) < 30:
            return ("That page has very little readable text (it may need JavaScript or a login).\n\n"
                    "Copy the post text and paste it here instead.")

        overrides = {}
        if listing:
            overrides = {"title": listing.title, "institution": listing.institution,
                         "deadline": listing.deadline}
        # Title first so it's picked up as the post title; source kept for the record
        post_text = f"{overrides.get('title') or title}\n{text}\n\nSource: {url}"
        return self._process_post(post_text, overrides=overrides, source_url=url)

    def _handle_find(self, query):
        results, errors = find_opportunities(query, profile_text(self.resources), self.fetcher)
        self.search_results = results[:10]
        if not self.search_results:
            resp = f"No listings found for **{query}**."
        else:
            resp = self._format_listings(self.search_results, f"Top matches for **{query}**")
            self.watch.mark_seen(self.search_results)
        if errors:
            resp += "\n\n_Some sources failed:_\n" + "\n".join(f"- {e}" for e in errors)
        return resp

    def _format_listings(self, listings, heading):
        lines = [f"{heading} (ranked against your profile):", ""]
        for i, l in enumerate(listings, 1):
            new = " · _new_" if self.watch.is_new(l) else ""
            details = " · ".join(p for p in [l.institution, l.deadline] if p)
            lines.append(f"{i}. **{l.title}**{new}  \n   {details} · [{l.source}]({l.url})")
        lines += ["", "Type `apply N` to load one and draft the email."]
        return "\n".join(lines)

    def _handle_pick(self, n):
        if not self.search_results:
            return "No search results yet. Try `find machine learning phd`."
        if not 1 <= n <= len(self.search_results):
            return f"Pick a number from 1 to {len(self.search_results)}."
        listing = self.search_results[n - 1]
        return self._process_url(listing.url, listing=listing)

    def _handle_digest(self):
        listings = self.watch.take_inbox()
        errors = []
        if not listings:
            if not self.watch.queries:
                return "No watched searches. Type e.g. `watch machine learning phd`."
            listings, errors = run_digest(self.watch, profile_text(self.resources), self.fetcher)
            self.watch.mark_seen(listings)
        if not listings:
            resp = "Nothing new for your watched searches since the last check."
        else:
            self.search_results = listings[:10]
            resp = self._format_listings(self.search_results, f"**{len(listings)} new** listing(s)")
        if errors:
            resp += "\n\n_Some sources failed:_\n" + "\n".join(f"- {e}" for e in errors)
        return resp

    def _handle_new_message(self, message):
        ml = message.lower()

        if self._is_web_command(message):
            return self._handle_web_command(message)

        post_kw = [
            "phd", "msc", "master", "masters", "master's", "research assistant",
            "teaching assistant", "graduate", "fellowship", "scholarship",
            "position", "positions", "opening", "openings", "apply", "deadline",
            "supervisor", "lab", "department",
        ]
        is_post = has_word(ml, post_kw) and len(message.split()) > 20

        if is_post:
            return self._process_post(message)

        doc_kind = self._doc_request(message)
        if doc_kind == "sop":
            return self._handle_sop_request(message)
        if doc_kind == "proposal":
            return self._handle_proposal_request(message)

        # Explicit commands first, so they aren't swallowed by broader keywords
        if has_word(ml, ["setup email", "set up email", "configure email", "smtp"]):
            return self._handle_setup_email(message)

        if re.search(r"\bemail\s*:", ml) and re.search(r"\bpassword\s*:", ml):
            return self._handle_email_credentials(message)

        if has_word(ml, ["setup groq", "set up groq", "api key"]):
            return self._handle_setup_groq(message)

        if has_word(ml, ["my project", "my projects", "my publication", "my publications",
                         "my skill", "my skills", "my education", "my experience"]):
            return self._show_detailed_profile(message)

        if has_word(ml, ["hello", "hi", "hey"]):
            return self._handle_greeting()

        if has_word(ml, ["help", "what can you do"]):
            return self._handle_help()

        if has_word(ml, ["cv", "resume", "profile", "my info", "status", "about me", "who am i"]):
            return self._show_status()

        if has_word(ml, ["write", "draft", "generate"]):
            return self._handle_write()

        if has_word(ml, ["send"]):
            return self._handle_send()

        # If a post is loaded, treat as modification or question
        if self.current_post:
            return self._handle_post_question(message)

        return ("Paste a scholarship/position announcement or a link to one, upload a screenshot, "
                "or search with `find machine learning phd`.\n\n"
                "I'll parse it and guide you through the application.")

    def _process_post(self, text, overrides=None, source_url=""):
        post = self.post_processor.process_text(text)
        for field_name, value in (overrides or {}).items():
            if value:
                setattr(post, field_name, value)
        if source_url:
            post.url = source_url

        self.current_post = {
            "title": post.title,
            "institution": post.institution,
            "content": post.content,
            "deadline": post.deadline,
            "requirements": post.requirements,
            "type": post.post_type,
            "metadata": post.metadata or {},
        }

        has_cv = bool(self.resources.name and self.resources.email)

        resp = f"**Found: {post.title}**\n"
        if source_url:
            resp += f"**Link:** {source_url}\n"
        if post.institution:
            resp += f"**Institution:** {post.institution}\n"
        if post.deadline:
            resp += f"**Deadline:** {post.deadline}\n"
        
        prof_email = (post.metadata or {}).get("professor_email", "")
        if prof_email:
            resp += f"**Email:** {prof_email}\n"
        
        subject_fmt = (post.metadata or {}).get("subject_format", "")
        if subject_fmt:
            resp += f"**Subject format:** {subject_fmt}\n"
        resp += "\n"

        needs = self._analyze_requirements(post.content)
        self.post_documents = {}
        self.current_doc = None

        if not has_cv:
            self.flow_state = "waiting_cv"
            resp += "I don't have your CV yet. Please upload it so I can personalize the application."
        else:
            self.flow_state = "idle"
            asks = [name for kind, name in [("sop", "statement of purpose"), ("proposal", "research proposal")]
                    if needs[kind]]
            if asks:
                cmds = " and ".join(f"**'write {'sop' if 'statement' in a else 'proposal'}'**" for a in asks)
                resp += f"This post asks for a {' and a '.join(asks)}. Type {cmds} to draft it.\n\n"
            resp += "Type **'write'** to generate the email."

        rag_post = ApplicationPost(
            id="", title=post.title, institution=post.institution,
            content=post.content, post_type=post.post_type,
            deadline=post.deadline, requirements=post.requirements,
            url=post.url, metadata=post.metadata or {},
        )
        self.rag_store.add_post(rag_post)
        return resp

    def _analyze_requirements(self, post_content):
        """Which documents the post asks for, beyond the email."""
        cl = post_content.lower()
        return {
            "sop": has_word(cl, SOP_WORDS),
            "proposal": has_word(cl, PROPOSAL_WORDS),
        }

    def _handle_cv_response(self, message):
        if has_word(message.lower(), ["upload", "here", "how"]) and len(message.split()) <= 8:
            return "Use the uploader in the sidebar to send your CV (PDF, DOCX, or TXT)."
        # Anything else (a new post, a command, a question) is handled normally
        self.flow_state = "idle"
        return self._handle_new_message(message)

    def _handle_send_response(self, message):
        ml = message.lower().strip()

        # Only a short, explicit confirmation sends. Longer messages are edit instructions,
        # so "make it longer and go into detail" never sends by accident.
        if len(ml.split()) <= 4 and has_word(ml, ["yes", "send", "send it", "sure", "go ahead"]) \
                and not has_word(ml, ["no", "not", "don't", "dont"]):
            return self._handle_send()

        to_match = re.match(r"^\s*(?:to|recipient)\s*:?\s*(" + EMAIL_RE + r")\s*$", message, re.IGNORECASE)
        if to_match and self.pending_email:
            self.pending_email.to_email = to_match.group(1)
            return (f"Recipient set to **{self.pending_email.to_email}**.\n\n"
                    "Type **'send'** to send, or tell me what to change.")

        if has_word(ml, ["no", "cancel", "discard"]) and len(ml.split()) <= 4:
            self.pending_email = None
            self.flow_state = "idle"
            return "Cancelled. Paste a new post anytime."

        if ml in ["edit", "change", "modify", "edit it", "change it"]:
            return "What would you like me to change? Tell me specifically (e.g., 'make it shorter', 'add my publication about X')."

        if has_word(ml, ["about me", "my info", "my profile", "who am i"]):
            return self._show_status()

        if has_word(ml, ["what did you write", "show email", "show the email", "what's in it", "summary"]):
            if self.pending_email:
                return (f"**To:** {self.pending_email.to_email or '(not set)'}\n"
                        f"**Subject:** {self.pending_email.subject}\n\n"
                        f"{self.pending_email.body}\n\n---\n\n"
                        "Type **'send'** to send, or tell me what to change.")
            return "No email ready."

        if has_word(ml, ["help", "what can"]):
            return ("**Options:**\n"
                    "- `send` — send the email\n"
                    "- `to: someone@uni.edu` — set the recipient\n"
                    "- `cancel` — discard\n"
                    "- Anything else is treated as an edit (e.g. 'make it shorter')")

        # Default: treat as modification request
        if self.pending_email:
            return self._handle_email_edit(message)

        self.flow_state = "idle"
        return "What would you like to do?"

    def _handle_email_edit(self, instruction):
        """Modify the pending email based on user instruction."""
        if not self.pending_email:
            return "No email to edit."
        revised = self.writer.revise_email(self.pending_email, instruction)
        if revised.error:
            return f"Couldn't edit the email: {revised.error}"
        self.pending_email = revised
        return (f"**Updated email:**\n\n{revised.body}\n\n---\n\n"
                f"_{self.humanizer.report(revised.issues)}_\n\n"
                "Type **'send'** to send, or tell me what else to change.")

    def _handle_clarification_response(self, message):
        self.context["clarification"] = message
        self.flow_state = "idle"
        if self.current_post:
            return "Got it. Type **'write'** to generate the email/application."
        return "Saved. What would you like to do next?"

    def _process_cv_upload(self, filename, file_bytes, file_type):
        try:
            # Keep the CV as resources/cv.<ext> so it can be attached when sending
            RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
            for ext in CV_EXTENSIONS:
                (RESOURCES_DIR / f"cv.{ext}").unlink(missing_ok=True)
            cv_path = RESOURCES_DIR / f"cv.{file_type}"
            cv_path.write_bytes(file_bytes)
            extracted = self.cv_extractor.extract_from_file(cv_path)
            self.resources.cv_text = extracted.raw_text

            if extracted.name:
                self.resources.name = extracted.name
            if extracted.email:
                self.resources.email = extracted.email
            if extracted.phone:
                self.resources.phone = extracted.phone
            if extracted.skills:
                self.resources.skills = extracted.skills
            if extracted.education:
                self.resources.education = extracted.education
            if extracted.experience:
                self.resources.experience = extracted.experience
            if extracted.publications:
                self.resources.publications = extracted.publications
            if extracted.awards:
                self.resources.awards = extracted.awards

            self._save_resources()
            self.writer = EmailWriter(self.resources, self.humanizer, self.llm)

            resp = "**CV uploaded and saved!**\n\n"
            resp += f"- Name: {extracted.name}\n"
            resp += f"- Email: {extracted.email}\n"
            if extracted.skills:
                resp += f"- Skills: {', '.join(extracted.skills[:8])}\n"

            if self.current_post:
                resp += "\nReady to write. Type **'write'** to generate."
                self.flow_state = "idle"
            else:
                resp += "\nNow paste a scholarship/position post to apply."
                self.flow_state = "idle"

            return resp
        except Exception as e:
            return f"Failed to extract CV: {e}\n\nPlease try again."

    def _process_image_upload(self, filename, file_bytes):
        if not self.ocr.is_available():
            return ("OCR not installed. Run:\n"
                    "```\nsudo apt-get install tesseract-ocr\npip install pytesseract Pillow\n```")
        try:
            ocr_text = self.ocr.extract_text_from_bytes(file_bytes, filename)
            if not ocr_text.strip():
                return "No text found in the image. Try a clearer image."
            return self._process_post(ocr_text)
        except Exception as e:
            return f"Failed to process image: {e}"

    def _handle_write(self):
        if not self.current_post:
            return "No post loaded. Paste a scholarship/position announcement first."

        if not self.resources.name:
            self.flow_state = "waiting_cv"
            return "I need your CV first. Please upload it."

        clarification = self.context.get("clarification", "")

        post_obj = ApplicationPost(
            id="", title=self.current_post["title"],
            institution=self.current_post["institution"],
            content=self.current_post["content"],
            post_type=self.current_post["type"],
            deadline=self.current_post["deadline"],
            requirements=self.current_post["requirements"],
            metadata=self.current_post.get("metadata", {}),
        )

        generated = self.writer.write_scholarship_application(
            post_obj, clarification or None, attachments=self._attachment_names()
        )
        if generated.error:
            return (f"**Couldn't generate the email:** {generated.error}\n\n"
                    "Check that Ollama is running with the configured model, or that your Groq key "
                    "and model in `config/llm_config.json` are valid. Then type **'write'** again.")

        self.pending_email = generated
        self.flow_state = "waiting_send"

        llm_status = self.llm.get_status()
        model_info = ""
        if llm_status["local_available"] and llm_status["groq_configured"]:
            model_info = "Local model drafted, Groq edited"
        elif llm_status["groq_configured"]:
            model_info = "Groq"
        elif llm_status["local_available"]:
            model_info = "Local model"

        resp = f"**To:** {generated.to_email or '(email not in post)'}\n"
        resp += f"**Subject:** {generated.subject}\n"
        if model_info:
            resp += f"_{model_info}_\n"
        resp += "\n---\n\n"
        resp += generated.body
        resp += "\n\n---\n\n"
        resp += f"_{self.humanizer.report(generated.issues)}_\n\n"
        resp += self._fact_report(generated)
        attached = self._attachment_names()
        if attached:
            resp += f"Sending will attach your {', '.join(attached)}.\n\n"
        resp += "Type **'send'** to send, tell me what to change, or **'cancel'**."

        return resp

    def _handle_send(self):
        if not self.pending_email:
            return "No email ready. Paste a post first, then type 'write'."
        if not self.email_sender.is_configured():
            return ("Email not configured. Type **'setup email gmail'** to set up.\n\n"
                    "Or copy the email above and send manually.")
        if not self.pending_email.to_email:
            return ("The post didn't include a recipient address.\n\n"
                    "Type `to: professor@university.edu` to set one, then **'send'**.")
        attachments = self._attachments()
        result = self.email_sender.send_email(
            to_email=self.pending_email.to_email,
            subject=self.pending_email.subject,
            body=self.pending_email.body,
            from_name=self.resources.name,
            attachments=attachments or None,
        )
        if not result["success"]:
            # Keep the draft so the user can fix the problem and retry
            return f"**Failed:** {result['error']}\n\nFix the issue and type **'send'** again, or copy the email and send manually."
        self.pending_email = None
        self.flow_state = "idle"
        self.current_post = None
        self.post_documents = {}
        names = [name for _, name in attachments]
        attached = f" (attached: {', '.join(names)})" if names else ""
        return f"**Email sent to {result['to']}!**{attached}\n\nPaste another post anytime."

    def _attachment_names(self):
        names = ["CV"] if self._cv_file() else []
        names += [DOC_NAMES[k].lower() for k in ("sop", "proposal") if k in self.post_documents]
        return names

    def _attachments(self):
        """(path, filename the recipient sees) for the CV and documents written for this post."""
        person = re.sub(r"[^A-Za-z0-9]+", "_", (self.resources.name or "Applicant").title()).strip("_")
        files = []
        cv = self._cv_file()
        if cv:
            files.append((cv, f"{person}_CV{cv.suffix}"))
        for kind in ("sop", "proposal"):
            paths = self.post_documents.get(kind)
            if paths and "docx" in paths:
                files.append((paths["docx"], f"{person}_{DOC_NAMES[kind].replace(' ', '_')}.docx"))
        return files

    # ---------- statement of purpose / research proposal ----------

    def _doc_request(self, message):
        """'write sop', 'draft a research proposal', 'sop 800 words', 'sop notes' -> kind; else None."""
        ml = message.lower().strip()
        if len(ml.split()) > 8:
            return None
        verb = r"^\s*(?:(?:write|draft|generate|create|prepare|redo)\s+(?:(?:a|an|my|new)\s+)?)?"
        if re.match(verb + r"(?:" + "|".join(re.escape(w) for w in SOP_WORDS) + r")\b", ml) \
                or re.match(r"^\s*(?:update\s+)?sop\s+notes\b", ml):
            return "sop"
        if re.match(verb + r"(?:" + "|".join(re.escape(w) for w in sorted(PROPOSAL_WORDS, key=len, reverse=True)) + r")\b", ml):
            return "proposal"
        return None

    def _post_obj(self):
        if not self.current_post:
            return None
        return ApplicationPost(
            id="", title=self.current_post["title"],
            institution=self.current_post["institution"],
            content=self.current_post["content"],
            post_type=self.current_post["type"],
            deadline=self.current_post["deadline"],
            requirements=self.current_post["requirements"],
            metadata=self.current_post.get("metadata", {}),
        )

    @staticmethod
    def _word_limit(message):
        m = re.search(r"(\d{3,4})\s*(?:-\s*\d{3,4}\s*)?words?", message.lower())
        return f"about {m.group(1)} words" if m else ""

    def _sop_notes_file(self):
        return RESOURCES_DIR / "sop_notes.txt"

    def _handle_sop_request(self, message):
        if not self.resources.name:
            self.flow_state = "waiting_cv"
            return "I need your CV first. Please upload it."
        self.context["word_limit"] = self._word_limit(message)
        notes_file = self._sop_notes_file()
        if notes_file.exists() and "notes" not in message.lower():
            return self._generate_doc("sop", notes=notes_file.read_text(encoding="utf-8"),
                                      note="Using your saved SOP notes (type `sop notes` to update them).")
        self.flow_state = "waiting_sop_notes"
        return ("A good SOP needs a few things only you know. Answer in one message (short is fine):\n\n"
                "1. What got you into your research area, concretely (a course, project, problem)?\n"
                "2. What do you want to work on in the PhD/program?\n"
                "3. Your career goal after it?\n\n"
                "I'll save these for future SOPs. Or type **'skip'** and I'll write it from your CV only "
                "(without inventing a personal story).")

    def _handle_sop_notes(self, message):
        notes = "" if message.lower().strip() in ["skip", "no", "none"] else message.strip()
        if notes:
            RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
            self._sop_notes_file().write_text(notes, encoding="utf-8")
        return self._generate_doc("sop", notes=notes)

    def _handle_proposal_request(self, message):
        if not self.resources.name:
            self.flow_state = "waiting_cv"
            return "I need your CV first. Please upload it."
        self.context["word_limit"] = self._word_limit(message)
        self.flow_state = "waiting_proposal_topic"
        base = f" for **{self.current_post['title']}**" if self.current_post else ""
        return (f"What should the proposal{base} be about? Give me your idea in a few lines "
                "(problem, rough approach, any data you'd use).\n\n"
                "Or type **'skip'** and I'll propose a topic that fits the post and your background.")

    def _handle_proposal_topic(self, message):
        topic = "" if message.lower().strip() in ["skip", "no", "none"] else message.strip()
        return self._generate_doc("proposal", notes=topic)

    def _generate_doc(self, kind, notes="", note=""):
        post = self._post_obj()
        limit = self.context.pop("word_limit", "")
        if kind == "sop":
            doc = self.writer.write_sop(post, notes=notes, word_limit=limit)
        else:
            doc = self.writer.write_proposal(post, topic=notes, word_limit=limit)
        if doc.error:
            self.flow_state = "idle"
            return (f"**Couldn't generate the {DOC_NAMES[kind].lower()}:** {doc.error}\n\n"
                    "Check that Ollama is running or that your Groq key and model are valid.")
        return self._show_doc(doc, note)

    def _show_doc(self, doc, note="", heading=None):
        paths = save_document(doc, author=self.resources.name)
        self.current_doc = doc
        if self.current_post:
            self.post_documents[doc.kind] = paths
        post = doc.post_title or "general"
        short = post if len(post) <= 32 else post[:30].rstrip(" ,:-") + "…"
        self.saved_documents.append({"label": f"{doc.title} · {short}", "paths": paths})
        self.flow_state = "reviewing_doc"

        words = len(doc.text.split())
        resp = ""
        if note:
            resp += f"_{note}_\n\n"
        resp += f"**{heading or doc.title}**" + (f" for {doc.post_title}" if doc.post_title else "") + f" ({words} words)\n\n---\n\n"
        resp += doc.text
        resp += "\n\n---\n\n"
        resp += f"_{self.humanizer.report(doc.issues)}_\n\n"
        resp += self._fact_report(doc)
        if doc.kind == "proposal" and "[citation needed]" in doc.text.lower():
            resp += "_Marked [citation needed] where a claim needs a source I don't have. Add real references before sending._\n\n"
        resp += f"Saved as `{paths.get('docx', paths['md']).name}` (download from the sidebar).\n\n"
        if self.current_post:
            resp += "It'll be attached when you send the email for this post. "
        resp += "Tell me what to change, or type **'done'**."
        return resp

    @staticmethod
    def _fact_report(item):
        out = ""
        if item.fact_fixes:
            out += f"_Fact check: corrected {len(item.fact_fixes)} claim(s) that didn't match your CV or the post._\n\n"
        if item.fact_problems:
            out += "**Check these before sending**, they may not match your CV:\n"
            out += "\n".join(f"- \"{p['claim']}\" ({p.get('issue', 'unsupported')})" for p in item.fact_problems[:5])
            out += "\n\n"
        return out

    def _handle_doc_response(self, message):
        ml = message.lower().strip()
        if not ml:
            return "Tell me what to change, or type **'done'**."
        if ml in DONE_WORDS:
            self.flow_state = "idle"
            nxt = "Type **'write'** to draft the email." if self.current_post and not self.pending_email else \
                  "Type **'send'** when ready." if self.pending_email else "Paste a post or `find` one anytime."
            return f"Saved. {nxt}"
        if ml in ["show", "show it", "show again"]:
            return self._show_doc(self.current_doc, heading=self.current_doc.title)
        commands = ["write", "email", "send", "status", "help", "cancel", "digest"]
        if ml.split()[0] in commands or re.match(r"^\s*to\s*:", ml):
            self.flow_state = "waiting_send" if self.pending_email and ml.split()[0] in ["send", "cancel"] else "idle"
            return self.chat_dispatch(message)
        if ml in ["fix", "fix it", "fix style", "fix them"]:
            message = "Fix the remaining style problems without changing any facts."
        revised = self.writer.revise_document(self.current_doc, message)
        if revised.error:
            return f"Couldn't revise: {revised.error}"
        return self._show_doc(revised, heading=f"Updated {revised.title.lower()}")

    def chat_dispatch(self, message):
        """Route a message for the current state without recording it twice in history."""
        if self.flow_state == "waiting_send":
            return self._handle_send_response(message)
        return self._handle_new_message(message)

    def _cv_file(self):
        for ext in CV_EXTENSIONS:
            path = RESOURCES_DIR / f"cv.{ext}"
            if path.exists():
                return path
        return None

    def _handle_setup_email(self, message):
        ml = message.lower()
        provider = next((p for p in SMTP_PROVIDERS if p in ml), None)
        if not provider:
            return "Which provider? Type 'setup email gmail', 'setup email outlook', or 'setup email yahoo'."

        self.context["smtp_provider"] = provider
        server, _ = SMTP_PROVIDERS[provider]
        return (f"**Configure {server}**\n\n"
                "Type: `email: your@email.com password: your-app-password`\n\n"
                "_Gmail: use an App Password from https://myaccount.google.com/apppasswords_\n\n"
                "_Credentials are stored locally in `config/email_config.json` (gitignored)._")

    def _handle_email_credentials(self, message):
        email_match = re.search(r"email\s*:\s*(" + EMAIL_RE + r")", message, re.IGNORECASE)
        pw_match = re.search(r"password\s*:\s*(.+?)\s*$", message, re.IGNORECASE)
        if not email_match or not pw_match:
            return "Format: `email: your@email.com password: your-app-password`"

        address = email_match.group(1)
        provider = self.context.get("smtp_provider") or next(
            (p for p in SMTP_PROVIDERS if p in address.lower()), None)
        if not provider:
            return "Which provider? Type 'setup email gmail', 'outlook', or 'yahoo' first."

        server, port = SMTP_PROVIDERS[provider]
        config = EmailConfig(
            smtp_server=server, smtp_port=port, email_address=address,
            password=pw_match.group(1).replace(" ", ""), use_tls=True,
        )
        sender = EmailSender(config)
        test = sender.test_connection()
        if not test["success"]:
            return f"**Couldn't log in:** {test['error']}\n\nCheck the address and app password and try again."

        config.save()
        self.email_sender = sender
        return f"**Email configured** ({address} via {server}). You can now type **'send'** on a drafted email."

    def _handle_setup_groq(self, message):
        match = re.search(r"setup\s+groq\s+(\S+)", message)
        if match:
            self.llm.config.groq_api_key = match.group(1)
            self.llm.config.save()
            return "**Groq configured!** Now both Local Llama and Groq will work together."
        return "Type: `setup groq gsk_your_api_key`"

    def _handle_greeting(self):
        name = self.resources.name or ""
        has_cv = bool(name)
        llm = self.llm.get_status()

        resp = f"Hello{' ' + name if name else ''}!\n\n"
        resp += "Paste a scholarship/position post (text or image) to get started.\n\n"

        if has_cv:
            resp += f"_CV loaded: {name}_\n"
        else:
            resp += "_No CV loaded yet — I'll ask when needed._\n"

        models = []
        if llm["local_available"]:
            models.append("Local Llama")
        if llm["groq_configured"]:
            models.append("Groq")
        if models:
            resp += f"_Models: {', '.join(models)}_"
        if self.watch.inbox:
            resp += f"\n\n**{len(self.watch.inbox)} new listing(s)** from your watched searches. Type `digest` to see them."

        return resp

    def _handle_help(self):
        return ("**How it works:**\n\n"
                "1. Paste a post (text, screenshot or link), or search with `find ...`\n"
                "2. Upload your CV if needed\n"
                "3. I write the email\n"
                "4. You send it\n\n"
                "**Commands:**\n"
                "- `status` — show your profile\n"
                "- `setup email gmail` — configure sending\n"
                "- `setup groq KEY` — add Groq API\n"
                "- `find machine learning phd` — search EURAXESS + jobs.ac.uk\n"
                "- `apply 3` — load result 3 from the last search\n"
                "- paste a link — read a post straight from its web page\n"
                "- `watch QUERY` / `unwatch QUERY` / `watches` — saved searches\n"
                "- `digest` — new listings for your saved searches\n"
                "- `write` — generate email for loaded post\n"
                "- `write sop` — statement of purpose (add e.g. `800 words`)\n"
                "- `write proposal` — research proposal\n"
                "- `send` — send the drafted email")

    def _show_status(self):
        resp = "**Your Profile:**\n\n"
        resp += f"- Name: {self.resources.name or 'Not set'}\n"
        resp += f"- Email: {self.resources.email or 'Not set'}\n"
        resp += f"- Phone: {self.resources.phone or 'Not set'}\n"
        if self.resources.skills:
            resp += f"- Skills: {', '.join(self.resources.skills[:8])}\n"
        if self.resources.education:
            for e in self.resources.education[:2]:
                resp += f"- {e.get('degree', '')} from {e.get('institution', '')}\n"

        resp += "\n**Models:**\n"
        llm = self.llm.get_status()
        resp += f"- Local Llama: {'Running' if llm['local_available'] else 'Off'}\n"
        resp += f"- Groq: {'Configured' if llm['groq_configured'] else 'Not configured'}\n"

        resp += "\n**Email Sending:** "
        resp += "Configured" if self.email_sender.is_configured() else "Not configured"

        return resp

    def _show_detailed_profile(self, message):
        ml = message.lower()
        resp = ""

        if "project" in ml:
            resp = "**Your Projects:**\n\n"
            if self.resources.experience:
                for exp in self.resources.experience:
                    resp += f"- **{exp.get('title', 'Project')}** at {exp.get('organization', '')}\n"
                    if exp.get("description"):
                        resp += f"  {exp['description'][:150]}\n"
            else:
                resp = "No projects found. Upload your CV or add them manually."

        elif "publication" in ml:
            if self.resources.publications:
                resp = "**Your Publications:**\n\n"
                for i, pub in enumerate(self.resources.publications, 1):
                    resp += f"{i}. {pub}\n"
            else:
                resp = "No publications found. Upload your CV or add them."

        elif "skill" in ml:
            if self.resources.skills:
                resp = "**Your Skills:**\n\n" + ", ".join(self.resources.skills)
            else:
                resp = "No skills found. Upload your CV or add them."

        elif "education" in ml:
            if self.resources.education:
                resp = "**Your Education:**\n\n"
                for e in self.resources.education:
                    resp += f"- {e.get('degree', '')} in {e.get('field', '')} from {e.get('institution', '')} ({e.get('year', '')})\n"
            else:
                resp = "No education found. Upload your CV or add it."

        elif "experience" in ml:
            if self.resources.experience:
                resp = "**Your Experience:**\n\n"
                for exp in self.resources.experience:
                    resp += f"- {exp.get('title', '')} at {exp.get('organization', '')} ({exp.get('duration', '')})\n"
                    if exp.get("description"):
                        resp += f"  {exp['description'][:150]}\n"
            else:
                resp = "No experience found. Upload your CV or add it."

        else:
            resp = self._show_status()

        return resp

    def _handle_post_question(self, message):
        """Handle questions when a post is loaded."""
        ml = message.lower()

        # If asking about own projects/profile, show from profile
        if any(w in ml for w in ["my", "mine", "i have", "i did", "i worked", "contribution", "project"]):
            profile = self.resources.to_context_string()
            prompt = f"""The user asks: {message}

Their profile:
{profile}

Give a brief answer based on their actual profile. Be specific about what they have."""

            try:
                return self.llm.generate(prompt, use_groq=True)
            except Exception:
                return self._show_detailed_profile(message)

        if not self.current_post:
            return "No post loaded. Paste one first."

        prompt = f"""The user has this loaded post:
Title: {self.current_post['title']}
Institution: {self.current_post['institution']}

User asks: {message}

Give a brief, helpful answer."""

        try:
            return self.llm.generate(prompt, use_groq=True)
        except Exception:
            return "Try 'write' to generate the email, or ask something specific."
