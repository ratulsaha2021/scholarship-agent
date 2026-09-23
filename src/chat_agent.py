"""Chat agent - post-first flow with persistent storage."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .config import AgentConfig
from .resource_loader import UserResources
from .humanizer import Humanizer
from .writer import EmailWriter, GeneratedEmail
from .cv_extractor import CVExtractor, ExtractedCV
from .rag_store import RAGStore, ApplicationPost, PostProcessor
from .ocr_processor import OCRProcessor
from .email_sender import EmailSender, EmailConfig
from .hybrid_llm import HybridLLM, LLMConfig, NO_FABRICATION_RULE

RESOURCES_DIR = Path(__file__).parent.parent / "resources"
CV_EXTENSIONS = ["pdf", "docx", "txt"]
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

        if self.flow_state == "waiting_cv":
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

    def _handle_new_message(self, message):
        ml = message.lower()

        post_kw = [
            "phd", "msc", "master", "masters", "master's", "research assistant",
            "teaching assistant", "graduate", "fellowship", "scholarship",
            "position", "positions", "opening", "openings", "apply", "deadline",
            "supervisor", "lab", "department",
        ]
        is_post = has_word(ml, post_kw) and len(message.split()) > 20

        if is_post:
            return self._process_post(message)

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

        return ("Paste a scholarship/position announcement, or upload a screenshot.\n\n"
                "I'll parse it and guide you through the application.")

    def _process_post(self, text):
        post = self.post_processor.process_text(text)

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

        if not has_cv:
            self.flow_state = "waiting_cv"
            resp += "I don't have your CV yet. Please upload it so I can personalize the application."
        elif needs.get("needs_clarification"):
            self.flow_state = "waiting_clarification"
            resp += needs["clarification_question"]
        else:
            resp += "I have your profile. Type **'write'** to generate the email/application."
            self.flow_state = "idle"

        rag_post = ApplicationPost(
            id="", title=post.title, institution=post.institution,
            content=post.content, post_type=post.post_type,
            deadline=post.deadline, requirements=post.requirements,
            metadata=post.metadata or {},
        )
        self.rag_store.add_post(rag_post)
        return resp

    def _analyze_requirements(self, post_content):
        needs = {
            "needs_research_statement": False,
            "needs_proposal": False,
            "needs_clarification": False,
            "clarification_question": "",
        }
        cl = post_content.lower()

        if "research statement" in cl or "research plan" in cl:
            needs["needs_research_statement"] = True
            needs["needs_clarification"] = True
            needs["clarification_question"] = (
                "This position requires a research statement.\n\n"
                "Please describe your research interests and plans, "
                "or I can draft one based on your CV."
            )
        elif "proposal" in cl:
            needs["needs_proposal"] = True
            needs["needs_clarification"] = True
            needs["clarification_question"] = (
                "This position requires a research proposal.\n\n"
                "Do you have one ready, or should I draft one?"
            )
        return needs

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

        prompt = f"""Edit this email based on the instruction:

Current email:
{self.pending_email.body}

Instruction: {instruction}

Applicant background (the only source of facts):
{self.resources.to_context_string()}

Rules:
- Apply the instruction and keep everything else as it is.
- {NO_FABRICATION_RULE}
- Don't open with "I am writing to express".
- Return ONLY the email body (greeting to sign-off). No subject line, no explanations."""

        try:
            edited = self.llm.generate(prompt, use_groq=True)
            if not edited.strip() or edited == "No LLM available.":
                return "No LLM available to edit the email. Start Ollama or add a Groq key (`setup groq KEY`)."
            self.pending_email.body = edited
            return (f"**Updated email:**\n\n{edited}\n\n---\n\n"
                    "Type **'send'** to send, **'edit'** to change more.")
        except Exception:
            return "Couldn't edit. Please rephrase your request."

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
            post_obj, clarification or None, cv_attached=self._cv_file() is not None
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
            model_info = "*Local Llama drafted + Groq humanized*"
        elif llm_status["groq_configured"]:
            model_info = "*Groq API*"
        elif llm_status["local_available"]:
            model_info = "*Local Llama*"

        resp = f"**To:** {generated.to_email or '(email not in post)'}\n"
        resp += f"**Subject:** {generated.subject}\n"
        if model_info:
            resp += f"_{model_info}_\n"
        resp += "\n---\n\n"
        resp += generated.body
        resp += "\n\n---\n\n"
        resp += "Type **'send'** to send, **'edit'** to modify, or **'cancel'**."

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
        cv_file = self._cv_file()
        result = self.email_sender.send_email(
            to_email=self.pending_email.to_email,
            subject=self.pending_email.subject,
            body=self.pending_email.body,
            from_name=self.resources.name,
            attachments=[cv_file] if cv_file else None,
        )
        if not result["success"]:
            # Keep the draft so the user can fix the problem and retry
            return f"**Failed:** {result['error']}\n\nFix the issue and type **'send'** again, or copy the email and send manually."
        self.pending_email = None
        self.flow_state = "idle"
        self.current_post = None
        attached = " (CV attached)" if cv_file else ""
        return f"**Email sent to {result['to']}!**{attached}\n\nPaste another post anytime."

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

        return resp

    def _handle_help(self):
        return ("**How it works:**\n\n"
                "1. Paste a post (text or screenshot)\n"
                "2. Upload your CV if needed\n"
                "3. I write the email\n"
                "4. You send it\n\n"
                "**Commands:**\n"
                "- `status` — show your profile\n"
                "- `setup email gmail` — configure sending\n"
                "- `setup groq KEY` — add Groq API\n"
                "- `write` — generate email for loaded post\n"
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
