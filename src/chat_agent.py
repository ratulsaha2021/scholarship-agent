"""Chat agent - main conversational interface with hybrid LLM."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import AgentConfig
from .resource_loader import UserResources
from .humanizer import Humanizer
from .discovery import OpportunityDiscovery, Opportunity
from .writer import EmailWriter, GeneratedEmail
from .cv_extractor import CVExtractor, ExtractedCV
from .rag_store import RAGStore, ApplicationPost, PostProcessor
from .ocr_processor import OCRProcessor
from .email_sender import EmailSender, EmailConfig
from .intent_detector import IntentDetector, Intent, DetectedIntent
from .hybrid_llm import HybridLLM, LLMConfig

RESOURCES_DIR = Path(__file__).parent.parent / "resources"

class ChatAgent:
    """Conversational agent with hybrid LLM routing."""
    
    def __init__(self):
        self.config = AgentConfig.load()
        self.resources = UserResources.load(RESOURCES_DIR)
        self.humanizer = Humanizer(level="high")
        self.llm = HybridLLM()
        self.writer = EmailWriter(self.resources, self.humanizer, self.llm)
        self.cv_extractor = CVExtractor()
        self.rag_store = RAGStore()
        self.post_processor = PostProcessor()
        self.ocr = OCRProcessor()
        self.email_sender = EmailSender()
        self.intent_detector = IntentDetector()
        
        self.conversation_history: List[Dict] = []
        self.pending_email: Optional[GeneratedEmail] = None
        self.pending_post_text: Optional[str] = None
        self.context: Dict = {}
    
    def chat(self, user_message: str) -> str:
        """Process user message and return response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        intent = self.intent_detector.detect(user_message)
        
        response = self._handle_intent(intent, user_message)
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def handle_file_upload(self, filename: str, file_bytes: bytes, file_type: str) -> str:
        """Handle file uploads."""
        if file_type in ["pdf", "docx", "txt"]:
            return self._handle_cv_upload(filename, file_bytes, file_type)
        elif file_type in ["png", "jpg", "jpeg"]:
            return self._handle_image_upload(filename, file_bytes)
        else:
            return f"Unsupported file type: {file_type}"
    
    def _handle_intent(self, intent: DetectedIntent, text: str) -> str:
        """Route to appropriate handler."""
        handlers = {
            Intent.CV_UPLOAD: lambda: self._handle_cv_request(text),
            Intent.CV_EXTRACT: lambda: self._handle_cv_extract(),
            Intent.ADD_POST: lambda: self._handle_add_post(text),
            Intent.ADD_POST_IMAGE: lambda: "Please upload an image using the file uploader below.",
            Intent.SEARCH_POSTS: lambda: self._handle_search_posts(text),
            Intent.WRITE_EMAIL: lambda: self._handle_write_email(intent.entities, text),
            Intent.APPLY_SCHOLARSHIP: lambda: self._handle_apply("scholarship"),
            Intent.APPLY_PHD: lambda: self._handle_apply("phd_position"),
            Intent.SEND_EMAIL: lambda: self._handle_send_email(),
            Intent.SETUP_SMTP: lambda: self._handle_setup_smtp(text),
            Intent.SHOW_SAVED: lambda: self._handle_show_saved(),
            Intent.HELP: lambda: self._handle_help(),
            Intent.PROFILE_UPDATE: lambda: self._handle_profile_update(text),
            Intent.GENERAL_CHAT: lambda: self._handle_general(text),
        }
        
        handler = handlers.get(intent.intent, handlers[Intent.GENERAL_CHAT])
        return handler()
    
    def _handle_cv_request(self, text: str) -> str:
        """Handle CV upload request."""
        return ("Please upload your CV using the file uploader below.\n\n"
                "Supported formats: PDF, DOCX, TXT\n\n"
                "I'll automatically extract all your information.")
    
    def _handle_cv_upload(self, filename: str, file_bytes: bytes, file_type: str) -> str:
        """Process uploaded CV file."""
        try:
            temp_path = RESOURCES_DIR / f"temp_cv.{file_type}"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(file_bytes)
            
            extracted = self.cv_extractor.extract_from_file(temp_path)
            temp_path.unlink(missing_ok=True)
            
            self.context['extracted_cv'] = extracted
            
            response = f"**CV Extracted Successfully!**\n\n"
            response += f"**Name:** {extracted.name}\n"
            response += f"**Email:** {extracted.email}\n"
            response += f"**Phone:** {extracted.phone}\n\n"
            
            if extracted.education:
                response += "**Education:**\n"
                for edu in extracted.education[:3]:
                    response += f"- {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')}\n"
            
            if extracted.skills:
                response += f"\n**Skills:** {', '.join(extracted.skills[:10])}\n"
            
            if extracted.experience:
                response += "\n**Experience:**\n"
                for exp in extracted.experience[:3]:
                    response += f"- {exp.get('title', 'N/A')} at {exp.get('organization', 'N/A')}\n"
            
            response += "\nWould you like me to save this to your profile? (Type 'yes' to save)"
            
            return response
        
        except Exception as e:
            return f"Failed to extract CV: {str(e)}\n\nPlease try uploading again."
    
    def _handle_cv_extract(self) -> str:
        """Show extracted CV data."""
        if 'extracted_cv' in self.context:
            cv = self.context['extracted_cv']
            return cv.to_context_string()
        elif self.resources.name:
            return self.resources.to_context_string()
        else:
            return "No CV data found. Please upload your CV first."
    
    def _handle_add_post(self, text: str) -> str:
        """Handle adding a new post."""
        self.pending_post_text = text
        return ("Please paste the full scholarship/position announcement text, "
                "or upload an image/screenshot of the post.\n\n"
                "You can also just paste the text directly in the chat.")
    
    def _handle_image_upload(self, filename: str, file_bytes: bytes) -> str:
        """Process uploaded image with OCR."""
        if not self.ocr.is_available():
            return ("OCR is not installed. Please install it:\n"
                    "```bash\n"
                    "sudo apt-get install tesseract-ocr\n"
                    "pip install pytesseract Pillow\n"
                    "```")
        
        try:
            ocr_text = self.ocr.extract_text_from_bytes(file_bytes, filename)
            
            if not ocr_text.strip():
                return "No text found in the image. Please try a clearer image."
            
            post = self.post_processor.process_image_text(ocr_text)
            post_id = self.rag_store.add_post(post)
            
            response = f"**Image processed and post saved!**\n\n"
            response += f"**Extracted Text:**\n{ocr_text[:500]}\n\n"
            response += f"**Post ID:** {post_id}\n\n"
            response += "Would you like me to apply to this? (Type 'apply' to proceed)"
            
            self.context['last_post_id'] = post_id
            
            return response
        
        except Exception as e:
            return f"Failed to process image: {str(e)}"
    
    def _handle_search_posts(self, text: str) -> str:
        """Search saved posts."""
        query = re.sub(r'(?:search|find|look|for|saved|posts?|scholarships?|positions?)', '', text, flags=re.IGNORECASE).strip()
        
        if not query:
            query = "scholarship PhD position"
        
        results = self.rag_store.search(query, n_results=5)
        
        if not results:
            return "No saved posts found. Add some posts first using 'add post' command."
        
        response = f"**Found {len(results)} relevant posts:**\n\n"
        
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            score = result.get("score", 0)
            response += f"**{i}. {meta.get('title', 'Untitled')}**\n"
            response += f"   Institution: {meta.get('institution', 'N/A')}\n"
            response += f"   Type: {meta.get('post_type', 'N/A')}\n"
            if meta.get('deadline'):
                response += f"   Deadline: {meta['deadline']}\n"
            response += f"   Relevance: {score:.2f}\n\n"
        
        response += "Type 'apply 1' or 'apply 2' etc. to apply to a specific post."
        
        return response
    
    def _handle_write_email(self, entities: Dict, text: str) -> str:
        """Handle email writing request using hybrid LLM."""
        prof_name = entities.get('professor_name', '')
        prof_email = entities.get('email', '')
        
        if not prof_name:
            return ("Who would you like to email?\n\n"
                    "Please provide:\n"
                    "- Professor's name (e.g., 'Dr. Smith' or 'Professor John Smith')\n"
                    "- Their email address\n"
                    "- What you want to discuss\n\n"
                    "Example: 'Write email to Dr. Smith at smith@mit.edu about ML research'")
        
        if not prof_email:
            return f"What is Professor {prof_name}'s email address?"
        
        research_topic = ""
        topic_match = re.search(r'(?:about|regarding|for|in)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
        if topic_match:
            research_topic = topic_match.group(1).strip()
        
        # Generate email with hybrid LLM
        generated = self.writer.write_professor_email(
            prof_name, prof_email, research_topic or "research opportunities"
        )
        
        self.pending_email = generated
        
        # Get AI analysis
        ai_score = generated.ai_analysis.get("ai_score", "N/A") if generated.ai_analysis else "N/A"
        
        response = f"**Email drafted for Professor {prof_name}**\n\n"
        response += f"**To:** {generated.to_email}\n"
        response += f"**Subject:** {generated.subject}\n"
        response += f"**AI Detection Score:** {ai_score}/100\n\n"
        response += "---\n\n"
        response += generated.body
        response += "\n\n---\n\n"
        
        # Show model status
        llm_status = self.llm.get_status()
        if llm_status['local_available']:
            response += "*Drafted with local Qwen 2.5 + humanized with Groq*\n\n"
        elif llm_status['groq_configured']:
            response += "*Generated with Groq API*\n\n"
        
        if self.email_sender.is_configured():
            response += "Type 'send' to send this email, or 'edit' to modify it."
        else:
            response += "Type 'setup email' to configure email sending, or copy the email above."
        
        return response
    
    def _handle_apply(self, post_type: str) -> str:
        """Handle scholarship/PhD application."""
        posts = self.rag_store.get_all_posts()
        relevant = [p for p in posts if p.get("metadata", {}).get("post_type") == post_type]
        
        if not relevant:
            return f"No saved {post_type} posts found. Add some first using 'add post'."
        
        if len(relevant) == 1:
            post = relevant[0]
            meta = post.get("metadata", {})
            
            post_obj = ApplicationPost(
                id=post["id"],
                title=meta.get("title", ""),
                institution=meta.get("institution", ""),
                content=post.get("content", ""),
                post_type=post_type,
                deadline=meta.get("deadline", "")
            )
            
            generated = self.writer.write_scholarship_application(post_obj)
            self.pending_email = generated
            
            response = f"**Application written for: {meta.get('title', 'Untitled')}**\n\n"
            response += f"**To:** {generated.to_email or 'N/A'}\n"
            response += f"**Subject:** {generated.subject}\n\n"
            response += "---\n\n"
            response += generated.body
            response += "\n\n---\n\n"
            response += "Type 'send' to send, or 'edit' to modify."
            
            return response
        
        response = f"**Found {len(relevant)} {post_type} posts:**\n\n"
        for i, p in enumerate(relevant, 1):
            meta = p.get("metadata", {})
            response += f"{i}. {meta.get('title', 'Untitled')} - {meta.get('institution', 'N/A')}\n"
        
        response += "\nType 'apply 1', 'apply 2', etc. to select one."
        
        return response
    
    def _handle_send_email(self) -> str:
        """Send the pending email."""
        if not self.pending_email:
            return "No email ready to send. Write an email first."
        
        if not self.email_sender.is_configured():
            return ("Email not configured. Please set up SMTP first.\n\n"
                    "Type 'setup email' to configure.\n\n"
                    "Or copy the email above and send it manually.")
        
        result = self.email_sender.send_email(
            to_email=self.pending_email.to_email,
            subject=self.pending_email.subject,
            body=self.pending_email.body,
            from_name=self.resources.name
        )
        
        if result['success']:
            self.pending_email = None
            return f"**Email sent successfully!**\n\nTo: {result['to']}\nSubject: {result['subject']}"
        else:
            return f"**Failed to send email:** {result['error']}"
    
    def _handle_setup_smtp(self, text: str) -> str:
        """Setup SMTP configuration."""
        if 'gmail' in text.lower():
            server = "smtp.gmail.com"
            port = 587
        elif 'outlook' in text.lower() or 'hotmail' in text.lower():
            server = "smtp-mail.outlook.com"
            port = 587
        elif 'yahoo' in text.lower():
            server = "smtp.mail.yahoo.com"
            port = 587
        else:
            return ("What email provider do you use?\n\n"
                    "- Gmail\n"
                    "- Outlook/Hotmail\n"
                    "- Yahoo\n"
                    "- Other (provide SMTP server)")
        
        return (f"**Configure {server}**\n\n"
                "Please provide:\n"
                "1. Your email address\n"
                "2. Your password or app-specific password\n\n"
                "**Note for Gmail:** You need to use an App Password.\n"
                "Generate one at: https://myaccount.google.com/apppasswords\n\n"
                "Type: 'email: your@email.com password: yourpassword'")
    
    def _handle_show_saved(self) -> str:
        """Show saved emails."""
        return ("**Saved Emails:**\n\n"
                "Emails are saved in the `resources/saved_responses/` folder.\n\n"
                "To view them, check the folder directly.")
    
    def _handle_help(self) -> str:
        """Show help message with model status."""
        llm_status = self.llm.get_status()
        
        response = "**Here's what I can do:**\n\n"
        
        response += "**🤖 Model Status:**\n"
        response += f"- Local Qwen 2.5: {'✅ Running' if llm_status['local_available'] else '❌ Not running'}\n"
        response += f"- Groq API: {'✅ Configured' if llm_status['groq_configured'] else '❌ Not configured'}\n\n"
        
        response += "**📄 CV Management:**\n"
        response += "- Upload CV → 'upload cv' or just upload a file\n"
        response += "- Extract info → 'extract from cv'\n\n"
        
        response += "**📝 Posts & Opportunities:**\n"
        response += "- Add post → 'add scholarship' or paste text\n"
        response += "- Add from image → upload image\n"
        response += "- Search posts → 'search posts about machine learning'\n\n"
        
        response += "**✉️ Email:**\n"
        response += "- Write email → 'write email to Dr. Smith at smith@mit.edu'\n"
        response += "- Apply → 'apply to scholarship' or 'apply for PhD'\n"
        response += "- Send email → 'send' (after writing)\n\n"
        
        response += "**⚙️ Settings:**\n"
        response += "- Setup email → 'setup email gmail'\n"
        response += "- Setup Groq → 'setup groq [api_key]'\n"
        response += "- Update profile → 'update my name to John'\n\n"
        
        response += "**Just tell me what you want to do in plain language!**"
        
        return response
    
    def _handle_profile_update(self, text: str) -> str:
        """Handle profile updates."""
        name_match = re.search(r'(?:name|called)\s+(?:to\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        if name_match:
            self.resources.name = name_match.group(1)
            self._save_resources()
            return f"Name updated to: {self.resources.name}"
        
        email_match = re.search(r'(?:email|mail)\s+(?:to\s+)?([\w.-]+@[\w.-]+\.\w+)', text)
        if email_match:
            self.resources.email = email_match.group(1)
            self._save_resources()
            return f"Email updated to: {self.resources.email}"
        
        groq_match = re.search(r'setup\s+groq\s+(\S+)', text)
        if groq_match:
            api_key = groq_match.group(1)
            self.llm.config.groq_api_key = api_key
            self.llm.config.save()
            return "Groq API configured! You can now use Groq for humanization."
        
        return ("What would you like to update?\n\n"
                "- 'update my name to [name]'\n"
                "- 'update my email to [email]'\n"
                "- 'setup groq [api_key]'")
    
    def _handle_general(self, text: str) -> str:
        """Handle general conversation."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            llm_status = self.llm.get_status()
            models = []
            if llm_status['local_available']:
                models.append("Local Qwen 2.5")
            if llm_status['groq_configured']:
                models.append("Groq API")
            
            model_str = ", ".join(models) if models else "No LLM configured"
            
            return (f"Hello{(' ' + self.resources.name) if self.resources.name else ''}! 👋\n\n"
                    "I'm your Scholarship Application Agent.\n\n"
                    f"**Active Models:** {model_str}\n\n"
                    "I can help you:\n"
                    "- Upload and parse your CV\n"
                    "- Save scholarship/position announcements\n"
                    "- Write humanized emails to professors\n"
                    "- Apply to scholarships automatically\n\n"
                    "What would you like to do?")
        
        if any(word in text_lower for word in ['thank', 'thanks']):
            return "You're welcome! Let me know if you need anything else."
        
        if 'yes' in text_lower and 'extracted_cv' in self.context:
            cv = self.context['extracted_cv']
            user_data = cv.to_dict()
            user_data["research_interests"] = []
            user_data["target_universities"] = []
            
            RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESOURCES_DIR / "user_data.json", "w") as f:
                json.dump(user_data, f, indent=2)
            
            self.resources = UserResources.load(RESOURCES_DIR)
            self.writer = EmailWriter(self.resources, self.humanizer, self.llm)
            
            return "CV data saved to your profile! You can now use it for applications."
        
        if self.pending_post_text and len(text.split()) > 5:
            post = self.post_processor.process_text(self.pending_post_text + "\n" + text)
            post_id = self.rag_store.add_post(post)
            self.pending_post_text = None
            return f"Post saved with ID: {post_id}\n\nType 'apply' to apply to this, or 'search posts' to find more."
        
        return ("I'm not sure what you mean. Type 'help' to see what I can do.\n\n"
                "Or just tell me naturally:\n"
                "- 'upload my CV'\n"
                "- 'write email to professor'\n"
                "- 'apply for scholarship'\n"
                "- 'add this post'")
    
    def _save_resources(self):
        """Save resources to file."""
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
            "target_universities": self.resources.target_universities
        }
        
        RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESOURCES_DIR / "user_data.json", "w") as f:
            json.dump(user_data, f, indent=2)
