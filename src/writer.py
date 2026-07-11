"""Writer module - generates professional emails using hybrid LLM."""

from typing import Optional, Dict, List
from dataclasses import dataclass

from .humanizer import Humanizer, HumanizationResult
from .resource_loader import UserResources
from .discovery import Opportunity
from .hybrid_llm import HybridLLM, LLMConfig

@dataclass
class GeneratedEmail:
    to_email: str
    subject: str
    body: str
    humanization_result: HumanizationResult
    ai_analysis: Optional[Dict] = None
    opportunity: Optional[Opportunity] = None

class EmailWriter:
    """Generates professional, humanized emails using hybrid LLM."""
    
    def __init__(self, resources: UserResources, humanizer: Humanizer, llm: Optional[HybridLLM] = None):
        self.resources = resources
        self.humanizer = humanizer
        self.llm = llm or HybridLLM()
    
    def write_professor_email(
        self,
        professor_name: str,
        professor_email: str,
        research_topic: str,
        opportunity: Optional[Opportunity] = None,
        custom_message: Optional[str] = None
    ) -> GeneratedEmail:
        """Write a professional email to a professor using hybrid LLM."""
        
        context = self.resources.to_context_string()
        
        prompt = f"""Write an email to Professor {professor_name} about their research on {research_topic}.

My background:
{context}

{f'Additional context: {custom_message}' if custom_message else ''}

Write a natural, human-sounding email. Be specific about their work. 
Reference 1-2 specific papers if possible. Keep under 300 words."""
        
        # Run hybrid pipeline: Local Llama drafts + checks, Groq humanizes
        pipeline_result = self.llm.hybrid_generate(prompt, context)
        
        # Extract results
        raw_email = pipeline_result["draft"]
        ai_analysis = pipeline_result["ai_analysis"]
        final_text = pipeline_result["humanized"]
        
        # Extract subject
        subject = f"Interest in {research_topic} - Potential Collaboration"
        
        return GeneratedEmail(
            to_email=professor_email,
            subject=subject,
            body=final_text,
            humanization_result=HumanizationResult(
                original=raw_email,
                humanized=final_text,
                patterns_found=ai_analysis.get("patterns_found", []),
                changes_made=[
                    f"Local Llama: {'Used' if pipeline_result['local_used'] else 'Skipped'}",
                    f"Groq Humanize: {'Used' if pipeline_result['groq_used'] else 'Skipped'}"
                ],
                confidence_score=1.0 - (ai_analysis.get("ai_score", 50) / 100)
            ),
            ai_analysis=ai_analysis,
            opportunity=opportunity
        )
    
    def write_scholarship_application(
        self,
        opportunity,
        additional_info: Optional[str] = None
    ) -> GeneratedEmail:
        """Write a scholarship application using hybrid LLM."""
        
        context = self.resources.to_context_string()
        
        # Support both Opportunity (has description) and ApplicationPost (has content)
        desc = getattr(opportunity, "content", None) or getattr(opportunity, "description", "")
        
        prompt = f"""Write a PhD/scholarship application email for:

Title: {opportunity.title}
Institution: {opportunity.institution}
Post details: {desc}
{f'Deadline: {opportunity.deadline}' if opportunity.deadline else ''}

My background:
{context}

{f'Additional info: {additional_info}' if additional_info else ''}

STRICT RULES:
1. MAX 200 words. Count carefully. Do NOT exceed.
2. Do NOT repeat any phrase or idea. Every sentence must say something NEW.
3. Start with "Dear Dr. [Last Name]," or "Dear Professor [Last Name],"
4. First sentence: state the specific position and one reason you're interested
5. Second paragraph: ONE specific project/skill that matches their research
6. Final sentence: "I have attached my CV, transcript, and research statement."
7. Sign off with "Best regards," then name
8. Do NOT use: "I am writing to express", "I believe", "I think", "Furthermore", "Moreover"
9. Use contractions (I'm, don't, can't)
10. Do NOT mention "interdisciplinary" or "collaboration" unless the post specifically asks for it"""
        
        # Run hybrid pipeline
        pipeline_result = self.llm.hybrid_generate(prompt, context)
        
        raw_email = pipeline_result["draft"]
        ai_analysis = pipeline_result["ai_analysis"]
        final_text = pipeline_result["humanized"]
        
        # Extract email from metadata if available
        meta = getattr(opportunity, "metadata", None) or {}
        prof_email = meta.get("professor_email", "") or getattr(opportunity, "professor_email", "") or ""
        subject_format = meta.get("subject_format", "")
        
        # Use subject format from post if specified, otherwise default
        if subject_format:
            subject = subject_format.replace("Your Name", self.resources.name or "Applicant")
        else:
            subject = f"Application for {opportunity.title}"
        
        return GeneratedEmail(
            to_email=prof_email,
            subject=subject,
            body=final_text,
            humanization_result=HumanizationResult(
                original=raw_email,
                humanized=final_text,
                patterns_found=ai_analysis.get("patterns_found", []),
                changes_made=[
                    f"Local Llama: {'Used' if pipeline_result['local_used'] else 'Skipped'}",
                    f"Groq Humanize: {'Used' if pipeline_result['groq_used'] else 'Skipped'}"
                ],
                confidence_score=1.0 - (ai_analysis.get("ai_score", 50) / 100)
            ),
            ai_analysis=ai_analysis,
            opportunity=opportunity
        )
    
    def write_follow_up(
        self,
        original_email: GeneratedEmail,
        days_since: int = 7
    ) -> GeneratedEmail:
        """Write a follow-up email."""
        
        context = self.resources.to_context_string()
        
        prompt = f"""Write a follow-up email.

Original subject: {original_email.subject}
Sent {days_since} days ago.

My background:
{context}

Write a short, polite follow-up. Reference the original email.
Add new value (recent achievement or paper). Keep under 150 words."""
        
        # Run hybrid pipeline
        pipeline_result = self.llm.hybrid_generate(prompt, context)
        
        raw_email = pipeline_result["draft"]
        ai_analysis = pipeline_result["ai_analysis"]
        final_text = pipeline_result["humanized"]
        
        return GeneratedEmail(
            to_email=original_email.to_email,
            subject=f"Re: {original_email.subject}",
            body=final_text,
            humanization_result=HumanizationResult(
                original=raw_email,
                humanized=final_text,
                patterns_found=[],
                changes_made=[
                    f"Local Llama: {'Used' if pipeline_result['local_used'] else 'Skipped'}",
                    f"Groq Humanize: {'Used' if pipeline_result['groq_used'] else 'Skipped'}"
                ],
                confidence_score=0.9
            ),
            ai_analysis=ai_analysis,
            opportunity=original_email.opportunity
        )
    
    def review_email(self, email: GeneratedEmail) -> Dict:
        """Review email for quality and humanization."""
        patterns = self.humanizer.detect_patterns(email.body)
        
        word_count = len(email.body.split())
        sentence_count = len([s for s in email.body.split('.') if s.strip()])
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "humanization_score": email.humanization_result.confidence_score,
            "ai_patterns_found": len(patterns),
            "ai_analysis": email.ai_analysis,
            "patterns": patterns,
            "changes_made": email.humanization_result.changes_made
        }
