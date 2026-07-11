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
        
        # Step 1: Draft with local model (or Groq fallback)
        raw_email = self.llm.draft_email(prompt, context)
        
        # Step 2: Check for AI patterns
        ai_analysis = self.llm.check_ai_patterns(raw_email)
        
        # Step 3: Humanize using pattern replacement
        humanization = self.humanizer.humanize(raw_email)
        
        # Step 4: Final humanization pass with Groq
        final_text = self.llm.humanize_text(humanization.humanized, ai_analysis)
        
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
                changes_made=["Hybrid LLM processing"],
                confidence_score=1.0 - (ai_analysis.get("ai_score", 50) / 100)
            ),
            ai_analysis=ai_analysis,
            opportunity=opportunity
        )
    
    def write_scholarship_application(
        self,
        opportunity: Opportunity,
        additional_info: Optional[str] = None
    ) -> GeneratedEmail:
        """Write a scholarship application using hybrid LLM."""
        
        context = self.resources.to_context_string()
        
        prompt = f"""Write a scholarship application for:

Title: {opportunity.title}
Institution: {opportunity.institution}
Description: {opportunity.description}
{f'Deadline: {opportunity.deadline}' if opportunity.deadline else ''}

My background:
{context}

{f'Additional info: {additional_info}' if additional_info else ''}

Write a compelling, human-sounding application. Show genuine passion.
Be specific about achievements and goals. Keep under 500 words."""
        
        # Step 1: Draft
        raw_email = self.llm.draft_email(prompt, context)
        
        # Step 2: Check AI patterns
        ai_analysis = self.llm.check_ai_patterns(raw_email)
        
        # Step 3: Humanize
        humanization = self.humanizer.humanize(raw_email)
        
        # Step 4: Final pass
        final_text = self.llm.humanize_text(humanization.humanized, ai_analysis)
        
        return GeneratedEmail(
            to_email=opportunity.professor_email or "",
            subject=f"Application for {opportunity.title}",
            body=final_text,
            humanization_result=HumanizationResult(
                original=raw_email,
                humanized=final_text,
                patterns_found=ai_analysis.get("patterns_found", []),
                changes_made=["Hybrid LLM processing"],
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
        
        raw_email = self.llm.draft_email(prompt, context)
        ai_analysis = self.llm.check_ai_patterns(raw_email)
        humanization = self.humanizer.humanize(raw_email)
        final_text = self.llm.humanize_text(humanization.humanized, ai_analysis)
        
        return GeneratedEmail(
            to_email=original_email.to_email,
            subject=f"Re: {original_email.subject}",
            body=final_text,
            humanization_result=HumanizationResult(
                original=raw_email,
                humanized=final_text,
                patterns_found=[],
                changes_made=["Hybrid LLM processing"],
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
