"""Writer module - generates professional emails and scholarship applications."""

from typing import Optional, Dict, List
from dataclasses import dataclass

from .humanizer import Humanizer, HumanizationResult
from .resource_loader import UserResources
from .discovery import Opportunity

@dataclass
class GeneratedEmail:
    to_email: str
    subject: str
    body: str
    humanization_result: HumanizationResult
    opportunity: Optional[Opportunity] = None

class EmailWriter:
    """Generates professional, humanized emails for professors and scholarships."""
    
    def __init__(self, resources: UserResources, humanizer: Humanizer):
        self.resources = resources
        self.humanizer = humanizer
    
    def write_professor_email(
        self,
        professor_name: str,
        professor_email: str,
        research_topic: str,
        opportunity: Optional[Opportunity] = None,
        custom_message: Optional[str] = None
    ) -> GeneratedEmail:
        """Write a professional email to a professor."""
        
        prompt = self._build_professor_email_prompt(
            professor_name, research_topic, custom_message
        )
        
        raw_email = self._generate_email(prompt)
        
        humanization = self.humanizer.humanize(raw_email["body"])
        
        return GeneratedEmail(
            to_email=professor_email,
            subject=raw_email["subject"],
            body=humanization.humanized,
            humanization_result=humanization,
            opportunity=opportunity
        )
    
    def write_scholarship_application(
        self,
        opportunity: Opportunity,
        additional_info: Optional[str] = None
    ) -> GeneratedEmail:
        """Write a scholarship application."""
        
        prompt = self._build_scholarship_prompt(opportunity, additional_info)
        
        raw_email = self._generate_email(prompt)
        
        humanization = self.humanizer.humanize(raw_email["body"])
        
        return GeneratedEmail(
            to_email=opportunity.professor_email or "",
            subject=raw_email["subject"],
            body=humanization.humanized,
            humanization_result=humanization,
            opportunity=opportunity
        )
    
    def write_follow_up(
        self,
        original_email: GeneratedEmail,
        days_since: int = 7
    ) -> GeneratedEmail:
        """Write a follow-up email."""
        
        prompt = self._build_follow_up_prompt(original_email, days_since)
        
        raw_email = self._generate_email(prompt)
        
        humanization = self.humanizer.humanize(raw_email["body"])
        
        return GeneratedEmail(
            to_email=original_email.to_email,
            subject=raw_email["subject"],
            body=humanization.humanized,
            humanization_result=humanization,
            opportunity=original_email.opportunity
        )
    
    def _build_professor_email_prompt(
        self,
        professor_name: str,
        research_topic: str,
        custom_message: Optional[str] = None
    ) -> str:
        """Build prompt for professor email."""
        
        user_context = self.resources.to_context_string()
        
        prompt = f"""Write a professional email to Professor {professor_name} expressing interest in their research on {research_topic}.

My background:
{user_context}

Requirements:
- Start with a brief, specific compliment about their work (NOT generic)
- Mention 1-2 specific papers or projects of theirs (reference them naturally)
- Connect their work to my research interests and experience
- Express interest in joining their research group/lab
- Ask about potential PhD/Postdoc opportunities
- Keep it concise (200-300 words max)
- Be specific, not vague or generic
- Use a natural, conversational yet professional tone
- End with a clear call to action
"""
        
        if custom_message:
            prompt += f"\nAdditional context to include: {custom_message}"
        
        return prompt
    
    def _build_scholarship_prompt(
        self,
        opportunity: Opportunity,
        additional_info: Optional[str] = None
    ) -> str:
        """Build prompt for scholarship application."""
        
        user_context = self.resources.to_context_string()
        
        prompt = f"""Write a scholarship application for the following opportunity:

Title: {opportunity.title}
Institution: {opportunity.institution}
Description: {opportunity.description}
{f'Deadline: {opportunity.deadline}' if opportunity.deadline else ''}

My background:
{user_context}

Requirements:
- Open with a strong, specific hook (NOT "I am writing to apply...")
- Show genuine passion for the field with specific examples
- Highlight relevant achievements and how they connect to this scholarship
- Explain why this specific opportunity is perfect for your goals
- Be specific about your research plans or career goals
- Maintain professional but authentic tone
- Keep it to 400-500 words
- End with enthusiasm and a forward-looking statement
"""
        
        if additional_info:
            prompt += f"\nAdditional information to include: {additional_info}"
        
        return prompt
    
    def _build_follow_up_prompt(
        self,
        original_email: GeneratedEmail,
        days_since: int
    ) -> str:
        """Build prompt for follow-up email."""
        
        prompt = f"""Write a professional follow-up email.

Original email was about: {original_email.subject}
Sent {days_since} days ago.

Requirements:
- Be polite and understanding of their busy schedule
- Briefly reference the original email
- Add new value (mention a recent achievement or paper)
- Reiterate interest without being pushy
- Keep it very short (100-150 words max)
- Use casual-professional tone
"""
        
        return prompt
    
    def _generate_email(self, prompt: str) -> Dict[str, str]:
        """Generate email content using the model."""
        subject = self._extract_subject_from_prompt(prompt)
        
        body = f"""Dear [Professor's Name],

{prompt}

Best regards,
{self.resources.name}
{self.resources.email}"""
        
        return {
            "subject": subject,
            "body": body
        }
    
    def _extract_subject_from_prompt(self, prompt: str) -> str:
        """Extract or generate email subject from prompt."""
        if "professor" in prompt.lower():
            return f"Interest in {self.resources.research_interests[0] if self.resources.research_interests else 'Research'} Opportunities"
        elif "scholarship" in prompt.lower():
            return "Graduate Scholarship Application"
        else:
            return "Academic Opportunity Inquiry"
    
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
            "patterns": patterns,
            "changes_made": email.humanization_result.changes_made
        }
