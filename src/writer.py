"""Writer module: application emails, statements of purpose and research proposals."""

from typing import Optional, Dict, List
from dataclasses import dataclass, field

from .humanizer import Humanizer, HumanizationResult, Issue
from .resource_loader import UserResources
from .discovery import Opportunity
from .hybrid_llm import HybridLLM, NO_FABRICATION_RULE

DOC_NAMES = {"sop": "Statement of Purpose", "proposal": "Research Proposal"}


@dataclass
class GeneratedEmail:
    to_email: str
    subject: str
    body: str
    humanization_result: HumanizationResult
    ai_analysis: Optional[Dict] = None
    opportunity: Optional[Opportunity] = None
    error: str = ""  # set when no LLM produced the draft
    issues: List[Issue] = field(default_factory=list)
    fact_fixes: List[Dict] = field(default_factory=list)     # unsupported claims that were corrected
    fact_problems: List[Dict] = field(default_factory=list)  # still unsupported after correction


@dataclass
class GeneratedDocument:
    kind: str  # "sop" or "proposal"
    title: str
    text: str
    post_title: str = ""
    issues: List[Issue] = field(default_factory=list)
    error: str = ""
    fact_fixes: List[Dict] = field(default_factory=list)
    fact_problems: List[Dict] = field(default_factory=list)


def _join_names(names: List[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


class EmailWriter:
    """Generates professional, humanized application writing using the hybrid LLM."""

    def __init__(self, resources: UserResources, humanizer: Humanizer, llm: Optional[HybridLLM] = None):
        self.resources = resources
        self.humanizer = humanizer
        self.llm = llm or HybridLLM()

    # ---------- helpers ----------

    def _name(self) -> str:
        name = self.resources.name or ""
        return name.title() if name.isupper() else name

    def _signature(self) -> str:
        lines = [self._name()]
        contact = " | ".join(p for p in [self.resources.email, self.resources.phone] if p)
        if contact:
            lines.append(contact)
        return "\n".join(l for l in lines if l)

    @staticmethod
    def _post_block(opportunity) -> str:
        # Supports both Opportunity (description) and ApplicationPost (content)
        desc = getattr(opportunity, "content", None) or getattr(opportunity, "description", "")
        parts = [f"Title: {opportunity.title}"]
        if opportunity.institution:
            parts.append(f"Institution: {opportunity.institution}")
        if opportunity.deadline:
            parts.append(f"Deadline: {opportunity.deadline}")
        parts.append(f"Post text:\n{desc}")
        return "\n".join(parts)

    @staticmethod
    def _closing_rule(attachments: Optional[List[str]]) -> str:
        if attachments:
            return f'Mention once, near the end, that you\'ve attached your {_join_names(attachments)}.'
        return "Offer to send your CV and other documents. Do NOT claim anything is attached."

    def _email_result(self, to_email, subject, pipeline, opportunity=None) -> GeneratedEmail:
        ai = pipeline.get("ai_analysis") or {}
        return GeneratedEmail(
            to_email=to_email,
            subject=subject,
            body=pipeline["humanized"],
            humanization_result=HumanizationResult(
                original=pipeline["draft"],
                humanized=pipeline["humanized"],
                patterns_found=ai.get("patterns_found", []),
                changes_made=[
                    f"Local model: {'used' if pipeline['local_used'] else 'skipped'}",
                    f"Groq: {'used' if pipeline['groq_used'] else 'skipped'}",
                    f"Style rewrites: {pipeline.get('rewrites', 0)}",
                ],
                confidence_score=1.0 - ai.get("ai_score", 0) / 100,
            ),
            ai_analysis=ai,
            opportunity=opportunity,
            error=pipeline.get("error", "") if pipeline.get("fallback") else "",
            issues=pipeline.get("issues", []),
            fact_fixes=pipeline.get("fact_fixes", []),
            fact_problems=pipeline.get("fact_problems", []),
        )

    # ---------- emails ----------

    def write_professor_email(
        self,
        professor_name: str,
        professor_email: str,
        research_topic: str,
        opportunity: Optional[Opportunity] = None,
        custom_message: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> GeneratedEmail:
        """Cold email to a professor about joining their group."""
        context = self.resources.to_context_string()
        prompt = f"""Write a cold email to Professor {professor_name} asking about PhD opportunities in their group.
Their research area: {research_topic}
{f'Extra context from the applicant: {custom_message}' if custom_message else ''}

Structure:
1. "Dear Professor {professor_name.split()[-1] if professor_name else ''},"
2. One or two sentences: who you are (current degree/role) and why you're writing to them, tied to their area.
3. One short paragraph: the one project, paper or skill from your background that best connects to their area, with its concrete result.
4. A clear ask: whether they're taking PhD students, and whether a short call is possible.
5. {self._closing_rule(attachments)}
6. "Best regards," then this signature exactly:
{self._signature()}

Length: 150-200 words. Don't claim to have read specific papers of theirs unless named above.
Return only the email body."""
        pipeline = self.llm.hybrid_generate(prompt, context, kind="email")
        subject = f"Prospective PhD student: {research_topic}"
        return self._email_result(professor_email, subject, pipeline, opportunity)

    def write_scholarship_application(
        self,
        opportunity,
        additional_info: Optional[str] = None,
        cv_attached: bool = False,
        attachments: Optional[List[str]] = None,
    ) -> GeneratedEmail:
        """Application email for a posted position or scholarship."""
        if attachments is None and cv_attached:
            attachments = ["CV"]
        context = self.resources.to_context_string()
        prompt = f"""Write an application email for this position:

{self._post_block(opportunity)}
{f'Extra notes from the applicant: {additional_info}' if additional_info else ''}

Structure:
1. Greeting: if the post names a contact person or supervisor, use their title and surname ("Dear Professor Smith," / "Dear Dr. Rahman,"). If no one is named, "Dear Hiring Committee,". Never write placeholders.
2. Opening (1-2 sentences): the exact position you're applying for and your current degree or role.
3. Fit (one paragraph): the one project, paper or skill from the background that best matches what the post asks for, with its concrete result. Name the match explicitly.
4. {self._closing_rule(attachments)} If the post asks for other documents that aren't in that list, don't claim you've prepared or sent them.
5. "Best regards," then this signature exactly:
{self._signature()}

Length: 150-220 words. Follow any instructions in the post about format or content.
Return only the email body, no subject line."""
        pipeline = self.llm.hybrid_generate(prompt, context, kind="email")

        meta = getattr(opportunity, "metadata", None) or {}
        prof_email = meta.get("professor_email", "") or getattr(opportunity, "professor_email", "") or ""
        subject_format = meta.get("subject_format", "")
        if subject_format:
            subject = subject_format.replace("Your Name", self._name() or "Applicant")
            subject = " ".join(subject.split())
        else:
            subject = f"Application: {opportunity.title}"
        return self._email_result(prof_email, subject, pipeline, opportunity)

    def write_follow_up(self, original_email: GeneratedEmail, days_since: int = 7) -> GeneratedEmail:
        context = self.resources.to_context_string()
        prompt = f"""Write a short follow-up to an application email sent {days_since} days ago.
Original subject: {original_email.subject}

Politely ask whether they had a chance to review it. If the background has anything newer than the
original email's content, mention it in one sentence; otherwise don't add news.
End with "Best regards," and this signature:
{self._signature()}
Length: under 100 words. Return only the email body."""
        pipeline = self.llm.hybrid_generate(prompt, context, kind="email")
        return self._email_result(original_email.to_email, f"Re: {original_email.subject}",
                                  pipeline, original_email.opportunity)

    def revise_email(self, email: GeneratedEmail, instruction: str) -> GeneratedEmail:
        context = self.resources.to_context_string()
        prompt = f"""Revise this application email.

Instruction: {instruction}

Current email:
{email.body}

Apply the instruction and keep everything else. {NO_FABRICATION_RULE}
Return ONLY the email body (greeting to signature). No subject line, no explanations."""
        return self._revise(email, prompt, context)

    def _revise(self, email: GeneratedEmail, prompt: str, context: str) -> GeneratedEmail:
        try:
            edited = self.llm.generate(prompt, use_groq=True)
        except Exception as e:
            return self._email_result(email.to_email, email.subject,
                                      {"draft": "", "humanized": email.body, "fallback": True,
                                       "error": str(e), "local_used": False, "groq_used": False},
                                      email.opportunity)
        if not edited.strip() or edited == "No LLM available.":
            return self._email_result(email.to_email, email.subject,
                                      {"draft": "", "humanized": email.body, "fallback": True,
                                       "error": "No LLM available.", "local_used": False, "groq_used": False},
                                      email.opportunity)
        pipeline = self.llm.polish(edited, "email", context)
        pipeline.setdefault("draft", edited)
        return self._email_result(email.to_email, email.subject, pipeline, email.opportunity)

    # ---------- statement of purpose / research proposal ----------

    def write_sop(self, opportunity=None, notes: str = "", word_limit: str = "") -> GeneratedDocument:
        context = self.resources.to_context_string()
        target = self._post_block(opportunity) if opportunity else "No specific program given: write a general SOP for a PhD in the applicant's field."
        motivation = (f"The applicant's own notes on motivation, fit and goals (use these, in their words where possible):\n{notes}"
                      if notes else
                      "The applicant gave no personal notes. Do NOT invent personal stories, childhood moments, "
                      "feelings or life events. Ground the motivation in the work listed in the background.")
        prompt = f"""Write a statement of purpose for this application.

{target}

{motivation}

Format: 6-8 paragraphs of flowing prose. No headings, no bullet lists, no title line.
Do NOT reproduce the CV: select what supports the argument and explain it.

Content, in this order:
1. Opening: the research problem or question the applicant wants to work on, stated concretely. No quotes, no "since childhood", no sweeping claims about the field.
2. Academic background: degrees and the specific courses, thesis or projects that prepared them.
3. Research experience: what they did, the methods, and the results (papers, numbers) exactly as in the background.
4. Why this program: only details the post actually gives (lab, topic, supervisor, resources). Don't invent faculty names or facilities.
5. Goals: what they want to do during and after the degree (from the notes if given; otherwise keep it brief and general).
6. A short closing paragraph that adds something (e.g. what they'd start with), not a summary.

Length: {word_limit or "follow any limit in the post; otherwise 800-1000 words"}.
First person, formal but plain. Return only the statement text."""
        pipeline = self.llm.hybrid_generate(prompt, context, kind="sop", max_tokens=4000)
        return self._doc_result("sop", opportunity, pipeline)

    def write_proposal(self, opportunity=None, topic: str = "", word_limit: str = "") -> GeneratedDocument:
        context = self.resources.to_context_string()
        target = self._post_block(opportunity) if opportunity else "No specific position given."
        topic_line = (f"The applicant's proposed topic / idea:\n{topic}" if topic else
                      "The applicant gave no topic. Propose one that fits both the post's research area and the applicant's background.")
        prompt = f"""Write a PhD research proposal.

{target}

{topic_line}

Format (markdown):
# <A specific, descriptive title>
## Background and motivation
## Research questions
(2-4 numbered questions)
## Methodology
(concrete methods, data and evaluation for each question)
## Expected contributions
## Work plan
(a timeline by year or semester over a 3-4 year PhD)
## References
(only real publications; not projects or unpublished work)

Citation rules (important):
- Cite only the applicant's own publications from the background and works named in the post or notes.
- Where a claim needs support you don't have, write [citation needed]. Never invent authors, titles, venues or years.
- Name datasets, tools or models only if they appear in the background, post or notes, or are standard and widely known in the field.

Length: {word_limit or "follow any limit in the post; otherwise 1200-1500 words"}.
Headings in sentence case. Plain academic prose. Return only the proposal."""
        pipeline = self.llm.hybrid_generate(prompt, context, kind="proposal", max_tokens=5000)
        return self._doc_result("proposal", opportunity, pipeline)

    def revise_document(self, doc: GeneratedDocument, instruction: str) -> GeneratedDocument:
        context = self.resources.to_context_string()
        prompt = f"""Revise this {DOC_NAMES[doc.kind].lower()}.

Instruction: {instruction}

Current text:
{doc.text}

Apply the instruction and keep everything else, including the structure. {NO_FABRICATION_RULE}
{"Keep citation rules: no invented references; use [citation needed] instead." if doc.kind == "proposal" else ""}
Return ONLY the revised text."""
        try:
            edited = self.llm.generate(prompt, use_groq=True, max_tokens=5000)
        except Exception as e:
            return GeneratedDocument(doc.kind, doc.title, doc.text, doc.post_title, doc.issues, error=str(e))
        if not edited.strip() or edited == "No LLM available.":
            return GeneratedDocument(doc.kind, doc.title, doc.text, doc.post_title, doc.issues,
                                     error="No LLM available.")
        pipeline = self.llm.polish(edited, doc.kind, context, max_tokens=5000)
        return GeneratedDocument(doc.kind, doc.title, pipeline["humanized"], doc.post_title, pipeline["issues"])

    def _doc_result(self, kind, opportunity, pipeline) -> GeneratedDocument:
        post_title = getattr(opportunity, "title", "") if opportunity else ""
        error = pipeline.get("error", "") if pipeline.get("fallback") else ""
        return GeneratedDocument(kind=kind, title=DOC_NAMES[kind], text=pipeline["humanized"],
                                 post_title=post_title, issues=pipeline.get("issues", []), error=error,
                                 fact_fixes=pipeline.get("fact_fixes", []),
                                 fact_problems=pipeline.get("fact_problems", []))

    # ---------- review (CLI) ----------

    def review_email(self, email: GeneratedEmail) -> Dict:
        patterns = self.humanizer.detect_patterns(email.body)
        return {
            "word_count": len(email.body.split()),
            "sentence_count": len([s for s in email.body.split('.') if s.strip()]),
            "humanization_score": email.humanization_result.confidence_score,
            "ai_patterns_found": len(patterns),
            "ai_analysis": email.ai_analysis,
            "patterns": patterns,
            "changes_made": email.humanization_result.changes_made,
        }
