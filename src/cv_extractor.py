"""CV Extractor - automatically extracts structured data from CVs."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document
except ImportError:
    Document = None

@dataclass
class ExtractedCV:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    website: str = ""
    summary: str = ""
    education: List[Dict] = field(default_factory=list)
    experience: List[Dict] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "linkedin": self.linkedin,
            "website": self.website,
            "summary": self.summary,
            "education": self.education,
            "experience": self.experience,
            "skills": self.skills,
            "publications": self.publications,
            "awards": self.awards,
            "certifications": self.certifications,
            "languages": self.languages
        }
    
    def to_context_string(self) -> str:
        """Convert to context string for LLM."""
        parts = []
        
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.email:
            parts.append(f"Email: {self.email}")
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.summary:
            parts.append(f"\nSummary: {self.summary}")
        
        if self.education:
            parts.append("\nEducation:")
            for edu in self.education:
                parts.append(f"- {edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')} ({edu.get('year', '')})")
        
        if self.experience:
            parts.append("\nExperience:")
            for exp in self.experience:
                parts.append(f"- {exp.get('title', '')} at {exp.get('organization', '')} ({exp.get('duration', '')})")
                if 'description' in exp:
                    parts.append(f"  {exp['description']}")
        
        if self.skills:
            parts.append(f"\nSkills: {', '.join(self.skills)}")
        
        if self.publications:
            parts.append("\nPublications:")
            for pub in self.publications:
                parts.append(f"- {pub}")
        
        if self.awards:
            parts.append("\nAwards:")
            for award in self.awards:
                parts.append(f"- {award}")
        
        if self.certifications:
            parts.append(f"\nCertifications: {', '.join(self.certifications)}")
        
        return "\n".join(parts)

class CVExtractor:
    """Extracts structured data from CV documents."""
    
    def extract_from_file(self, file_path: Path) -> ExtractedCV:
        """Extract CV data from file."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            text = self._extract_pdf(file_path)
        elif suffix == ".docx":
            text = self._extract_docx(file_path)
        elif suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return self._parse_cv_text(text)
    
    def extract_from_text(self, text: str) -> ExtractedCV:
        """Extract CV data from raw text."""
        return self._parse_cv_text(text)
    
    def _extract_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        if not pdfplumber:
            raise ImportError("pdfplumber required for PDF extraction")
        
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n\n".join(text)
    
    def _extract_docx(self, docx_path: Path) -> str:
        """Extract text from Word document."""
        if not Document:
            raise ImportError("python-docx required for DOCX extraction")
        
        doc = Document(docx_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    def _parse_cv_text(self, text: str) -> ExtractedCV:
        """Parse CV text and extract structured data."""
        cv = ExtractedCV(raw_text=text)
        
        cv.email = self._extract_email(text)
        cv.phone = self._extract_phone(text)
        cv.linkedin = self._extract_linkedin(text)
        cv.website = self._extract_website(text)
        cv.name = self._extract_name(text)
        cv.skills = self._extract_skills(text)
        cv.education = self._extract_education(text)
        cv.experience = self._extract_experience(text)
        cv.publications = self._extract_publications(text)
        cv.awards = self._extract_awards(text)
        cv.summary = self._extract_summary(text)
        
        return cv
    
    def _extract_email(self, text: str) -> str:
        """Extract email address."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, text)
        return match.group(0) if match else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number."""
        patterns = [
            r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\d{10}\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""
    
    def _extract_linkedin(self, text: str) -> str:
        """Extract LinkedIn URL."""
        pattern = r'linkedin\.com/in/[a-zA-Z0-9_-]+'
        match = re.search(pattern, text)
        return match.group(0) if match else ""
    
    def _extract_website(self, text: str) -> str:
        """Extract personal website."""
        pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9-]+)*'
        match = re.search(pattern, text)
        return match.group(0) if match else ""
    
    def _extract_name(self, text: str) -> str:
        """Extract name (typically first line or after specific keywords)."""
        lines = text.strip().split('\n')
        
        if lines:
            first_line = lines[0].strip()
            if len(first_line.split()) <= 4 and not any(c in first_line for c in ['@', 'http', 'www']):
                return first_line
        
        name_patterns = [
            r'(?:Name|NAME)[:\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
            r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills section."""
        skills = []
        
        skill_section = re.search(r'(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|EXPERTISE)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        if skill_section:
            skill_text = skill_section.group(1)
            skills = [s.strip() for s in re.split(r'[,•\-\n]', skill_text) if s.strip() and len(s.strip()) > 1]
        
        common_skills = [
            'Python', 'Java', 'JavaScript', 'C\+\+', 'SQL', 'R', 'MATLAB',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
            'React', 'Node.js', 'Django', 'Flask', 'AWS', 'Docker', 'Git'
        ]
        
        for skill in common_skills:
            if re.search(r'\b' + skill.lower() + r'\b', text.lower()):
                if skill not in skills:
                    skills.append(skill)
        
        return skills[:20]
    
    def _extract_education(self, text: str) -> List[Dict]:
        """Extract education entries."""
        education = []
        
        edu_section = re.search(r'(?:EDUCATION|ACADEMIC BACKGROUND)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        if edu_section:
            edu_text = edu_section.group(1)
            entries = re.split(r'\n(?=[A-Z])', edu_text)
            
            for entry in entries:
                if entry.strip():
                    edu = {}
                    
                    degree_match = re.search(r'(Bachelor|Master|Ph\.?D?|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|MBA|PhD)[^,]*', entry, re.IGNORECASE)
                    if degree_match:
                        edu['degree'] = degree_match.group(0).strip()
                    
                    inst_match = re.search(r'(?:at|from|@)\s+([A-Z][A-Za-z\s&]+(?:University|Institute|College|School))', entry)
                    if inst_match:
                        edu['institution'] = inst_match.group(1).strip()
                    
                    year_match = re.search(r'20\d{2}(?:\s*[-–]\s*(?:20\d{2}|Present|Current))?', entry)
                    if year_match:
                        edu['year'] = year_match.group(0).strip()
                    
                    if edu:
                        education.append(edu)
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience."""
        experience = []
        
        exp_section = re.search(r'(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        if exp_section:
            exp_text = exp_section.group(1)
            entries = re.split(r'\n(?=[A-Z])', exp_text)
            
            for entry in entries:
                if entry.strip() and len(entry.strip()) > 20:
                    exp = {}
                    
                    lines = entry.strip().split('\n')
                    if lines:
                        first_line = lines[0]
                        parts = re.split(r'\s+at\s+|\s+@\s+', first_line, maxsplit=1)
                        if len(parts) == 2:
                            exp['title'] = parts[0].strip()
                            exp['organization'] = parts[1].strip()
                        else:
                            exp['title'] = first_line.strip()
                        
                        year_match = re.search(r'20\d{2}(?:\s*[-–]\s*(?:20\d{2}|Present|Current))?', entry)
                        if year_match:
                            exp['duration'] = year_match.group(0).strip()
                        
                        if len(lines) > 1:
                            exp['description'] = ' '.join(lines[1:]).strip()[:200]
                    
                    if exp:
                        experience.append(exp)
        
        return experience
    
    def _extract_publications(self, text: str) -> List[str]:
        """Extract publications."""
        publications = []
        
        pub_section = re.search(r'(?:PUBLICATIONS|PAPERS|RESEARCH)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        if pub_section:
            pub_text = pub_section.group(1)
            lines = [l.strip() for l in pub_text.split('\n') if l.strip()]
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['journal', 'conference', 'proceedings', 'arxiv', 'published']):
                    publications.append(line[:200])
        
        return publications[:10]
    
    def _extract_awards(self, text: str) -> List[str]:
        """Extract awards and honors."""
        awards = []
        
        award_section = re.search(r'(?:AWARDS|HONORS|SCHOLARSHIPS|FELLOWSHIPS)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        if award_section:
            award_text = award_section.group(1)
            awards = [l.strip() for l in award_text.split('\n') if l.strip()]
        
        return awards[:10]
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary/objective."""
        summary_section = re.search(r'(?:SUMMARY|OBJECTIVE|PROFILE|ABOUT)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        if summary_section:
            return summary_section.group(1).strip()[:500]
        
        lines = text.strip().split('\n')
        for i, line in enumerate(lines[:10]):
            if len(line.split()) > 20:
                return line.strip()[:500]
        
        return ""
