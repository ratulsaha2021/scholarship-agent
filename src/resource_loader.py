"""Resource loader module - handles PDF/Word CV parsing and resource management."""

import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document
except ImportError:
    Document = None

RESOURCES_DIR = Path(__file__).parent.parent / "resources"

@dataclass
class UserResources:
    name: str = ""
    email: str = ""
    phone: str = ""
    research_interests: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    experience: List[Dict] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    cv_text: str = ""
    cover_letter_template: str = ""
    target_universities: List[str] = field(default_factory=list)
    additional_notes: str = ""
    
    @classmethod
    def load(cls, resources_dir: Optional[Path] = None) -> "UserResources":
        """Load all user resources from the resources directory."""
        if resources_dir is None:
            resources_dir = RESOURCES_DIR
        
        resources = cls()
        
        resources.cv_text = cls._load_cv(resources_dir)
        resources.research_interests = cls._load_text_list(resources_dir / "research_interests.txt")
        resources.skills = cls._load_text_list(resources_dir / "skills.txt")
        resources.cover_letter_template = cls._load_text(resources_dir / "cover_letter_template.txt")
        resources.additional_notes = cls._load_text(resources_dir / "additional_notes.txt")
        
        data_file = resources_dir / "user_data.json"
        if data_file.exists():
            with open(data_file, "r") as f:
                data = json.load(f)
            resources.name = data.get("name", "")
            resources.email = data.get("email", "")
            resources.phone = data.get("phone", "")
            resources.education = data.get("education", [])
            resources.experience = data.get("experience", [])
            resources.publications = data.get("publications", [])
            resources.awards = data.get("awards", [])
            resources.target_universities = data.get("target_universities", [])
        
        return resources
    
    @classmethod
    def _load_cv(cls, resources_dir: Path) -> str:
        """Load CV from PDF or Word document."""
        pdf_path = resources_dir / "cv.pdf"
        docx_path = resources_dir / "cv.docx"
        
        if pdf_path.exists() and pdfplumber:
            return cls._extract_pdf(pdf_path)
        elif docx_path.exists() and Document:
            return cls._extract_docx(docx_path)
        else:
            txt_path = resources_dir / "cv.txt"
            if txt_path.exists():
                return cls._load_text(txt_path)
            return ""
    
    @classmethod
    def _extract_pdf(cls, pdf_path: Path) -> str:
        """Extract text from PDF."""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n\n".join(text)
    
    @classmethod
    def _extract_docx(cls, docx_path: Path) -> str:
        """Extract text from Word document."""
        doc = Document(docx_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    @classmethod
    def _load_text(cls, file_path: Path) -> str:
        """Load text from file."""
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return ""
    
    @classmethod
    def _load_text_list(cls, file_path: Path) -> List[str]:
        """Load list from text file (one item per line)."""
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return [line.strip() for line in content.split("\n") if line.strip()]
        return []
    
    def to_context_string(self) -> str:
        """Convert resources to context string for the model."""
        parts = []
        
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.email:
            parts.append(f"Email: {self.email}")
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        
        if self.research_interests:
            parts.append(f"\nResearch Interests: {', '.join(self.research_interests)}")
        
        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills)}")
        
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
        
        if self.publications:
            parts.append("\nPublications:")
            for pub in self.publications:
                parts.append(f"- {pub}")
        
        if self.awards:
            parts.append("\nAwards:")
            for award in self.awards:
                parts.append(f"- {award}")
        
        if self.cv_text:
            parts.append(f"\nFull CV Text:\n{self.cv_text[:3000]}")
        
        if self.additional_notes:
            parts.append(f"\nAdditional Notes: {self.additional_notes}")
        
        return "\n".join(parts)

def create_sample_resources():
    """Create sample resource files for testing."""
    resources_dir = RESOURCES_DIR
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    user_data = {
        "name": "Your Name",
        "email": "your.email@example.com",
        "phone": "+1-234-567-8900",
        "education": [
            {
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "institution": "Your University",
                "year": "2023"
            }
        ],
        "experience": [
            {
                "title": "Research Assistant",
                "organization": "AI Lab",
                "duration": "2022-2023",
                "description": "Worked on machine learning projects"
            }
        ],
        "publications": [],
        "awards": [],
        "target_universities": ["MIT", "Stanford", "CMU"]
    }
    
    with open(resources_dir / "user_data.json", "w") as f:
        json.dump(user_data, f, indent=2)
    
    research_file = resources_dir / "research_interests.txt"
    research_file.write_text("Machine Learning\nNatural Language Processing\nComputer Vision\n")
    
    skills_file = resources_dir / "skills.txt"
    skills_file.write_text("Python\nPyTorch\nTensorFlow\nSQL\nResearch\n")
    
    print(f"Sample resources created in {resources_dir}")
    print("Please edit the files with your actual information.")
