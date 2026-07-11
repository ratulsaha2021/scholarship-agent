"""Discovery module - finds scholarship and professor opportunities."""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

RESOURCES_DIR = Path(__file__).parent.parent / "resources"

@dataclass
class Opportunity:
    type: str  # "scholarship", "phd_position", "professor"
    title: str
    institution: str
    url: str
    description: str = ""
    deadline: str = ""
    professor_email: str = ""
    requires_proposal: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "title": self.title,
            "institution": self.institution,
            "url": self.url,
            "description": self.description[:500],
            "deadline": self.deadline,
            "professor_email": self.professor_email,
            "requires_proposal": self.requires_proposal
        }

class OpportunityDiscovery:
    """Discovers scholarship and PhD opportunities."""
    
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def load_manual_targets(self, targets_file: Optional[Path] = None) -> List[Opportunity]:
        """Load manually specified targets from JSON file."""
        if targets_file is None:
            targets_file = RESOURCES_DIR / "targets.json"
        
        if not targets_file.exists():
            return []
        
        with open(targets_file, "r") as f:
            targets = json.load(f)
        
        opportunities = []
        for target in targets:
            opp = Opportunity(
                type=target.get("type", "professor"),
                title=target.get("title", ""),
                institution=target.get("institution", ""),
                url=target.get("url", ""),
                description=target.get("description", ""),
                deadline=target.get("deadline", ""),
                professor_email=target.get("email", ""),
                requires_proposal=target.get("requires_proposal", False)
            )
            opportunities.append(opp)
        
        return opportunities
    
    def search_academic_positions(self, query: str, max_results: int = 10) -> List[Opportunity]:
        """Search for academic positions online."""
        opportunities = []
        
        try:
            search_url = f"https://scholarshipdb.net/scholarships?q={query}"
            time.sleep(self.delay)
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                results = soup.find_all("div", class_="scholarship-item")[:max_results]
                
                for result in results:
                    title_elem = result.find("h3") or result.find("h4")
                    link_elem = result.find("a")
                    
                    if title_elem and link_elem:
                        opp = Opportunity(
                            type="scholarship",
                            title=title_elem.get_text(strip=True),
                            institution=self._extract_institution(result),
                            url=link_elem.get("href", ""),
                            description=self._extract_description(result),
                            deadline=self._extract_deadline(result)
                        )
                        opportunities.append(opp)
        except Exception as e:
            print(f"Search error: {e}")
        
        return opportunities
    
    def find_professor_emails(self, department_url: str) -> List[Dict]:
        """Find professor emails from a department page."""
        professors = []
        
        try:
            time.sleep(self.delay)
            response = self.session.get(department_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
                emails = set(email_pattern.findall(response.text))
                
                for email in emails:
                    if not any(x in email.lower() for x in ["noreply", "support", "admin", "webmaster"]):
                        professors.append({
                            "email": email,
                            "department_url": department_url
                        })
        except Exception as e:
            print(f"Error fetching professor emails: {e}")
        
        return professors
    
    def search_research_gate(self, query: str) -> List[Opportunity]:
        """Search ResearchGate for opportunities."""
        opportunities = []
        
        try:
            time.sleep(self.delay)
            search_url = f"https://www.researchgate.net/search?q={query}%20PhD%20position"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                results = soup.find_all("li", class_="nova-legacy-elevated-card")[:5]
                
                for result in results:
                    title_elem = result.find("h2")
                    if title_elem:
                        opp = Opportunity(
                            type="phd_position",
                            title=title_elem.get_text(strip=True),
                            institution="",
                            url="https://www.researchgate.net" + (title_elem.find("a") or {}).get("href", ""),
                            description=self._extract_description(result)
                        )
                        opportunities.append(opp)
        except Exception as e:
            print(f"ResearchGate search error: {e}")
        
        return opportunities
    
    def _extract_institution(self, element) -> str:
        """Extract institution name from HTML element."""
        inst_elem = element.find(class_="institution") or element.find("span", class_="uni")
        return inst_elem.get_text(strip=True) if inst_elem else ""
    
    def _extract_description(self, element) -> str:
        """Extract description from HTML element."""
        desc_elem = element.find("p") or element.find("div", class_="description")
        return desc_elem.get_text(strip=True)[:500] if desc_elem else ""
    
    def _extract_deadline(self, element) -> str:
        """Extract deadline from HTML element."""
        deadline_elem = element.find(class_="deadline") or element.find("span", string=re.compile(r"deadline|due|close", re.I))
        return deadline_elem.get_text(strip=True) if deadline_elem else ""
    
    def save_opportunities(self, opportunities: List[Opportunity], filename: str = "discovered_opportunities.json"):
        """Save discovered opportunities to file."""
        output_file = RESOURCES_DIR / filename
        
        data = [opp.to_dict() for opp in opportunities]
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return output_file

def create_sample_targets():
    """Create sample targets file."""
    targets = [
        {
            "type": "professor",
            "title": "PhD Position in Machine Learning",
            "institution": "MIT",
            "url": "https://www.csail.mit.edu/research",
            "email": "professor@mit.edu",
            "description": "Looking for PhD students in ML/NLP",
            "requires_proposal": True
        },
        {
            "type": "scholarship",
            "title": "Graduate Fellowship",
            "institution": "Stanford University",
            "url": "https://stanford.edu/fellowships",
            "deadline": "December 15, 2024",
            "description": "Full funding for PhD students"
        }
    ]
    
    targets_file = RESOURCES_DIR / "targets.json"
    with open(targets_file, "w") as f:
        json.dump(targets, f, indent=2)
    
    print(f"Sample targets created at {targets_file}")
