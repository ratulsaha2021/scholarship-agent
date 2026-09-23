"""Discovery module - finds scholarship and professor opportunities."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

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
    
    def search_academic_positions(self, query: str, max_results: int = 10, profile: str = "") -> List[Opportunity]:
        """Search EURAXESS and jobs.ac.uk (sites whose robots.txt allows it), best match first."""
        from .sources import find_opportunities
        from .web_fetch import PoliteFetcher

        listings, errors = find_opportunities(query, profile, PoliteFetcher(delay=self.delay))
        for e in errors:
            print(f"Search error: {e}")
        return [
            Opportunity(
                type="phd_position" if "phd" in l.title.lower() else "scholarship",
                title=l.title, institution=l.institution, url=l.url,
                description=l.snippet, deadline=l.deadline,
            )
            for l in listings[:max_results]
        ]

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
