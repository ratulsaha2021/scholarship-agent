"""Shared types for job/scholarship sources."""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class Listing:
    title: str
    url: str
    source: str
    institution: str = ""
    location: str = ""
    deadline: str = ""
    snippet: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def search_text(self) -> str:
        return " ".join([self.title, self.institution, self.location, self.snippet])
