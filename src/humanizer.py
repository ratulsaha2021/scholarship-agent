"""Humanization module - detects and avoids AI writing patterns."""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

CONFIG_DIR = Path(__file__).parent.parent / "config"

# AI writing patterns to avoid (based on Wikipedia AI detection research)
DEFAULT_AI_PATTERNS = {
    "overused_phrases": [
        "in conclusion", "it is important to note", "furthermore",
        "moreover", "in addition", "significantly", "notably",
        "crucial", "paramount", "substantially", "comprehensive",
        "multifaceted", "myriad", "plethora", "utilize",
        "facilitate", "endeavor", "commence", "approximately",
        "regarding", "pertaining to", "in terms of", "with respect to"
    ],
    "structural_tells": [
        "transition words at sentence starts",
        "perfect paragraph structure",
        "no contractions",
        "overly formal tone throughout",
        "listing with semicolons",
        "passive voice dominance"
    ],
    "vocabulary_tells": [
        "delve", "embark", "landscape", "realm", "tapestry",
        "landscape", "underscores", "underscoring", "pivotal",
        "nuanced", "holistic", "synergy", "leverage",
        "foster", "nurture", "spearhead", "revolutionize"
    ],
    "sentence_patterns": [
        r"^[A-Z][a-z]+(?:ly)?,?\s",  # Starts with adverb
        r"(?:Furthermore|Moreover|Additionally|Consequently),?\s",
        r"(?:It is|There is|It's worth) (?:important|crucial|essential) to",
        r"(?:I am|I'm) writing to (?:express|inquire|apply)",
        r"(?:I believe|I think|In my opinion) that"
    ]
}

@dataclass
class HumanizationResult:
    original: str
    humanized: str
    patterns_found: List[str]
    changes_made: List[str]
    confidence_score: float  # 0-1, higher = more human-like

class Humanizer:
    """Detects and avoids AI writing patterns."""
    
    def __init__(self, level: str = "high"):
        self.level = level
        self.patterns = self._load_patterns()
        self.human_transitions = self._get_human_transitions()
        self.contraction_map = self._get_contraction_map()
    
    def _load_patterns(self) -> Dict:
        """Load AI patterns from config or use defaults."""
        config_path = CONFIG_DIR / "ai_patterns.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        
        return DEFAULT_AI_PATTERNS
    
    def _get_human_transitions(self) -> List[str]:
        """Get natural human transition phrases."""
        return [
            "Also,", "Plus,", "And,", "But,", "So ",
            "Though,", "Still,", "Yet,", "Even so,",
            "On top of that,", "Besides,", "Anyway,",
            "That said,", "Having said that,", "Then again,",
            "Looking at it differently,", "In practice,",
            "What this means is", "Here's the thing:",
            "The way I see it,", "From my experience,"
        ]
    
    def _get_contraction_map(self) -> Dict[str, str]:
        """Get common contractions for natural writing."""
        return {
            "I am": "I'm", "I have": "I've", "I will": "I'll",
            "I would": "I'd", "do not": "don't", "does not": "doesn't",
            "did not": "didn't", "cannot": "can't", "could not": "couldn't",
            "would not": "wouldn't", "should not": "shouldn't",
            "will not": "won't", "it is": "it's", "that is": "that's",
            "there is": "there's", "we are": "we're", "they are": "they're",
            "you are": "you're", "we have": "we've", "they have": "they've",
            "you have": "you've", "we will": "we'll", "they will": "they'll",
            "you will": "you'll", "is not": "isn't", "are not": "aren't",
            "was not": "wasn't", "were not": "weren't", "has not": "hasn't",
            "have not": "haven't", "had not": "hadn't", "let us": "let's",
            "who is": "who's", "what is": "what's", "where is": "where's",
            "how is": "how's", "going to": "gonna", "want to": "wanna",
            "got to": "gotta", "kind of": "kinda", "sort of": "sorta"
        }
    
    def detect_patterns(self, text: str) -> List[str]:
        """Detect AI writing patterns in text."""
        found_patterns = []
        text_lower = text.lower()
        
        for phrase in self.patterns["overused_phrases"]:
            if phrase.lower() in text_lower:
                found_patterns.append(f"overused_phrase: {phrase}")
        
        for vocab in self.patterns["vocabulary_tells"]:
            if vocab.lower() in text_lower:
                found_patterns.append(f"ai_vocabulary: {vocab}")
        
        for pattern in self.patterns["sentence_patterns"]:
            matches = re.findall(pattern, text)
            if matches:
                found_patterns.append(f"pattern_match: {pattern[:30]}...")
        
        sentences = re.split(r'[.!?]+', text)
        for i, sent in enumerate(sentences):
            if sent.strip() and i < len(sentences) - 1:
                next_sent = sentences[i + 1].strip() if i + 1 < len(sentences) else ""
                if next_sent and len(sent.strip().split()) > 2:
                    if sent.strip().split()[0] in ["Furthermore", "Moreover", "Additionally", "Consequently"]:
                        found_patterns.append(f"transition_start: {sent.strip()[:40]}...")
        
        if text.count(";") > 3:
            found_patterns.append("excessive_semicolons")
        
        if not any(c in text for c in ["'", '"', "n't", "'re", "'ve", "'ll", "'d"]):
            found_patterns.append("no_contractions")
        
        return found_patterns
    
    def add_contractions(self, text: str) -> str:
        """Add contractions to make text more natural."""
        result = text
        for full, contraction in self.contraction_map.items():
            result = re.sub(r'\b' + re.escape(full) + r'\b', contraction, result, flags=re.IGNORECASE)
        return result
    
    def vary_sentence_starts(self, text: str) -> str:
        """Vary sentence beginnings to avoid patterns."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        varied = []
        
        for i, sent in enumerate(sentences):
            if i > 0 and random.random() < 0.3:
                words = sent.split()
                if words and words[0] in ["Furthermore", "Moreover", "Additionally", "Consequently"]:
                    transition = random.choice(self.human_transitions)
                    words[0] = transition.rstrip(", ")
                    sent = " ".join(words)
            varied.append(sent)
        
        return " ".join(varied)
    
    def add_natural_imperfections(self, text: str) -> str:
        """Add subtle natural imperfections."""
        if self.level == "low":
            return text
        
        lines = text.split("\n")
        result = []
        
        for line in lines:
            if random.random() < 0.05 and self.level == "high":
                words = line.split()
                if len(words) > 5:
                    idx = random.randint(1, len(words) - 2)
                    words.insert(idx, random.choice(["really", "quite", "basically", "honestly"]))
                    line = " ".join(words)
            result.append(line)
        
        return "\n".join(result)
    
    def humanize(self, text: str) -> HumanizationResult:
        """Main humanization function."""
        patterns_found = self.detect_patterns(text)
        changes_made = []
        humanized = text
        
        if "no_contractions" in patterns_found or self.level in ["medium", "high"]:
            humanized = self.add_contractions(humanized)
            changes_made.append("Added contractions")
        
        if any("transition" in p for p in patterns_found) or self.level == "high":
            humanized = self.vary_sentence_starts(humanized)
            changes_made.append("Varied sentence starts")
        
        if self.level == "high":
            humanized = self.add_natural_imperfections(humanized)
            changes_made.append("Added natural variations")
        
        for phrase in self.patterns["overused_phrases"]:
            if phrase.lower() in humanized.lower():
                alternatives = self._get_alternatives(phrase)
                if alternatives:
                    replacement = random.choice(alternatives)
                    humanized = re.sub(re.escape(phrase), replacement, humanized, flags=re.IGNORECASE)
                    changes_made.append(f"Replaced '{phrase}' with '{replacement}'")
        
        confidence = 1.0 - (len(patterns_found) * 0.1)
        confidence = max(0.3, min(1.0, confidence))
        
        return HumanizationResult(
            original=text,
            humanized=humanized,
            patterns_found=patterns_found,
            changes_made=changes_made,
            confidence_score=confidence
        )
    
    def _get_alternatives(self, phrase: str) -> List[str]:
        """Get human-friendly alternatives for AI phrases."""
        alternatives_map = {
            "in conclusion": ["to wrap up", "looking at the big picture", "bottom line"],
            "it is important to note": ["worth mentioning", "keep in mind", "just so you know"],
            "furthermore": ["also", "plus", "on top of that", "and"],
            "moreover": ["also", "besides", "what's more", "plus"],
            "in addition": ["also", "plus", "and", "along with that"],
            "significantly": ["a lot", "really", "quite a bit", "markedly"],
            "notably": ["especially", "particularly", "in particular", "standout"],
            "crucial": ["key", "important", "vital", "essential"],
            "utilize": ["use", "put to use", "make use of"],
            "facilitate": ["help", "make easier", "enable"],
            "endeavor": ["try", "attempt", "aim", "strive"],
            "commence": ["start", "begin", "kick off"],
            "approximately": ["about", "around", "roughly", "more or less"],
            "regarding": ["about", "on", "concerning", "for"],
            "pertaining to": ["about", "related to", "for"],
            "in terms of": ["for", "when it comes to", "as far as"],
            "with respect to": ["for", "about", "regarding"],
            "substantially": ["a lot", "significantly", "quite a bit"],
            "comprehensive": ["thorough", "full", "complete", "detailed"],
            "multifaceted": ["complex", "varied", "diverse", "many-sided"],
            "myriad": ["many", "lots of", "plenty of", "tons of"],
            "plethora": ["lots of", "plenty of", "a ton of", "many"],
            "delve": ["explore", "dig into", "look into", "examine"],
            "embark": ["start", "begin", "set out", "kick off"],
            "landscape": ["field", "area", "scene", "world"],
            "realm": ["area", "field", "domain", "space"],
            "tapestry": ["mix", "blend", "combination", "collection"],
            "underscores": ["shows", "highlights", "emphasizes", "points out"],
            "pivotal": ["key", "crucial", "critical", "important"],
            "nuanced": ["subtle", "complex", "detailed", "layered"],
            "holistic": ["complete", "full", "overall", "broad"],
            "synergy": ["teamwork", "collaboration", "working together"],
            "leverage": ["use", "take advantage of", "build on"],
            "foster": ["encourage", "support", "nurture", "promote"],
            "nurture": ["support", "develop", "grow", "build"],
            "spearhead": ["lead", "drive", "head", "champion"],
            "revolutionize": ["change", "transform", "shake up", "reinvent"]
        }
        
        return alternatives_map.get(phrase.lower(), [])
    
    def analyze_wikipedia_style(self) -> Dict:
        """Analyze Wikipedia academic writing style for human patterns."""
        return {
            "characteristics": [
                "Uses specific examples and citations",
                "Varied sentence lengths",
                "Mix of formal and accessible language",
                "Natural flow between ideas",
                "Occasional casual transitions",
                "Specific over vague language",
                "Active voice preferred"
            ],
            "avoid": [
                "Overly perfect structure",
                "Consistent paragraph lengths",
                "Every sentence starting with transition words",
                "Using rare/sophisticated words when simple ones work",
                "Perfect grammar in every clause"
            ],
            "emulate": [
                "Natural thought progression",
                "Personal voice and perspective",
                "Specific details over generalizations",
                "Varied vocabulary without forced sophistication",
                "Conversational yet professional tone"
            ]
        }
    
    def create_humanization_plan(self) -> Dict:
        """Create a plan to make writing more human-like."""
        wiki_style = self.analyze_wikipedia_style()
        
        return {
            "phase_1_analysis": {
                "description": "Analyze AI patterns in generated text",
                "action": "Run detect_patterns() on each generation"
            },
            "phase_2_avoidance": {
                "description": "Apply avoidance rules during generation",
                "rules": [
                    "Use contractions (don't, can't, I'm)",
                    "Vary sentence length (mix short and long)",
                    "Use casual transitions (Also, But, So)",
                    "Include specific examples",
                    "Add personal perspective",
                    "Avoid overused AI vocabulary"
                ]
            },
            "phase_3_refinement": {
                "description": "Post-generation humanization",
                "action": "Apply humanize() with appropriate level"
            },
            "phase_4_verification": {
                "description": "Verify humanization score",
                "action": "Check confidence_score > 0.7"
            }
        }
