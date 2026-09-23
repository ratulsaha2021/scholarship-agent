"""Humanizer: detects AI-writing patterns and fixes them.

Patterns follow Wikipedia's "Signs of AI writing" (WikiProject AI Cleanup), tuned for
professional academic writing: application emails, statements of purpose and proposals.

The aim is text that reads like a careful human applicant wrote it: plain, specific and
professional. That means removing staging, inflation and filler, not adding slang or typos.

Three layers:
- detect(): rule-based detector, returns Issues (strong ones justify a fix on one sighting)
- clean(): safe deterministic fixes (dashes, curly quotes, bold, emoji, chatbot residue)
- rewrite_prompt(): a targeted LLM rewrite prompt listing exactly what was found
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_DIR = Path(__file__).parent.parent / "config"

KINDS = ("email", "sop", "proposal")

# Rules given to the model when drafting. Kept short: long rule lists get ignored.
STYLE_GUIDE = """Write like a careful human applicant, not a chatbot:
- Plain, specific, professional. Every sentence adds a fact the reader didn't have.
- State points directly. No "not X but Y" contrasts, no one-line dramatic closers, no rhetorical questions.
- No em or en dashes. Use commas, colons, periods or parentheses instead.
- Don't group things in threes for rhythm; list only what's real.
- Avoid: delve, tapestry, testament, pivotal, crucial, showcase, underscore, foster, vibrant, intricate, meticulous, robust (figurative), landscape (abstract), realm, embark, additionally, leverage, seamless, passionate, esteemed, cutting-edge, groundbreaking, renowned.
- No inflated significance ("a pivotal moment", "plays a crucial role", "a testament to", "setting the stage for").
- Prefer "is/are/has" over "serves as/stands as/boasts".
- No flattery of the reader, no "I hope this email finds you well", no "I am writing to express".
- Vary sentence length. Active voice. Use "I" naturally, but don't start several sentences in a row with it.
- No bold, no emojis, no bracketed placeholders like [Name]."""


@dataclass
class Issue:
    id: str
    label: str
    strong: bool
    advice: str
    examples: List[str] = field(default_factory=list)

    def describe(self) -> str:
        ex = "; ".join(f'"{e}"' for e in self.examples[:3])
        return f"{self.label}{': ' + ex if ex else ''}. Fix: {self.advice}"


@dataclass
class HumanizationResult:
    original: str
    humanized: str
    patterns_found: List[str]
    changes_made: List[str]
    confidence_score: float  # 0-1, higher = fewer AI tells


def _p(*phrases: str) -> List[str]:
    """Whole-word, case-insensitive regexes for literal phrases."""
    return [r"\b" + re.escape(ph).replace(r"\ ", r"\s+") + r"\b" for ph in phrases]


# id, label, strong, advice, regexes
PHRASE_PATTERNS = [
    ("not_x_but_y", "Not-X-but-Y contrast", True,
     "state the point directly; keep a contrast only if the reader actually believes the negative half",
     [r"\bnot\s+(?:just|only|merely|simply)\b[^.!?\n]{0,80}?\bbut\b",
      r"\bit'?s\s+not\b[^.!?\n]{0,60}?[,;]\s*it'?s\b",
      r"\bisn'?t\s+(?:just|only|merely)\b",
      r"\b(?:does\s+not|doesn'?t)\s+mean\b[^.!?\n]*[.!?]\s+It\s+means\b"]),
    ("closer", "Dramatic closer", True,
     "cut lines that only ask the reader to pause; end on the last concrete fact",
     _p("let that sink in", "read that again", "that is the real win", "that's the real win",
        "and that makes all the difference", "and that is what matters", "the rest is history")),
    ("deep_saying", "Saying that sounds deep", True,
     "replace with the specific claim",
     _p("the real question is", "at its core", "what really matters", "the heart of the matter",
        "the deeper issue", "at the end of the day")),
    ("run_up", "Staged run-up", True,
     "remove the announcement and make the point",
     _p("let's dive in", "let us dive in", "let's explore", "let's break this down",
        "here's what you need to know", "without further ado", "here's the thing",
        "the thing is", "let's be honest", "real talk")),
    ("no_one", "Arguing with no one", True,
     "remove the defence against an objection nobody raised",
     _p("to be clear", "don't get me wrong", "i'm not saying", "this is not to say",
        "one might be tempted", "a tempting approach would be")),
    ("chat_residue", "Chatbot residue", True,
     "remove the wrapper and keep the content",
     _p("i hope this helps", "great question", "let me know if you'd like",
        "would you like me to", "here is the revised", "here is the rewritten",
        "here's the revised", "here's a revised", "here is your", "certainly!", "of course!")),
    ("disclaimer", "Knowledge-limit disclaimer", True,
     "state what the source doesn't show, or cut the sentence",
     _p("as of my last", "based on available information", "not publicly available",
        "while specific details", "up to my last training")),
    ("email_cliche", "Stock email opener", True,
     "open with the position and why you fit it",
     _p("i hope this email finds you well", "i hope this message finds you well",
        "i hope you are doing well", "i am writing to express", "i'm writing to express",
        "i am writing to inquire", "i am reaching out to express")),
    ("inflation", "Inflated significance", True,
     "keep the fact and drop the significance",
     _p("stands as a testament", "a testament to", "pivotal moment", "pivotal role",
        "plays a crucial role", "plays a key role", "plays a vital role", "plays a pivotal role",
        "setting the stage for", "evolving landscape", "ever-evolving", "indelible mark",
        "the future looks bright", "exciting times ahead", "enduring legacy", "rich tapestry",
        "a step in the right direction", "reflects a broader")),
    ("sales", "Sales language", True,
     "say what the thing is, without advertising it",
     _p("nestled", "breathtaking", "in the heart of", "groundbreaking", "world-class",
        "cutting-edge", "state-of-the-art", "stunning", "must-visit", "boasts", "renowned",
        "esteemed", "prestigious")),
    ("flattery", "Flattery", True,
     "one specific reason for applying beats praise",
     _p("your esteemed", "your prestigious", "your renowned", "i have long admired",
        "i am deeply impressed", "your groundbreaking")),
    ("authority", "Borrowed authority", False,
     "name the source or cut the claim",
     _p("experts argue", "experts believe", "experts say", "studies show", "research shows",
        "observers have noted")),
    ("copula", "Avoiding is/are/has", False,
     "use is, are or has",
     _p("serves as", "stands as", "functions as")),
    ("qualifiers", "Stacked qualifiers", False,
     "keep one qualifier only where the doubt is real",
     _p("could potentially", "might arguably", "it's also possible", "could possibly",
        "may potentially")),
    ("overclaim", "Stock applicant claims", False,
     "show it with a specific fact instead of asserting it",
     _p("passionate about", "deeply passionate", "i am confident that", "perfect fit",
        "ideal candidate", "unique opportunity", "i believe that", "strong foundation")),
]

AI_WORDS = [
    "delve", "delves", "delving", "tapestry", "testament", "underscore", "underscores",
    "underscored", "underscoring", "showcase", "showcases", "showcased", "showcasing",
    "pivotal", "intricate", "intricacies", "meticulous", "meticulously", "foster", "fostering",
    "fostered", "garner", "garnered", "bolster", "bolstered", "vibrant", "crucial", "enhance",
    "enhancing", "interplay", "landscape", "realm", "embark", "embarked", "additionally",
    "seamless", "seamlessly", "leverage", "leveraging", "synergy", "holistic", "multifaceted",
    "paramount", "invaluable", "commendable", "profound", "enduring", "align with", "aligns with",
    "deep dive",
]

ING_RIDERS = r",\s+(?:highlighting|underscoring|emphasizing|showcasing|reflecting|symbolizing|fostering|cultivating|ensuring|contributing\s+to|encompassing)\b"
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
PLACEHOLDER = re.compile(r"\[[^\]\n]{1,60}\]")
ALLOWED_PLACEHOLDERS = {"[citation needed]"}


def _load_extra_words() -> List[str]:
    """Extra AI words from config/ai_patterns.json ("ai_words": [...]) so users can extend the list."""
    path = CONFIG_DIR / "ai_patterns.json"
    try:
        return json.loads(path.read_text()).get("ai_words", [])
    except (OSError, ValueError):
        return []


def _snippet(text: str, match: re.Match, width: int = 40) -> str:
    start = max(0, match.start() - 10)
    return re.sub(r"\s+", " ", text[start:match.end() + 10]).strip()[:width + 20]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


class Humanizer:
    def __init__(self, level: str = "high"):
        self.level = level  # kept for config compatibility
        self.ai_words = sorted(set(AI_WORDS + [w.lower() for w in _load_extra_words()]))

    # ---------- detection ----------

    def detect(self, text: str, kind: str = "email") -> List[Issue]:
        issues: List[Issue] = []

        for pid, label, strong, advice, regexes in PHRASE_PATTERNS:
            examples = []
            for rx in regexes:
                for m in re.finditer(rx, text, re.IGNORECASE):
                    examples.append(_snippet(text, m))
            if examples:
                issues.append(Issue(pid, label, strong, advice, examples))

        found_words = []
        for w in self.ai_words:
            if re.search(r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\b", text, re.IGNORECASE):
                found_words.append(w)
        if found_words:
            issues.append(Issue("ai_words", "Overused AI vocabulary", len(found_words) >= 2,
                                "use the plain word (use, show, important, key → main, etc.)",
                                found_words))

        dashes = re.findall(r"(?<!\d)\s*[—–]\s*(?!\d)|\s--\s", text)
        if dashes:
            issues.append(Issue("dashes", "Em/en dashes", len(dashes) >= 2,
                                "replace each dash with a comma, colon, period or parentheses",
                                [f"{len(dashes)} dash(es)"]))

        riders = [_snippet(text, m) for m in re.finditer(ING_RIDERS, text, re.IGNORECASE)]
        if riders:
            issues.append(Issue("ing_riders", "Shallow -ing riders", len(riders) >= 2,
                                "keep the fact, drop the ', highlighting ...' add-on", riders))

        triads = re.findall(r"\b\w+(?:\s\w+){0,2},\s\w+(?:\s\w+){0,2},?\s(?:and|or)\s\w+", text)
        if len(triads) >= 3:
            issues.append(Issue("triads", "Too many three-item lists", False,
                                "list only real items; vary the structure", triads[:3]))

        placeholders = [p for p in PLACEHOLDER.findall(text)
                        if not (kind == "proposal" and p.lower() in ALLOWED_PLACEHOLDERS)
                        and not re.fullmatch(r"\[\d+\]", p)]  # numeric citations are fine
        if placeholders:
            issues.append(Issue("placeholder", "Unfilled placeholder", True,
                                "replace with the real detail from the background, or rewrite the sentence without it",
                                placeholders))

        if kind == "sop" and re.search(r"^\s*(?:#{1,6}\s|[-*]\s|\d+[.)]\s)", text, re.MULTILINE):
            issues.append(Issue("sop_structure", "Headings or lists in an SOP", True,
                                "write it as flowing paragraphs and don't repeat the CV", []))
        if "**" in text or "__" in text:
            issues.append(Issue("bold", "Bold as decoration", kind == "email",
                                "remove the bold", []))
        if EMOJI.search(text):
            issues.append(Issue("emoji", "Emoji", True, "remove emojis", []))
        if re.search(r"[“”‘’]", text):
            issues.append(Issue("curly_quotes", "Curly quotes", False, "use straight quotes", []))

        heads = [h for h in re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
                 if len(h.split()) >= 3 and sum(w[0].isupper() for w in h.split() if w[0].isalpha()) >= 3]
        if heads:
            issues.append(Issue("title_case", "Title Case headings", False,
                                "use sentence case for headings", heads[:3]))

        sents = _sentences(text)
        runs = []
        for a, b, c in zip(sents, sents[1:], sents[2:]):
            words = [s.split()[0].lower().strip(",") for s in (a, b, c)]
            if words[0] == words[1] == words[2] and words[0] not in ("dear", "best", "-"):
                runs.append(words[0])
        if runs:
            issues.append(Issue("openings", "Repeated sentence openings", False,
                                "merge sentences or start with the action", sorted(set(runs))))

        return issues

    def detect_patterns(self, text: str) -> List[str]:
        """Labels only (used by the CLI review)."""
        return [i.label for i in self.detect(text)]

    @staticmethod
    def needs_rewrite(issues: List[Issue]) -> bool:
        return any(i.strong for i in issues) or sum(not i.strong for i in issues) >= 2

    @staticmethod
    def score(issues: List[Issue]) -> float:
        strong = sum(i.strong for i in issues)
        weak = len(issues) - strong
        return max(0.0, min(1.0, 1.0 - 0.15 * strong - 0.05 * weak))

    # ---------- deterministic fixes ----------

    def clean(self, text: str, kind: str = "email") -> str:
        t = text.strip()
        t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        t = re.sub(r"(\d)\s*[–—]\s*(\d)", r"\1-\2", t)          # number ranges keep a hyphen
        t = re.sub(r"\s*[—–]\s*|\s--\s", ", ", t)              # other dashes become commas
        t = re.sub(r",\s*([.,;:!?])", r"\1", t)                 # ", ." left over from a dash
        t = EMOJI.sub("", t)
        t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
        if kind == "email":
            t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)

        # Drop whole sentences that are pure chatbot residue or stock openers
        residue = re.compile(
            r"(?:^|(?<=[.!?\n]))[ \t]*(?:I hope (?:this|you)[^.!?\n]*(?:well|helps)[.!]|"
            r"(?:Certainly|Of course|Sure)!|Great question!|Let me know if you'?d like[^.!?\n]*[.!?]|"
            r"Here(?: is|'s) (?:the|a|your) (?:revised|rewritten|edited|updated)[^:\n]*:)[ \t]*",
            re.IGNORECASE)
        t = residue.sub("", t)

        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r" +([,.;:!?])", r"\1", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return "\n".join(line.rstrip() for line in t.splitlines()).strip()

    # ---------- LLM rewrite ----------

    def rewrite_prompt(self, text: str, issues: List[Issue], kind: str, context: str = "") -> str:
        what = {"email": "application email", "sop": "statement of purpose",
                "proposal": "research proposal"}[kind]
        found = "\n".join(f"- {i.describe()}" for i in issues) or "- (none flagged; do a light polish)"
        keep_format = {
            "email": "Keep the greeting and the signature exactly.",
            "sop": "Flowing prose paragraphs only: no headings, no lists, no CV-style sections.",
            "proposal": "Keep the section headings (sentence case, markdown '#'/'##') and lists.",
        }[kind]
        return f"""Edit this {what} so it reads like a careful human applicant wrote it.

Problems found:
{found}

Rules:
{STYLE_GUIDE}
- Keep every fact, name, number, paper, date and requirement. Don't add any new ones.
- Keep the professional register: no slang, no deliberate typos, no casual filler.
- {keep_format}
- Keep roughly the same length.

{"Applicant background (the only source of facts):" + chr(10) + context + chr(10) if context else ""}
Text:
{text}

Return ONLY the edited {what}, nothing else."""

    # ---------- no-LLM path ----------

    def humanize(self, text: str, kind: str = "email") -> HumanizationResult:
        """Deterministic cleanup plus a report (used when no LLM is available)."""
        before = self.detect(text, kind)
        cleaned = self.clean(text, kind)
        after = self.detect(cleaned, kind)
        fixed = sorted({i.label for i in before} - {i.label for i in after})
        return HumanizationResult(
            original=text, humanized=cleaned,
            patterns_found=[i.label for i in after],
            changes_made=[f"Fixed: {label}" for label in fixed],
            confidence_score=self.score(after),
        )

    def report(self, issues: List[Issue]) -> str:
        """One-line summary for the chat."""
        if not issues:
            return "Style check: no AI-writing patterns found."
        labels = ", ".join(i.label.lower() for i in issues[:4])
        return f"Style check: {len(issues)} possible pattern(s) left ({labels}). Ask me to fix them, or edit by hand."
