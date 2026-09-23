"""Hybrid LLM module: local model (Ollama) drafts, Groq polishes, rule-based humanizer checks."""

import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

from .humanizer import Humanizer, Issue, STYLE_GUIDE

# Appended to every prompt that writes or rewrites application text
NO_FABRICATION_RULE = (
    "Use ONLY facts stated in the applicant's background or the original text. "
    "Never invent projects, employers, results, numbers, papers or skills. "
    "If a detail isn't given, leave it out."
)

WRITER_ROLES = {
    "email": "You write short, professional application emails for PhD and scholarship applicants.",
    "sop": "You write statements of purpose for graduate applicants.",
    "proposal": "You write research proposals for PhD applicants.",
}
MAX_REWRITES = 2


def _norm(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _appears_in(claim: str, text: str) -> bool:
    """Checkers sometimes 'quote' the sources instead of the document; keep only real quotes."""
    words, doc = _norm(claim), set(_norm(text))
    if not words:
        return False
    return sum(w in doc for w in words) / len(words) >= 0.8

@dataclass
class LLMConfig:
    # Local model (Llama 3.1 8B via Ollama)
    local_model: str = "llama3.1:8b"
    local_base_url: str = "http://localhost:11434"
    
    # Groq API
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    
    @classmethod
    def load(cls) -> "LLMConfig":
        config_file = Path(__file__).parent.parent / "config" / "llm_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                data = json.load(f)
            config = cls(**data)
            if not config.groq_api_key:
                config.groq_api_key = os.environ.get("GROQ_API_KEY", "")
            return config
        
        config = cls()
        config.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        return config
    
    def save(self):
        config_file = Path(__file__).parent.parent / "config" / "llm_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump({
                "local_model": self.local_model,
                "local_base_url": self.local_base_url,
                "groq_api_key": self.groq_api_key,
                "groq_model": self.groq_model
            }, f, indent=2)

class HybridLLM:
    """Drafts with the local model (or Groq), then checks and rewrites against the humanizer rules."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.load()
        self.humanizer = Humanizer()
        self._local_available = None
        self._groq_client = None
    
    def _check_local_available(self) -> bool:
        """Check if Ollama is running with Llama model."""
        if self._local_available is not None:
            return self._local_available
        
        try:
            import requests
            response = requests.get(f"{self.config.local_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self._local_available = any(
                    self.config.local_model in m.get("name", "") for m in models
                )
                return self._local_available
        except Exception:
            pass
        
        self._local_available = False
        return False
    
    def _get_groq_client(self):
        """Get Groq client."""
        if self._groq_client is None:
            if not self.config.groq_api_key:
                return None
            
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.config.groq_api_key)
            except Exception:
                return None
        
        return self._groq_client
    
    def _call_local(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        """Call local Ollama model."""
        import requests
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.config.local_base_url}/api/chat",
            json={
                "model": self.config.local_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_tokens
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def _call_groq(self, prompt: str, system: str = "", max_tokens: int = 2048, reasoning: str = "low") -> str:
        """Call Groq API."""
        client = self._get_groq_client()
        if not client:
            raise Exception("Groq API not configured")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        if "gpt-oss" in self.config.groq_model:
            # Reasoning tokens count against the limit; keep reasoning short so the text isn't cut off
            kwargs = {"reasoning_effort": reasoning, "include_reasoning": False}
        response = client.chat.completions.create(
            model=self.config.groq_model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=0.7,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise Exception(f"Groq returned no text (finish reason: {response.choices[0].finish_reason})")
        return content

    def _drafting_system(self, kind: str, context: str) -> str:
        return f"""{WRITER_ROLES[kind]}

{STYLE_GUIDE}
- {NO_FABRICATION_RULE}

Applicant background:
{context}"""

    def _draft(self, prompt: str, system: str, max_tokens: int, result: Dict) -> str:
        if self._check_local_available():
            result["local_used"] = True
            return self._call_local(prompt, system, max_tokens)
        if self.config.groq_api_key:
            result["groq_used"] = True
            return self._call_groq(prompt, system, max_tokens)
        raise Exception("No LLM available (Ollama model not found and no Groq key).")

    def _rewrite(self, prompt: str, max_tokens: int, result: Dict) -> str:
        # Groq is usually the stronger editor; fall back to the local model
        if self.config.groq_api_key:
            result["groq_used"] = True
            return self._call_groq(prompt, max_tokens=max_tokens)
        result["local_used"] = True
        return self._call_local(prompt, max_tokens=max_tokens)

    def polish(self, text: str, kind: str = "email", context: str = "",
               max_tokens: int = 2048, result: Optional[Dict] = None) -> Dict:
        """Clean deterministically, then rewrite with the LLM until no strong AI patterns remain."""
        result = result if result is not None else {"local_used": False, "groq_used": False}
        text = self.humanizer.clean(text, kind)
        issues = self.humanizer.detect(text, kind)
        rewrites = 0
        # Long documents: rewrite only for strong tells (each pass regenerates the whole document)
        max_rewrites = MAX_REWRITES if kind == "email" else 1
        needs = (lambda iss: self.humanizer.needs_rewrite(iss)) if kind == "email" else \
                (lambda iss: any(i.strong for i in iss))
        while needs(issues) and rewrites < max_rewrites and self.is_available():
            prompt = self.humanizer.rewrite_prompt(text, issues, kind, context)
            try:
                candidate = self.humanizer.clean(self._rewrite(prompt, max_tokens, result), kind)
            except Exception:
                break
            rewrites += 1
            new_issues = self.humanizer.detect(candidate, kind)
            # Guard against a rewrite that drops most of the content
            if len(candidate.split()) < 0.6 * len(text.split()):
                continue
            text, issues = candidate, new_issues

        result.update({
            "humanized": text,
            "issues": issues,
            "rewrites": rewrites,
            "ai_analysis": {
                "ai_score": round(100 * (1 - self.humanizer.score(issues))),
                "patterns_found": [i.label for i in issues],
                "suggestions": [i.advice for i in issues],
            },
        })
        return result

    def fact_check(self, text: str, sources: str, result: Optional[Dict] = None) -> List[Dict]:
        """Ask the model to list claims in `text` that the sources don't support. Returns [] if clean
        or if no model is available."""
        result = result if result is not None else {}
        prompt = f"""You are fact-checking an application document against its sources.

SOURCES (the only ground truth: the applicant's background, the job post, the applicant's notes):
{sources}

DOCUMENT:
{text}

List every statement in the DOCUMENT that is not supported by the SOURCES or contradicts them. Check:
- degrees, thesis level (undergraduate vs master's), institutions, dates, grades
- paper titles, venues, co-authorship, results and numbers, dataset and tool names (exact names)
- roles, employers, projects and skills
- claims about the program, lab, supervisor or resources that the post doesn't state
Plans, research questions and proposed methods are fine; they're not claims of fact.

Return ONLY JSON: {{"problems": [{{"claim": "<exact words from the document>", "issue": "<what the sources say instead>"}}]}}
Return {{"problems": []}} if everything is supported."""
        try:
            if self.config.groq_api_key:
                # Checking needs more care than writing
                result["groq_used"] = True
                raw = self._call_groq(prompt, max_tokens=6000, reasoning="medium")
            else:
                raw = self._rewrite(prompt, 2000, result)
        except Exception:
            return []
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            problems = json.loads(match.group()).get("problems", [])
        except (ValueError, AttributeError):
            return []
        return [p for p in problems
                if isinstance(p, dict) and p.get("claim") and _appears_in(p["claim"], text)]

    def fix_facts(self, text: str, problems: List[Dict], kind: str, sources: str,
                  max_tokens: int, result: Dict) -> str:
        listed = "\n".join(f'- "{p["claim"]}": {p.get("issue", "not supported")}' for p in problems)
        prompt = f"""Correct these factual problems in the document. Fix each one using the sources, or remove
the claim if the sources don't support it. Change nothing else.

Problems:
{listed}

Sources:
{sources}

Document:
{text}

Return ONLY the corrected document."""
        return self.humanizer.clean(self._rewrite(prompt, max_tokens, result), kind)

    def hybrid_generate(self, prompt: str, context: str = "", kind: str = "email",
                        max_tokens: int = 2048) -> Dict:
        """
        1. Draft with the local model (Groq if it isn't available), with the style guide as system prompt
        2. Rule-based humanizer check (no LLM call)
        3. Targeted rewrite of what the check found (Groq preferred), repeated at most twice
        4. Fact check against the background + post; fix what's unsupported, then check once more
        """
        result = {"draft": "", "humanized": "", "issues": [], "ai_analysis": {}, "fact_fixes": [],
                  "fact_problems": [], "local_used": False, "groq_used": False, "fallback": False,
                  "error": "", "rewrites": 0}
        try:
            result["draft"] = self._draft(prompt, self._drafting_system(kind, context), max_tokens, result)
        except Exception as e:
            result.update({"fallback": True, "error": str(e), "draft": "", "humanized": ""})
            return result
        self.polish(result["draft"], kind, context, max_tokens, result)

        # The prompt holds the post and the applicant's notes, so it's part of the ground truth
        sources = f"{context}\n\nREQUEST (post and applicant notes):\n{prompt}"
        problems = self.fact_check(result["humanized"], sources, result)
        if problems:
            try:
                fixed = self.fix_facts(result["humanized"], problems, kind, sources, max_tokens, result)
                if len(fixed.split()) >= 0.6 * len(result["humanized"].split()):
                    result["fact_fixes"] = problems
                    self.polish(fixed, kind, context, max_tokens, result)
                    # Second check only for long documents; emails are short and this halves their wait
                    result["fact_problems"] = (self.fact_check(result["humanized"], sources, result)
                                               if kind != "email" else [])
            except Exception:
                result["fact_problems"] = problems
        return result

    def is_available(self) -> bool:
        return bool(self.config.groq_api_key) or self._check_local_available()

    def generate(self, prompt: str, system: str = "", use_groq: bool = False, max_tokens: int = 2048) -> str:
        """General generation."""
        if use_groq and self.config.groq_api_key:
            return self._call_groq(prompt, system, max_tokens)
        elif self._check_local_available():
            return self._call_local(prompt, system, max_tokens)
        elif self.config.groq_api_key:
            return self._call_groq(prompt, system, max_tokens)
        else:
            return "No LLM available."

    def get_status(self) -> Dict:
        """Get status of both models."""
        return {
            "local_available": self._check_local_available(),
            "local_model": self.config.local_model,
            "groq_configured": bool(self.config.groq_api_key),
            "groq_model": self.config.groq_model
        }
