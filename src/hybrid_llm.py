"""Hybrid LLM module - routes between local and API models."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    LOCAL = "local"
    GROQ = "groq"

@dataclass
class LLMConfig:
    # Local model (Qwen 2.5 7B via Ollama)
    local_model: str = "qwen2.5:7b"
    local_base_url: str = "http://localhost:11434"
    
    # Groq API
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    
    # Routing
    use_local_for_draft: bool = True
    use_groq_for_humanize: bool = True
    
    @classmethod
    def load(cls) -> "LLMConfig":
        config_file = Path(__file__).parent.parent / "config" / "llm_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                data = json.load(f)
            return cls(**data)
        
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
                "groq_model": self.groq_model,
                "use_local_for_draft": self.use_local_for_draft,
                "use_groq_for_humanize": self.use_groq_for_humanize
            }, f, indent=2)

class HybridLLM:
    """Routes between local and API models."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.load()
        self._local_available = None
        self._groq_client = None
    
    def _check_local_available(self) -> bool:
        """Check if Ollama is running with Qwen model."""
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
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def _call_groq(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        """Call Groq API."""
        client = self._get_groq_client()
        if not client:
            raise Exception("Groq API not configured")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.config.groq_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def draft_email(self, prompt: str, context: str = "") -> str:
        """Draft initial email using local model."""
        system = f"""You are a professional academic email writer. Write concise, specific, 
human-like emails for scholarship/PhD applications. Be direct, avoid AI patterns.

{context}

Rules:
- Use contractions (I'm, don't, can't)
- Vary sentence length
- Be specific, not generic
- No "I am writing to express..." openings
- Reference specific papers/projects when possible
- Keep under 300 words"""
        
        if self.config.use_local_for_draft and self._check_local_available():
            return self._call_local(prompt, system)
        elif self.config.groq_api_key:
            return self._call_groq(prompt, system)
        else:
            return self._fallback_draft(prompt)
    
    def check_ai_patterns(self, text: str) -> Dict:
        """Check text for AI patterns using local model."""
        prompt = f"""Analyze this email for AI writing patterns. Return JSON with:
- "ai_score": 0-100 (higher = more AI-like)
- "patterns_found": list of specific AI patterns found
- "suggestions": list of specific fixes

Email:
{text}"""
        
        system = """You are an AI detection expert. Analyze text for common AI writing patterns:
- Overused phrases (furthermore, moreover, crucial, leverage, etc.)
- Perfect paragraph structure
- No contractions
- Listing with semicolons
- Generic statements
- Lack of specific examples
- Passive voice overuse

Return ONLY valid JSON."""
        
        try:
            if self.config.use_local_for_draft and self._check_local_available():
                response = self._call_local(prompt, system, max_tokens=1000)
            elif self.config.groq_api_key:
                response = self._call_groq(prompt, system, max_tokens=1000)
            else:
                return {"ai_score": 50, "patterns_found": [], "suggestions": []}
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"ai_score": 50, "patterns_found": [], "suggestions": []}
        
        except Exception:
            return {"ai_score": 50, "patterns_found": [], "suggestions": []}
    
    def humanize_text(self, text: str, ai_analysis: Dict) -> str:
        """Final humanization pass using Groq API."""
        patterns = ai_analysis.get("patterns_found", [])
        suggestions = ai_analysis.get("suggestions", [])
        
        prompt = f"""Rewrite this email to sound more human and natural.

Original:
{text}

Issues found:
{json.dumps(patterns, indent=2)}

Suggestions:
{json.dumps(suggestions, indent=2)}

Rewrite rules:
1. Add contractions (I'm, don't, can't, won't)
2. Vary sentence lengths (mix short and long)
3. Use casual transitions (Also, But, So, Plus)
4. Add specific details/examples
5. Include natural imperfections
6. Sound like a real person, not a template
7. Keep the core message but make it flow naturally
8. Remove any remaining AI-sounding phrases

Return ONLY the rewritten email text, no explanations."""
        
        system = """You are an expert at making AI-generated text sound human. 
Your goal is to make the text undetectable by AI detection tools while maintaining 
professional quality. Be creative and natural."""
        
        try:
            if self.config.groq_api_key:
                return self._call_groq(prompt, system)
            elif self._check_local_available():
                return self._call_local(prompt, system)
            else:
                return self._fallback_humanize(text)
        except Exception:
            return text
    
    def generate(self, prompt: str, system: str = "", use_groq: bool = False) -> str:
        """General generation endpoint."""
        if use_groq and self.config.groq_api_key:
            return self._call_groq(prompt, system)
        elif self._check_local_available():
            return self._call_local(prompt, system)
        elif self.config.groq_api_key:
            return self._call_groq(prompt, system)
        else:
            return "No LLM available. Please configure Groq API or install Ollama."
    
    def _fallback_draft(self, prompt: str) -> str:
        """Fallback when no LLM available."""
        return """Dear Professor,

I hope this email finds you well. I am writing to express my interest in potential 
research opportunities in your group.

I have a background in [Your Field] and am particularly interested in [Specific Topic]. 
I believe my skills align well with your research focus.

I would welcome the opportunity to discuss potential openings or collaborations.

Best regards,
[Your Name]"""
    
    def _fallback_humanize(self, text: str) -> str:
        """Fallback humanization without LLM."""
        import re
        
        result = text
        
        # Add contractions
        contractions = {
            "I am": "I'm", "I have": "I've", "I will": "I'll",
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "cannot": "can't", "would not": "wouldn't", "should not": "shouldn't",
            "it is": "it's", "that is": "that's", "there is": "there's"
        }
        
        for full, contraction in contractions.items():
            result = re.sub(r'\b' + full + r'\b', contraction, result, flags=re.IGNORECASE)
        
        return result
    
    def get_status(self) -> Dict:
        """Get status of both models."""
        return {
            "local_available": self._check_local_available(),
            "local_model": self.config.local_model,
            "groq_configured": bool(self.config.groq_api_key),
            "groq_model": self.config.groq_model
        }
