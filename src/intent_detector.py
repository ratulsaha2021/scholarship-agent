"""Intent detector - understands what the user wants to do."""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class Intent(Enum):
    CV_UPLOAD = "cv_upload"
    CV_EXTRACT = "cv_extract"
    ADD_POST = "add_post"
    ADD_POST_IMAGE = "add_post_image"
    SEARCH_POSTS = "search_posts"
    WRITE_EMAIL = "write_email"
    APPLY_SCHOLARSHIP = "apply_scholarship"
    APPLY_PHD = "apply_phd"
    SEND_EMAIL = "send_email"
    SETUP_SMTP = "setup_smtp"
    SHOW_SAVED = "show_saved"
    HELP = "help"
    GENERAL_CHAT = "general_chat"
    PROFILE_UPDATE = "profile_update"

@dataclass
class DetectedIntent:
    intent: Intent
    confidence: float
    entities: Dict
    raw_text: str

class IntentDetector:
    """Detects user intent from natural language."""
    
    def __init__(self):
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[Intent, List[str]]:
        return {
            Intent.CV_UPLOAD: [
                r'upload\s*(?:my\s*)?cv',
                r'upload\s*(?:my\s*)?(?:resume|cv|简历)',
                r'(?:parse|extract|read)\s*(?:my\s*)?cv',
                r'(?:parse|extract|read)\s*(?:my\s*)?(?:resume|cv)',
                r'my\s*cv\s*(?:is|here)',
                r'(?:here\s*is|this\s*is)\s*my\s*(?:cv|resume)',
            ],
            Intent.CV_EXTRACT: [
                r'extract\s*(?:from|my)\s*cv',
                r'what\s*(?:is|does)\s*(?:my\s*)?cv\s*(?:say|contain|have)',
                r'(?:show|tell)\s*me\s*(?:my\s*)?cv\s*(?:data|info|details)',
                r'(?:analyze|check)\s*(?:my\s*)?cv',
            ],
            Intent.ADD_POST: [
                r'add\s*(?:a\s*)?(?:scholarship|position|opportunity|post)',
                r'(?:new|another)\s*(?:scholarship|position|opportunity)',
                r'(?:i\s*found|there\s*is)\s*(?:a\s*)?(?:scholarship|position)',
                r'(?:save|store)\s*(?:this|the)\s*(?:post|position|opportunity)',
                r'(?:paste|add)\s*(?:this\s*)?(?:text|announcement)',
                r'here\s*(?:is|\'s)\s*(?:a\s*)?(?:post|scholarship|position)',
            ],
            Intent.ADD_POST_IMAGE: [
                r'(?:upload|scan|read)\s*(?:this\s*)?(?:image|photo|screenshot)',
                r'(?:ocr|extract)\s*(?:from\s*)?(?:image|photo|picture)',
                r'i\s*(?:have|took)\s*(?:a\s*)?(?:photo|screenshot)\s*(?:of|for)',
                r'(?:this\s*)?(?:image|photo|picture)\s*(?:has|contains|shows)',
            ],
            Intent.SEARCH_POSTS: [
                r'(?:search|find|look)\s*(?:for\s*)?(?:saved\s*)?(?:posts?|scholarships?|positions?)',
                r'(?:what|which)\s*(?:posts?|scholarships?|positions?)\s*(?:do\s*i\s*have|are\s*saved)',
                r'(?:show|list)\s*(?:me\s*)?(?:all\s*)?(?:saved|my)\s*(?:posts?|scholarships?)',
                r'(?:search|find)\s*(?:me\s*)?.*(?:ml|ai|machine\s*learning|nlp|cv|computer\s*vision)',
            ],
            Intent.WRITE_EMAIL: [
                r'(?:write|draft|compose|generate)\s*(?:an?\s*)?(?:email|mail|letter)',
                r'(?:write|draft|compose)\s*(?:to\s*)?(?:prof|professor|dr)',
                r'i\s*(?:want|need)\s*to\s*(?:email|write|contact)',
                r'(?:email|contact)\s*(?:this\s*)?(?:professor|prof|dr)',
                r'(?:write|draft)\s*(?:me\s*)?(?:an?\s*)?(?:email|letter)',
                r'(?:compose|draft)\s*(?:an?\s*)?(?:email|letter)\s*(?:to|for)',
            ],
            Intent.APPLY_SCHOLARSHIP: [
                r'(?:apply|application)\s*(?:for\s*)?(?:this\s*)?(?:scholarship|position)',
                r'i\s*(?:want|like)\s*to\s*apply',
                r'(?:help\s*me\s*)?apply\s*(?:to|for)',
                r'(?:submit|send)\s*(?:an?\s*)?application',
                r'(?:apply|sending)\s*(?:to|for)\s*(?:this\s*)?(?:scholarship|fellowship)',
            ],
            Intent.APPLY_PHD: [
                r'(?:apply|looking)\s*for\s*(?:phd|ph\.d|doctoral|graduate)',
                r'(?:phd|ph\.d|doctoral)\s*(?:position|opportunity|student)',
                r'(?:apply|contact)\s*(?:professor|prof)\s*(?:for|about)\s*(?:phd|position)',
                r'i\s*(?:want|am)\s*(?:to\s*)?(?:do|pursue)\s*phd',
            ],
            Intent.SEND_EMAIL: [
                r'(?:send|deliver|dispatch)\s*(?:the\s*)?(?:email|mail|message)',
                r'(?:send|deliver)\s*(?:it|this|that)',
                r'(?:go\s*ahead\s*and\s*)?send',
                r'(?:yes|sure|please)\s*send',
                r'(?:send|dispatch)\s*(?:this\s*)?(?:to)',
            ],
            Intent.SETUP_SMTP: [
                r'(?:setup|set\s*up|configure)\s*(?:email|smtp|mail)',
                r'(?:email|smtp)\s*(?:settings?|config|setup)',
                r'(?:how|where)\s*(?:do\s*i|can\s*i)\s*(?:set|configure)\s*(?:up\s*)?(?:email|smtp)',
                r'(?:connect|link)\s*(?:my\s*)?(?:email|gmail|outlook)',
            ],
            Intent.SHOW_SAVED: [
                r'(?:show|display|list)\s*(?:me\s*)?(?:my\s*)?(?:saved|generated)\s*(?:emails?|messages?)',
                r'(?:what|which)\s*(?:emails?|messages?)\s*(?:have\s*i|are)',
                r'(?:check|view)\s*(?:my\s*)?(?:sent|saved|drafted)\s*(?:emails?|messages?)',
            ],
            Intent.HELP: [
                r'(?:what|how)\s*(?:can|do)\s*(?:you|i)\s*(?:do|use)',
                r'(?:help|commands?|options?)',
                r'(?:what|which)\s*(?:features?|capabilities?)',
                r'how\s*(?:does\s*this|do\s*i)',
            ],
            Intent.PROFILE_UPDATE: [
                r'(?:update|change|edit|modify)\s*(?:my\s*)?(?:profile|info|information|details)',
                r'(?:my\s*)?(?:name|email|phone|research\s*interest|skill)',
                r'(?:set|change)\s*(?:my\s*)?(?:name|email|phone)',
            ]
        }
    
    def detect(self, text: str) -> DetectedIntent:
        """Detect intent from user message."""
        text_lower = text.lower().strip()
        
        best_intent = Intent.GENERAL_CHAT
        best_confidence = 0.0
        entities = {}
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = 0.9
                    
                    if intent == Intent.WRITE_EMAIL:
                        email_match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text)
                        if email_match:
                            entities['email'] = email_match.group(0)
                            confidence = 0.95
                        
                        name_match = re.search(r'(?:prof|professor|dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
                        if name_match:
                            entities['professor_name'] = name_match.group(1)
                            confidence = 0.95
                    
                    if intent == Intent.ADD_POST:
                        if any(kw in text_lower for kw in ['scholarship', 'fellowship', 'grant']):
                            entities['post_type'] = 'scholarship'
                        elif any(kw in text_lower for kw in ['phd', 'doctoral', 'position']):
                            entities['post_type'] = 'phd_position'
                        elif any(kw in text_lower for kw in ['professor', 'lab', 'research group']):
                            entities['post_type'] = 'professor'
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        return DetectedIntent(
            intent=best_intent,
            confidence=best_confidence,
            entities=entities,
            raw_text=text
        )
