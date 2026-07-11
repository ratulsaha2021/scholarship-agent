"""RAG module - simple in-memory vector store for applications."""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"

@dataclass
class ApplicationPost:
    id: str
    title: str
    institution: str
    content: str
    post_type: str
    url: str = ""
    deadline: str = ""
    requirements: str = ""
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "institution": self.institution,
            "content": self.content,
            "post_type": self.post_type,
            "url": self.url,
            "deadline": self.deadline,
            "requirements": self.requirements,
            "metadata": self.metadata
        }

class SimpleEmbedder:
    """Simple TF-IDF style embedder."""
    
    def __init__(self):
        self.vocab = {}
        self.idf = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                      'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                      'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                      'don', 'now', 'and', 'but', 'or', 'if', 'while', 'this', 'that',
                      'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                      'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their'}
        return [t for t in tokens if t not in stop_words and len(t) > 2]
    
    def fit(self, documents: List[str]):
        """Build vocabulary from documents."""
        doc_count = len(documents)
        word_doc_count = {}
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                word_doc_count[token] = word_doc_count.get(token, 0) + 1
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(word_doc_count.keys()))}
        self.idf = {word: np.log(doc_count / (count + 1)) + 1 for word, count in word_doc_count.items()}
    
    def transform(self, text: str) -> np.ndarray:
        """Transform text to vector."""
        tokens = self._tokenize(text)
        vector = np.zeros(len(self.vocab))
        
        for token in tokens:
            if token in self.vocab:
                tf = tokens.count(token) / len(tokens)
                vector[self.vocab[token]] = tf * self.idf.get(token, 1)
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

class RAGStore:
    """Simple in-memory vector store for application posts."""
    
    def __init__(self):
        self.posts: List[ApplicationPost] = []
        self.embedder = SimpleEmbedder()
        self.data_file = DATA_DIR / "posts.json"
        self._load_posts()
    
    def _load_posts(self):
        """Load posts from file."""
        if self.data_file.exists():
            with open(self.data_file, "r") as f:
                data = json.load(f)
            for item in data:
                post = ApplicationPost(
                    id=item["id"],
                    title=item.get("title", ""),
                    institution=item.get("institution", ""),
                    content=item.get("content", ""),
                    post_type=item.get("post_type", ""),
                    url=item.get("url", ""),
                    deadline=item.get("deadline", ""),
                    requirements=item.get("requirements", "")
                )
                self.posts.append(post)
            if self.posts:
                self.embedder.fit([p.content for p in self.posts])
    
    def _save_posts(self):
        """Save posts to file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self.posts]
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_post(self, post: ApplicationPost) -> str:
        """Add an application post to the store."""
        if not post.id:
            post.id = hashlib.md5(post.content.encode()).hexdigest()[:12]
        
        existing = [p for p in self.posts if p.id == post.id]
        if existing:
            return post.id
        
        self.posts.append(post)
        
        if len(self.posts) > 1:
            self.embedder.fit([p.content for p in self.posts])
        
        self._save_posts()
        return post.id
    
    def search(self, query: str, n_results: int = 5, post_type: Optional[str] = None) -> List[Dict]:
        """Search for relevant posts using cosine similarity."""
        if not self.posts:
            return []
        
        if len(self.posts) == 1:
            return [{
                "content": self.posts[0].content,
                "metadata": self.posts[0].to_dict(),
                "score": 1.0
            }]
        
        self.embedder.fit([p.content for p in self.posts])
        query_vector = self.embedder.transform(query)
        
        scored_posts = []
        for post in self.posts:
            if post_type and post.post_type != post_type:
                continue
            
            post_vector = self.embedder.transform(post.content)
            score = np.dot(query_vector, post_vector)
            scored_posts.append((post, score))
        
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for post, score in scored_posts[:n_results]:
            results.append({
                "content": post.content,
                "metadata": post.to_dict(),
                "score": float(score)
            })
        
        return results
    
    def get_post(self, post_id: str) -> Optional[Dict]:
        """Get a specific post by ID."""
        for post in self.posts:
            if post.id == post_id:
                return {
                    "id": post.id,
                    "content": post.content,
                    "metadata": post.to_dict()
                }
        return None
    
    def get_all_posts(self) -> List[Dict]:
        """Get all posts in the store."""
        return [{
            "id": p.id,
            "content": p.content,
            "metadata": p.to_dict()
        } for p in self.posts]
    
    def delete_post(self, post_id: str) -> bool:
        """Delete a post from the store."""
        for i, post in enumerate(self.posts):
            if post.id == post_id:
                self.posts.pop(i)
                self._save_posts()
                return True
        return False
    
    def clear(self) -> bool:
        """Clear all posts from the store."""
        self.posts = []
        self._save_posts()
        return True
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "total_posts": len(self.posts),
            "collection_name": "application_posts"
        }

class PostProcessor:
    """Processes raw text/images into structured application posts."""
    
    def process_text(self, text: str, post_type: str = "scholarship") -> ApplicationPost:
        """Process raw text into an ApplicationPost."""
        import re
        
        lines = text.strip().split('\n')
        title = lines[0].strip() if lines else "Untitled"
        
        institution = ""
        for line in lines[:10]:
            if any(kw in line.lower() for kw in ['university', 'institute', 'college', 'lab', 'school']):
                institution = line.strip()
                break
        
        deadline = ""
        deadline_match = re.search(r'(?:deadline|due|close|apply by)[:\s]*(.*?)(?:\n|$)', text, re.IGNORECASE)
        if deadline_match:
            deadline = deadline_match.group(1).strip()
        
        requirements = ""
        req_match = re.search(r'(?:requirements?|eligibility|qualifications?)[:\s]*\n(.*?)(?:\n[A-Z]|\Z)', text, re.DOTALL | re.IGNORECASE)
        if req_match:
            requirements = req_match.group(1).strip()[:500]
        
        return ApplicationPost(
            id="",
            title=title,
            institution=institution,
            content=text[:2000],
            post_type=post_type,
            deadline=deadline,
            requirements=requirements
        )
    
    def process_image_text(self, ocr_text: str, post_type: str = "scholarship") -> ApplicationPost:
        """Process OCR text from an image into an ApplicationPost."""
        return self.process_text(ocr_text, post_type)
