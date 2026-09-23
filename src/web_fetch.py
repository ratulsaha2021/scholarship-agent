"""Polite web fetching: robots.txt, per-host rate limiting, on-disk cache."""

import hashlib
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"
USER_AGENT = "ScholarAgent/0.1 (personal job-search assistant; +https://github.com/ratulsaha2021/scholarship-agent)"
UA_TOKEN = "scholaragent"


class FetchError(Exception):
    pass


class RobotsRules:
    """robots.txt matcher using Google's rules: longest matching path wins, Allow wins ties,
    `*` and `$` wildcards supported. (urllib.robotparser uses first-match and no wildcards,
    which misreads sites like EURAXESS.)"""

    def __init__(self, text: str):
        self.rules = self._parse(text)

    @staticmethod
    def _parse(text: str) -> List[Tuple[bool, str]]:
        groups: Dict[str, List[Tuple[bool, str]]] = {}
        agents: List[str] = []
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = [p.strip() for p in line.split(":", 1)]
            key = key.lower()
            if key == "user-agent":
                if in_rules:
                    agents, in_rules = [], False
                agents.append(value.lower())
            elif key in ("allow", "disallow"):
                in_rules = True
                for agent in agents:
                    groups.setdefault(agent, [])
                    if value:
                        groups[agent].append((key == "allow", value))
        for agent, rules in groups.items():
            if agent == UA_TOKEN:
                return rules
        return groups.get("*", [])

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        regex = re.escape(pattern).replace(r"\*", ".*")
        if regex.endswith(r"\$"):
            regex = regex[:-2] + "$"
        return re.match(regex, path) is not None

    def allowed(self, path: str) -> bool:
        best_len, allowed = -1, True
        for is_allow, pattern in self.rules:
            if self._matches(pattern, path):
                length = len(pattern)
                if length > best_len or (length == best_len and is_allow):
                    best_len, allowed = length, is_allow
        return allowed


class PoliteFetcher:
    def __init__(self, delay: float = 2.0, cache_ttl: float = 6 * 3600, cache_dir: Optional[Path] = None):
        self.delay = delay
        self.cache_ttl = cache_ttl
        self.cache_dir = cache_dir or DATA_DIR / "cache"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._robots: Dict[str, RobotsRules] = {}
        self._last_request: Dict[str, float] = {}

    def _wait_for_host(self, host: str):
        elapsed = time.time() - self._last_request.get(host, 0)
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request[host] = time.time()

    def _robots_for(self, parsed) -> RobotsRules:
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            try:
                self._wait_for_host(parsed.netloc)
                resp = self.session.get(base + "/robots.txt", timeout=10)
                text = resp.text if resp.status_code == 200 else ""
            except requests.RequestException:
                text = ""
            self._robots[base] = RobotsRules(text)
        return self._robots[base]

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        return self._robots_for(parsed).allowed(path)

    def fetch(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise FetchError(f"Not a web URL: {url}")

        cache_file = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".html")
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < self.cache_ttl:
            return cache_file.read_text(encoding="utf-8")

        if not self.is_allowed(url):
            raise FetchError(f"{parsed.netloc} doesn't allow automated access to this page (robots.txt).")

        self._wait_for_host(parsed.netloc)
        try:
            resp = self.session.get(url, timeout=20)
        except requests.RequestException as e:
            raise FetchError(f"Couldn't reach {parsed.netloc}: {e}")
        if resp.status_code == 403:
            raise FetchError(f"{parsed.netloc} blocked the request (403). Copy the post text and paste it instead.")
        if resp.status_code != 200:
            raise FetchError(f"{parsed.netloc} returned HTTP {resp.status_code}.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(resp.text, encoding="utf-8")
        return resp.text


BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th",
              "dt", "dd", "section", "article", "table", "blockquote", "pre"]
GENERIC_TITLES = {"job offer", "job details", "vacancy", "home"}


def extract_main_text(html: str) -> Tuple[str, str]:
    """Return (title, readable main text) from a page."""
    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        candidates.append(og["content"])
    h1 = soup.find("h1")
    if h1:
        candidates.append(h1.get_text(" ", strip=True))
    if soup.title:
        candidates.append(soup.title.get_text(" ", strip=True))
    title = next((c.strip() for c in candidates if c.strip() and c.strip().lower() not in GENERIC_TITLES), "")

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup

    # Keep contact addresses that only appear in mailto: links
    for a in main.find_all("a", href=True):
        if a["href"].lower().startswith("mailto:"):
            address = a["href"][7:].split("?")[0]
            if address and address not in a.get_text():
                a.append(f" ({address})")
    # Line breaks only at block boundaries, so inline <a>/<strong> and source-code
    # newlines don't split sentences
    for node in main.find_all(string=True):
        if not node.find_parent("pre"):
            node.replace_with(re.sub(r"\s+", " ", str(node)))
    for br in main.find_all("br"):
        br.replace_with("\n")
    for tag in main.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    lines = [re.sub(r"\s+", " ", line).strip() for line in main.get_text().splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text
