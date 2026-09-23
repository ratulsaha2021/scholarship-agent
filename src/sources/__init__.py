"""Opportunity search across sites that allow automated access, ranked against the user's profile."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..rag_store import SimpleEmbedder
from ..web_fetch import FetchError, PoliteFetcher, extract_main_text
from . import euraxess, jobs_ac_uk
from .base import Listing

SEARCH_SOURCES = [euraxess, jobs_ac_uk]
RESOURCES_DIR = Path(__file__).parent.parent.parent / "resources"


def search_sources(query: str, fetcher: PoliteFetcher) -> Tuple[List[Listing], List[str]]:
    """Search every source. Returns (listings, errors); one failing site doesn't stop the rest."""
    listings, errors = [], []
    for source in SEARCH_SOURCES:
        try:
            listings.extend(source.parse(fetcher.fetch(source.search_url(query))))
        except FetchError as e:
            errors.append(f"{source.NAME}: {e}")
        except Exception as e:  # markup changed, etc.
            errors.append(f"{source.NAME}: couldn't read results ({e})")
    return listings, errors


def target_listings(query: str, fetcher: PoliteFetcher, targets_file: Optional[Path] = None) -> Tuple[List[Listing], List[str]]:
    """Lab/department pages from resources/targets.json that mention the query terms."""
    targets_file = targets_file or RESOURCES_DIR / "targets.json"
    if not targets_file.exists():
        return [], []
    try:
        targets = json.loads(targets_file.read_text())
    except json.JSONDecodeError as e:
        return [], [f"targets.json: invalid JSON ({e})"]

    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    listings, errors = [], []
    for t in targets:
        url = t.get("url", "")
        if not url:
            continue
        try:
            title, text = extract_main_text(fetcher.fetch(url))
        except FetchError as e:
            errors.append(f"{url}: {e}")
            continue
        haystack = text.lower()
        if terms and not all(term in haystack for term in terms):
            continue
        listings.append(Listing(
            title=t.get("title") or title or url, url=url, source="Your targets",
            institution=t.get("institution", ""), deadline=t.get("deadline", ""),
            snippet=text[:400],
        ))
    return listings, errors


def rank(listings: List[Listing], profile_text: str, query: str = "") -> List[Listing]:
    """Order by TF-IDF similarity to the profile + query, and drop duplicate URLs."""
    unique: Dict[str, Listing] = {}
    for listing in listings:
        unique.setdefault(listing.url, listing)
    listings = list(unique.values())
    if not listings:
        return []

    target = f"{query} {query} {profile_text}"  # weight the query above the profile
    embedder = SimpleEmbedder()
    embedder.fit([l.search_text() for l in listings] + [target])
    target_vec = embedder.transform(target)
    for listing in listings:
        listing.score = float(np.dot(target_vec, embedder.transform(listing.search_text())))
    return sorted(listings, key=lambda l: l.score, reverse=True)


def find_opportunities(query: str, profile_text: str, fetcher: PoliteFetcher) -> Tuple[List[Listing], List[str]]:
    found, errors = search_sources(query, fetcher)
    targets, target_errors = target_listings(query, fetcher)
    return rank(found + targets, profile_text, query), errors + target_errors


def profile_text(resources) -> str:
    """The parts of the user's profile that describe their research fit."""
    parts = list(resources.research_interests) + list(resources.skills) + list(resources.publications)
    parts += [e.get("degree", "") + " " + e.get("field", "") for e in resources.education]
    parts += [e.get("title", "") for e in resources.experience]
    return " ".join(p for p in parts if p)
