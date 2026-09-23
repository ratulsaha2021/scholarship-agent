"""jobs.ac.uk (UK academic jobs, incl. PhD studentships). robots.txt allows search pages."""

import re
from typing import List
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup

from .base import Listing

NAME = "jobs.ac.uk"
BASE = "https://www.jobs.ac.uk"


def search_url(query: str) -> str:
    return f"{BASE}/search/?keywords={quote_plus(query)}&sortOrder=0&pageSize=25"


def parse(html: str) -> List[Listing]:
    soup = BeautifulSoup(html, "html.parser")
    listings = []
    for item in soup.select("div.j-search-result__result"):
        link = item.select_one(".j-search-result__text a[href^='/job/']")
        if not link:
            continue

        def text(selector):
            el = item.select_one(selector)
            return el.get_text(" ", strip=True) if el else ""

        location = ""
        for div in item.select(".j-search-result__text > div"):
            t = div.get_text(" ", strip=True)
            if t.startswith("Location:"):
                location = t.replace("Location:", "").strip()

        closes = text(".j-search-result__date--blue")
        salary = re.sub(r"\s+", " ", text(".j-search-result__info"))
        listings.append(Listing(
            title=link.get_text(" ", strip=True),
            url=urljoin(BASE, link["href"]),
            source=NAME,
            institution=text(".j-search-result__employer"),
            location=location,
            deadline=f"Closes {closes}" if closes else "",
            snippet=" · ".join(p for p in [text(".j-search-result__department"), salary] if p),
        ))
    return listings
