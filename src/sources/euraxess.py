"""EURAXESS (EU research jobs portal). robots.txt allows /jobs, including keyword searches."""

import re
from typing import List
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup

from .base import Listing

NAME = "EURAXESS"
BASE = "https://euraxess.ec.europa.eu"


def search_url(query: str) -> str:
    return f"{BASE}/jobs/search?f%5B0%5D=keywords%3A{quote(query)}"


def parse(html: str) -> List[Listing]:
    soup = BeautifulSoup(html, "html.parser")
    listings = []
    for item in soup.select("article.ecl-content-item"):
        link = item.select_one("h3 a[href]")
        if not link:
            continue
        meta = item.select_one(".ecl-content-block__primary-meta-item a")
        desc = item.select_one(".ecl-content-block__description")

        fields = {}
        for block in item.select("[class*='id-']"):
            label = block.find(class_="ecl-u-type-bold")
            value = block.find(class_="ecl-text-standard")
            if label and value:
                fields[label.get_text(" ", strip=True).rstrip(":").lower()] = value.get_text(" ", strip=True)

        deadline = next((v for k, v in fields.items() if "deadline" in k), "")
        location = fields.get("work locations", "")
        listings.append(Listing(
            title=link.get_text(" ", strip=True),
            url=urljoin(BASE, link["href"]),
            source=NAME,
            institution=meta.get_text(" ", strip=True) if meta else "",
            location=re.sub(r"^Number of offers: \d+,\s*", "", location)[:120],
            deadline=deadline,
            snippet=desc.get_text(" ", strip=True)[:400] if desc else "",
        ))
    return listings
