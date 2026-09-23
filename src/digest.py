"""Saved searches ("watches") and a digest of listings you haven't seen yet.

Run on a schedule to collect new matches, which the chat shows next time you open it:

    uv run python -m src.digest
"""

import json
from pathlib import Path
from typing import List, Tuple

from .sources import Listing, find_opportunities
from .web_fetch import PoliteFetcher

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_SEEN = 5000


class WatchList:
    """Persists watched queries, URLs already shown to the user, and undelivered digest results."""

    def __init__(self, path: Path = None):
        self.path = path or DATA_DIR / "watch.json"
        data = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                data = {}
        self.queries: List[str] = data.get("queries", [])
        self.seen: List[str] = data.get("seen", [])
        self.inbox: List[dict] = data.get("inbox", [])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "queries": self.queries,
            "seen": self.seen[-MAX_SEEN:],
            "inbox": self.inbox,
        }, indent=2))

    def add_query(self, query: str) -> bool:
        if query.lower() in (q.lower() for q in self.queries):
            return False
        self.queries.append(query)
        self.save()
        return True

    def remove_query(self, query: str) -> bool:
        before = len(self.queries)
        self.queries = [q for q in self.queries if q.lower() != query.lower()]
        self.save()
        return len(self.queries) < before

    def is_new(self, listing: Listing) -> bool:
        return listing.url not in self.seen

    def mark_seen(self, listings: List[Listing]):
        seen = set(self.seen)
        self.seen.extend(l.url for l in listings if l.url not in seen)
        self.save()

    def take_inbox(self) -> List[Listing]:
        items = [Listing(**d) for d in self.inbox]
        self.inbox = []
        self.save()
        return items


def run_digest(watch: WatchList, profile_text: str, fetcher: PoliteFetcher) -> Tuple[List[Listing], List[str]]:
    """Search every watched query and return only listings not seen before (best match first)."""
    new, errors, urls = [], [], set()
    for query in watch.queries:
        found, errs = find_opportunities(query, profile_text, fetcher)
        errors.extend(f"'{query}' — {e}" for e in errs)
        for listing in found:
            if watch.is_new(listing) and listing.url not in urls:
                urls.add(listing.url)
                new.append(listing)
    new.sort(key=lambda l: l.score, reverse=True)
    return new, errors


def main():
    from .config import AgentConfig
    from .resource_loader import UserResources
    from .sources import profile_text

    watch = WatchList()
    if not watch.queries:
        print("No watched searches. In the chat, type e.g. 'watch machine learning phd'.")
        return

    config = AgentConfig.load()
    fetcher = PoliteFetcher(delay=config.discovery.delay_between_requests)
    new, errors = run_digest(watch, profile_text(UserResources.load()), fetcher)

    # Queue for the chat and mark as seen so the next run only reports newer listings
    watch.inbox.extend(l.to_dict() for l in new)
    watch.mark_seen(new)

    print(f"{len(new)} new listing(s) for: {', '.join(watch.queries)}")
    for l in new[:20]:
        print(f"- {l.title} | {l.institution} | {l.deadline} | {l.url}")
    for e in errors:
        print(f"! {e}")


if __name__ == "__main__":
    main()
