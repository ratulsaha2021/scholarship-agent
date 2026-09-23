"""Tests for web search: robots rules, source parsers, page extraction and chat commands (offline)."""

import pytest

from src import chat_agent as ca
from src import digest, email_sender as es, rag_store
from src.sources import euraxess, jobs_ac_uk, rank
from src.sources.base import Listing
from src.web_fetch import FetchError, PoliteFetcher, RobotsRules, extract_main_text

from test_chat_agent import FakeLLM, FakeSender

EURAXESS_ROBOTS = """
User-agent: *
Disallow: /*?
Disallow: /*?_format=json
Allow: /jobs
Allow: /jobs/search?page=
"""

EURAXESS_HTML = """
<article class="ecl-content-item"><div>
  <ul><li class="ecl-content-block__primary-meta-item"><a href="/org/ntnu">NTNU</a></li></ul>
  <h3 class="ecl-content-block__title"><a href="/jobs/111"><span>PhD Candidate in Graph Neural Networks</span></a></h3>
  <div class="ecl-content-block__description"><p>Deep learning on graphs.</p></div>
  <div class="id-Work-Locations"><div class="ecl-u-type-bold"><span>Work Locations:</span></div>
    <div class="ecl-text-standard">Number of offers: 1, Norway, Trondheim</div></div>
  <div class="id-Application-Deadline"><div class="ecl-u-type-bold"><span>Application Deadline:</span></div>
    <div class="ecl-text-standard">20 Oct 2026 - 23:59 (Europe/Oslo)</div></div>
</div></article>
<article class="ecl-content-item"><div>
  <h3><a href="/jobs/222"><span>Postdoc in Marine Biology</span></a></h3>
</div></article>
"""

JOBS_AC_UK_HTML = """
<div class="j-search-result__result" data-advert-id="1">
  <div class="j-search-result__text">
    <a href="/job/ABC123/phd-studentship-machine-learning">PhD Studentship: Machine Learning for X-ray Imaging</a>
    <div class="j-search-result__department">Computer Science</div>
    <div class="j-search-result__employer"><b>University of Bristol</b></div>
    <div>Location: Bristol</div>
    <div class="j-search-result__info"><strong>Salary:</strong> £20,780 stipend</div>
  </div>
  <div class="j-search-result__date"><span class="j-search-result__date--blue">04 Nov</span></div>
</div>
"""

DETAIL_HTML = """
<html><head><title>Job offer</title><meta property="og:title" content="PhD Candidate in Graph Neural Networks"></head>
<body><nav>Menu Login</nav><main>
  <h1>Job offer</h1>
  <p>We are hiring a <strong>PhD candidate</strong> to work on <a href="/gnn">graph neural networks</a>
  for medical imaging. The position is fully funded for three years and includes a generous travel budget.
  Applicants should hold a master's degree in computer science, statistics or a related field, with strong Python skills.</p>
  <p>Application Deadline: 20 Oct 2026</p>
  <p>Contact <a href="mailto:jane.doe@ntnu.no">Professor Jane Doe</a> with questions.</p>
</main><footer>Cookie policy</footer></body></html>
"""


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def fetch(self, url):
        self.requested.append(url)
        for prefix, html in self.pages.items():
            if url.startswith(prefix):
                return html
        raise FetchError(f"blocked the request (403): {url}")


def test_robots_longest_match_allows_euraxess_keyword_search():
    rules = RobotsRules(EURAXESS_ROBOTS)
    assert rules.allowed("/jobs/search?f%5B0%5D=keywords%3Aml")
    assert rules.allowed("/jobs/467780")
    assert not rules.allowed("/partnering/search?q=x")


def test_robots_prefers_group_for_our_agent():
    rules = RobotsRules("User-agent: *\nAllow: /\n\nUser-agent: ScholarAgent\nDisallow: /private\n")
    assert not rules.allowed("/private/page")
    assert rules.allowed("/public")


def test_fetcher_refuses_disallowed_pages(tmp_path, monkeypatch):
    fetcher = PoliteFetcher(delay=0, cache_dir=tmp_path)
    monkeypatch.setattr(fetcher, "is_allowed", lambda url: False)
    with pytest.raises(FetchError, match="robots"):
        fetcher.fetch("https://example.org/jobs")


def test_euraxess_parser():
    listings = euraxess.parse(EURAXESS_HTML)
    assert len(listings) == 2
    first = listings[0]
    assert first.title == "PhD Candidate in Graph Neural Networks"
    assert first.url == "https://euraxess.ec.europa.eu/jobs/111"
    assert first.institution == "NTNU"
    assert first.location == "Norway, Trondheim"
    assert first.deadline.startswith("20 Oct 2026")


def test_jobs_ac_uk_parser():
    [listing] = jobs_ac_uk.parse(JOBS_AC_UK_HTML)
    assert listing.title.startswith("PhD Studentship")
    assert listing.url == "https://www.jobs.ac.uk/job/ABC123/phd-studentship-machine-learning"
    assert listing.institution == "University of Bristol"
    assert listing.location == "Bristol"
    assert listing.deadline == "Closes 04 Nov"


def test_extract_main_text_keeps_sentences_and_mailto():
    title, text = extract_main_text(DETAIL_HTML)
    assert title == "PhD Candidate in Graph Neural Networks"
    assert "to work on graph neural networks for medical imaging" in text
    assert "jane.doe@ntnu.no" in text
    assert "Menu" not in text and "Cookie" not in text


def test_rank_prefers_relevant_and_dedupes():
    listings = euraxess.parse(EURAXESS_HTML) + euraxess.parse(EURAXESS_HTML)
    ranked = rank(listings, "deep learning graph neural networks python", "graph neural networks")
    assert len(ranked) == 2
    assert ranked[0].title == "PhD Candidate in Graph Neural Networks"


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "RESOURCES_DIR", tmp_path / "resources")
    monkeypatch.setattr(rag_store, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(digest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(es, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(ca, "HybridLLM", FakeLLM)
    a = ca.ChatAgent()
    a.email_sender = FakeSender()
    a.fetcher = FakeFetcher({
        "https://euraxess.ec.europa.eu/jobs/search": EURAXESS_HTML,
        "https://www.jobs.ac.uk/search": JOBS_AC_UK_HTML,
        "https://euraxess.ec.europa.eu/jobs/111": DETAIL_HTML,
        "https://lab.example.edu/join": DETAIL_HTML,
    })
    return a


def test_find_lists_ranked_results_from_both_sources(agent):
    resp = agent.chat("find graph neural networks phd")
    assert len(agent.search_results) == 3
    assert "EURAXESS" in resp and "jobs.ac.uk" in resp
    assert "apply N" in resp


def test_source_failure_is_reported_not_fatal(agent):
    del agent.fetcher.pages["https://www.jobs.ac.uk/search"]
    resp = agent.chat("find graph neural networks")
    assert len(agent.search_results) == 2
    assert "jobs.ac.uk" in resp and "failed" in resp


def test_apply_loads_listing_as_post(agent):
    agent.chat("find graph neural networks")
    n = next(i for i, l in enumerate(agent.search_results, 1) if l.url.endswith("/jobs/111"))
    resp = agent.chat(f"apply {n}")
    assert agent.current_post["title"] == "PhD Candidate in Graph Neural Networks"
    assert agent.current_post["institution"] == "NTNU"
    assert agent.current_post["metadata"]["professor_email"] == "jane.doe@ntnu.no"
    assert "euraxess.ec.europa.eu/jobs/111" in resp


def test_pasting_a_link_loads_the_post(agent):
    agent.chat("https://lab.example.edu/join")
    assert agent.current_post["title"] == "PhD Candidate in Graph Neural Networks"
    assert agent.current_post["deadline"] == "20 Oct 2026"


def test_blocked_link_explains_fallback(agent):
    resp = agent.chat("https://www.findaphd.com/phds/project/x")
    assert "Couldn't read that page" in resp
    assert agent.current_post is None


def test_link_during_pending_draft_is_not_an_edit(agent):
    from test_chat_agent import draft
    agent.pending_email = draft()
    agent.flow_state = "waiting_send"
    agent.chat("https://lab.example.edu/join")
    assert agent.llm.prompts == []  # no edit call
    assert agent.current_post["title"] == "PhD Candidate in Graph Neural Networks"


def test_watch_and_digest_only_show_new_listings(agent):
    agent.chat("watch graph neural networks")
    assert agent.watch.queries == ["graph neural networks"]
    first = agent.chat("digest")
    assert "3 new" in first
    assert "Nothing new" in agent.chat("digest")
    # watches persist across restarts
    assert digest.WatchList().queries == ["graph neural networks"]


def test_scheduled_digest_inbox_is_shown_in_chat(agent):
    agent.watch.inbox = [Listing(title="Queued PhD", url="https://x.org/1", source="EURAXESS").to_dict()]
    agent.watch.save()
    assert "1 new listing" in agent.chat("hi")
    resp = agent.chat("digest")
    assert "Queued PhD" in resp
    assert agent.watch.inbox == []
