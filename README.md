# Scholarship & PhD Application Agent

A chat assistant that turns a scholarship/PhD position post into a personalised, human-sounding application email, and can send it for you.

Paste a post, a link to one, or a screenshot, or search job sites from the chat, and the agent:

1. Finds openings on EURAXESS and jobs.ac.uk, ranked against your profile (optional)
2. Parses the title, institution, deadline, contact email and any required subject line
3. Uses your CV/profile to draft a short, specific email (no invented facts)
4. Lets you edit it in plain language ("make it shorter", "mention my TB paper")
5. Sends it over SMTP with your CV attached, only after you explicitly confirm

## How it works

Drafting runs as a pipeline in `src/hybrid_llm.py`:

| Step | Preferred model | Fallback |
|------|-----------------|----------|
| Draft email | Local model via [Ollama](https://ollama.com) | Groq |
| Check for AI-writing patterns | Local model via Ollama | Groq |
| Final humanising rewrite | Groq | Local model |

You need **at least one** of Ollama or a Groq API key. If neither works, the agent reports the error rather than producing a placeholder email.

## Setup

### 1. Install dependencies

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

Optional, for reading screenshots of posts:

```bash
sudo apt-get install tesseract-ocr
```

### 2. Configure a model

**Groq (cloud, free tier available):** get a key from https://console.groq.com/keys, then either type `setup groq gsk_...` in the chat or `export GROQ_API_KEY=gsk_...`.

**Ollama (local):**

```bash
ollama pull llama3.1:8b
```

Model names and the Groq key are stored in `config/llm_config.json` (gitignored), which is created the first time you save a setting:

```json
{
  "local_model": "llama3.1:8b",
  "local_base_url": "http://localhost:11434",
  "groq_api_key": "gsk_...",
  "groq_model": "openai/gpt-oss-120b"
}
```

Groq retires models from time to time. If generation fails with `model_not_found`, pick a current model from https://console.groq.com/docs/models.

### 3. Run

```bash
uv run streamlit run web_app.py
```

Open http://localhost:8501.

## Using the chat

1. **Upload your CV** (PDF/DOCX/TXT) from the sidebar. It's parsed into `resources/user_data.json` and kept as `resources/cv.<ext>` so it can be attached to emails.
2. **Load a post**: paste its text (at least ~20 words) or its link, upload a screenshot, or search (`find machine learning phd`, then `apply 3`).
3. Type **`write`** to draft the email.
4. Refine it: anything you type is treated as an edit instruction.
5. Type **`send`** to send.

| Command | What it does |
|---------|--------------|
| `find machine learning phd` | Search EURAXESS + jobs.ac.uk (+ your `targets.json` pages), ranked against your profile |
| `apply 3` | Load result 3 from the last search as the current post |
| *a link* | Read a post straight from its web page |
| `watch QUERY` / `unwatch QUERY` / `watches` | Manage saved searches |
| `digest` | Show listings for your saved searches that you haven't seen yet |
| `write` | Draft an email for the loaded post |
| `send` / `yes` | Send the drafted email (short, explicit confirmations only) |
| `to: prof@uni.edu` | Set the recipient if the post didn't include one |
| `cancel` | Discard the draft |
| `status` | Show your profile and model/email status |
| `my skills`, `my publications`, … | Show parts of your profile |
| `setup email gmail` (or `outlook`, `yahoo`) | Start email setup |
| `email: you@gmail.com password: app-password` | Save SMTP credentials (tested before saving) |
| `setup groq KEY` | Save a Groq API key |

For Gmail, use an [App Password](https://myaccount.google.com/apppasswords), not your normal password. Credentials are stored in `config/email_config.json` (gitignored).

## Web search

Search only uses sites whose `robots.txt` allows automated access:

| Source | What it covers |
|--------|----------------|
| [EURAXESS](https://euraxess.ec.europa.eu/jobs/search) | EU research jobs, PhD and postdoc positions |
| [jobs.ac.uk](https://www.jobs.ac.uk) | UK academic jobs and PhD studentships |
| `resources/targets.json` | Lab/department pages you list yourself, included when they mention your search terms |

FindAPhD, ScholarshipDb, AcademicPositions and ResearchGate block automated requests (HTTP 403), so they aren't used. For posts on those sites, copy the text and paste it.

Fetching is deliberately polite (`src/web_fetch.py`):
- Checks `robots.txt` before every request.
- Waits at least 2 s between requests to the same site (`discovery.delay_between_requests` in `config/settings.json`).
- Caches pages for 6 hours in `data/cache/`.
- Identifies itself with an honest `ScholarAgent` user-agent.

To add a site, create `src/sources/<site>.py` with `NAME`, `search_url(query)` and `parse(html) -> List[Listing]`, then add it to `SEARCH_SOURCES` in `src/sources/__init__.py`.

### Daily digest

Saved searches (`watch ...`) live in `data/watch.json`. To collect new matches on a schedule, add a cron job (`crontab -e`):

```cron
0 8 * * * cd /path/to/scholarship-agent && ~/.local/bin/uv run python -m src.digest >> data/digest.log 2>&1
```

New listings are queued, and the chat mentions them when you say hi. Type `digest` to see them. Nothing is ever emailed automatically: every email still goes through `write`, your review, and an explicit `send`.

## CLI mode (legacy)

There is also a Rich-based CLI for writing cold emails to professors and working through a list of targets in `resources/targets.json`:

```bash
uv run python -m src.agent --setup      # create sample resources/targets
uv run python -m src.agent              # interactive menu
uv run python -m src.agent --auto       # draft emails for all targets
uv run python -m src.agent --professor "Jane Doe" --email jane@uni.edu --topic "graph neural networks"
```

CLI drafts are saved to `resources/saved_responses/` and are never sent automatically.

## Project structure

```
scholarship-agent/
├── web_app.py              # Streamlit chat UI (main entry point)
├── src/
│   ├── chat_agent.py       # Chat flow: post → CV → draft → edit → send
│   ├── hybrid_llm.py       # Ollama + Groq pipeline
│   ├── writer.py           # Email prompts
│   ├── humanizer.py        # Rule-based AI-pattern detection
│   ├── cv_extractor.py     # CV parsing (PDF/DOCX/TXT)
│   ├── rag_store.py        # Post parsing + local post store (data/posts.json)
│   ├── ocr_processor.py    # Screenshot OCR (tesseract)
│   ├── email_sender.py     # SMTP sending with attachments
│   ├── resource_loader.py  # Loads the user profile from resources/
│   ├── web_fetch.py        # Polite fetching: robots.txt, rate limit, cache, text extraction
│   ├── sources/            # EURAXESS, jobs.ac.uk parsers + ranking
│   ├── digest.py           # Saved searches and new-listing digest (cron entry point)
│   ├── discovery.py        # Target loading / search (CLI)
│   ├── agent.py            # CLI entry point
│   └── config.py           # config/settings.json loader
├── config/
│   ├── settings.json       # Humanisation/discovery settings
│   └── ai_patterns.json    # AI-writing patterns to avoid
├── resources/              # Your CV and profile (gitignored)
├── data/                   # Saved posts, page cache, watches (gitignored)
└── tests/                  # pytest suite
```

## Tests

```bash
uv run --group dev pytest
```

## Privacy

Your CV, profile, saved posts, API keys and SMTP credentials stay on your machine in gitignored files. Post text and your profile are sent to Groq when Groq is used.
