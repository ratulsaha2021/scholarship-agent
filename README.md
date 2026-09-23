# Scholarship & PhD Application Agent

A chat assistant that turns a scholarship/PhD position post into a personalised, human-sounding application email, and can send it for you.

Paste a post (or upload a screenshot), and the agent:

1. Parses the title, institution, deadline, contact email and any required subject line
2. Uses your CV/profile to draft a short, specific email (no invented facts)
3. Lets you edit it in plain language ("make it shorter", "mention my TB paper")
4. Sends it over SMTP with your CV attached, only after you explicitly confirm

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
2. **Paste a post** (at least ~20 words), or upload a screenshot of one.
3. Type **`write`** to draft the email.
4. Refine it: anything you type is treated as an edit instruction.
5. Type **`send`** to send.

| Command | What it does |
|---------|--------------|
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
│   ├── discovery.py        # Target loading / scraping (CLI)
│   ├── agent.py            # CLI entry point
│   └── config.py           # config/settings.json loader
├── config/
│   ├── settings.json       # Humanisation/discovery settings
│   └── ai_patterns.json    # AI-writing patterns to avoid
├── resources/              # Your CV and profile (gitignored)
├── data/                   # Saved posts (gitignored)
└── tests/                  # pytest suite
```

## Tests

```bash
uv run --group dev pytest
```

## Privacy

Your CV, profile, saved posts, API keys and SMTP credentials stay on your machine in gitignored files. Post text and your profile are sent to Groq when Groq is used.
