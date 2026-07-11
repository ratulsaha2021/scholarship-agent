# Scholarship & PhD Application Agent

An AI agent that automatically applies for Master's/PhD scholarships and sends professional emails to professors. Built with Llama 3.1 8B Instruct for humanized, professional writing.

## Features

- **Humanized Writing**: Avoids AI detection patterns using Wikipedia-based analysis
- **Auto-Apply**: Discovers and applies to scholarships/PhD positions
- **Professor Emails**: Generates personalized, professional emails
- **Resource Management**: Loads your CV, research interests, and templates
- **Dual Discovery**: Manual URL list + web scraping for opportunities

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model (First Run)

The agent will automatically download Llama 3.1 8B Instruct on first run (~4.7GB).

### 3. Configure Resources

Place your files in `resources/`:
- `cv.pdf` or `cv.docx` - Your CV/Resume
- `research_interests.txt` - Your research areas
- `cover_letter_template.txt` - Optional cover letter template
- `targets.json` - List of target URLs/emails

### 4. Run the Agent

**Web UI (Recommended):**
```bash
streamlit run web_app.py
```
Then open http://localhost:8501 in your browser.

**CLI Mode:**
```bash
# Interactive mode
python -m src.agent

# Auto-apply to all targets
python -m src.agent --auto

# Generate email for specific professor
python -m src.agent --professor "professor@university.edu"
```

## Project Structure

```
scholarship-agent/
├── src/
│   ├── agent.py              # Main agent orchestrator
│   ├── humanizer.py          # AI pattern detection & avoidance
│   ├── resource_loader.py    # PDF/Word parsing
│   ├── discovery.py          # Opportunity discovery
│   ├── writer.py             # Email/scholarship writer
│   └── config.py             # Configuration
├── web_app.py                # Web UI (Streamlit)
├── resources/
│   ├── templates/            # Email templates
│   └── saved_responses/      # Generated outputs
├── config/
│   └── ai_patterns.json      # AI writing patterns to avoid
├── requirements.txt
└── README.md
```

## How Humanization Works

1. **Pattern Analysis**: Loads AI writing patterns from Wikipedia/research
2. **Avoidance Rules**: Creates rules to avoid common AI tells
3. **Style Matching**: Matches human academic writing style
4. **Variety Injection**: Adds natural variation to prevent detection

## Configuration

Edit `config/settings.json`:
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "max_length": 2048,
  "temperature": 0.8,
  "humanization_level": "high"
}
```
