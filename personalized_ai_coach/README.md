# Personalized AI Learning & Career Coach

A production-grade multi-agent system that analyzes your GitHub, Kaggle, and documents to build an adaptive, week-by-week learning path toward your target engineering role — with a full-duplex voice interface and a locally fine-tuned LLM that learns your personal note-taking style.

## Architecture

**LangGraph** orchestrates the stateful workflow. **CrewAI** executes specialized agent crews within each node. **Ollama** serves the LLM locally — no data leaves your machine.

```
Profile Ingestion → Skill Gap Assessment → Learning Path
                                               ↓         ↓
                                      Project Gen  Fine-Tuning (parallel)
                                               ↓         ↓
                                          Progress Report → HITL → (loop)
```

See [`docs/architecture_document.md`](docs/architecture_document.md) for the full design.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) (setup script provided)
- Redis (for checkpointing in production)
- PostgreSQL (optional; defaults to SQLite in dev)

## Setup

```bash
# 1. Clone and install
git clone <repo>
cd personalized_ai_coach
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set GITHUB_TOKEN, OLLAMA_BASE_URL, etc.

# 3. Install and start Ollama + pull models
./scripts/setup_ollama.sh

# 4. Verify system health
python main.py health
```

## Usage

```bash
# Start a coaching session
python main.py start \
  --user-id alice \
  --target-role "ML Engineer" \
  --github https://github.com/alice \
  --kaggle alice_kaggle \
  --docs ./resume.pdf ./notes.md

# With voice interface
python main.py start --user-id alice --target-role "ML Engineer" --voice

# Start the Web GUI (Streamlit)
streamlit run app.py
```

## Running Tests

```bash
pytest tests/ -v --asyncio-mode=auto
pytest tests/unit/ -v           # Unit tests only (fast, no Ollama required)
pytest tests/integration/ -v    # Integration tests (mocked crews)
```

## Project Structure

```
personalized_ai_coach/
├── config/              # agents.yaml, tasks.yaml, llm_config.yaml
├── src/
│   ├── langgraph_workflow/   # Graph, state, and individual nodes
│   ├── crewai_agents/        # Six specialized crews
│   ├── tools/                # GitHub, Kaggle, web search, doc parser, Ollama
│   ├── services/             # Voice (STT/TTS), database, S3 storage
│   ├── utils/                # LLM client, error handling, data preprocessing
│   └── models/               # Pydantic schemas (SkillProfile, LearningPath, ProjectSpec)
├── tests/unit/          # Fast unit tests (mocked external calls)
├── tests/integration/   # Full workflow tests
├── scripts/             # setup_ollama.sh, deploy.sh
├── notebooks/           # Fine-tuning experiment notebook
└── docs/                # Architecture document
```

## Deployment

```bash
# Build, test, and deploy with Docker Compose
./scripts/deploy.sh all
```

## Configuration

Key settings in `config/system_settings.yaml`:

| Setting | Default | Description |
|---|---|---|
| `checkpoint_backend` | `redis` | `memory` \| `redis` \| `postgres` |
| `hitl_timeout_seconds` | `300` | Seconds before HITL auto-expires |
| `fine_tuning.min_examples_required` | `50` | Notes needed before fine-tuning runs |
| `learning.default_duration_weeks` | `12` | Default curriculum length |
| `learning.default_hours_per_week` | `10` | Weekly time budget |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | Yes | Ollama endpoint (default: `http://localhost:11434`) |
| `GITHUB_TOKEN` | Recommended | GitHub PAT for profile analysis |
| `TAVILY_API_KEY` | Optional | Higher-quality web search (falls back to DuckDuckGo) |
| `DATABASE_URL` | Production | PostgreSQL connection string |
| `REDIS_URL` | Production | Redis connection string |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing |