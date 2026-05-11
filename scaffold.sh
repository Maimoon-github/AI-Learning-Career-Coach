#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ai-coach project scaffolding script
# Creates the full directory tree and placeholder files.
# Run: chmod +x scaffold.sh && ./scaffold.sh
# =============================================================================

PROJECT_ROOT="ai-coach"

# Detect if we are running inside the project directory
if [[ -d "$PROJECT_ROOT" ]]; then
    echo "[INFO] Project directory '$PROJECT_ROOT' already exists. Re-running is safe."
else
    mkdir "$PROJECT_ROOT"
fi

cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# 1. Top-level files
# -----------------------------------------------------------------------------
echo "[INFO] Creating top-level files..."

# .env template
cat > .env <<'EOF'
# Environment variables for AI Coach
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=ls_...
SERPER_API_KEY=...
EOF

# requirements.txt
cat > requirements.txt <<'EOF'
# Core
langgraph
langchain
langchain-openai
langchain-community
crewai
unsloth
peft
accelerate
bitsandbytes

# Storage
chromadb
sqlite-utils

# Voice
openai-whisper
kokoro-tts

# UI
chainlit

# Observability
opentelemetry-api
opentelemetry-sdk
prometheus-client

# Utilities
python-dotenv
pydantic
duckduckgo-search
PyGithub
kaggle
EOF

# README.md placeholder
cat > README.md <<'EOF'
# AI Coach – Personalised AI Learning Assistant

Project scaffolding created. Replace this with a proper README.
EOF

# -----------------------------------------------------------------------------
# 2. Directory structure
# -----------------------------------------------------------------------------
echo "[INFO] Creating directory tree..."

# Create all directories with -p to be safe
mkdir -p \
    src/state \
    src/agents \
    src/graph \
    src/memory \
    src/rag/sources \
    src/tools \
    src/voice \
    src/finetune \
    src/ui \
    src/observability \
    data/chroma \
    data/training \
    scripts \
    tests/unit \
    tests/integration \
    tests/evals

# -----------------------------------------------------------------------------
# 3. Placeholder files (empty __init__.py and basic Python modules)
# -----------------------------------------------------------------------------
echo "[INFO] Creating Python placeholder files..."

# Helper function to create a file with a docstring
write_py_file() {
    local filepath="$1"
    local description="$2"
    cat > "$filepath" <<EOF
"""${description}"""

EOF
}

# ---------- src/ ----------
touch src/__init__.py               # can remain completely empty

# state
write_py_file src/state/__init__.py              "State management package."
write_py_file src/state/schema.py                "TypedDict state and Pydantic models."

# agents
write_py_file src/agents/__init__.py             "Agent implementations."
write_py_file src/agents/supervisor.py           "LangGraph supervisor node."
write_py_file src/agents/profile_analyst.py      "CrewAI Agent: GitHub/Kaggle analysis."
write_py_file src/agents/curriculum_planner.py   "CrewAI Agent: learning path design."
write_py_file src/agents/project_builder.py      "CrewAI Agent: project generation."
write_py_file src/agents/evaluator.py            "LLM-as-Judge evaluator node."
write_py_file src/agents/reporter.py             "Weekly report generator."

# graph
write_py_file src/graph/__init__.py              "Graph package."
write_py_file src/graph/builder.py               "StateGraph construction."
write_py_file src/graph/nodes.py                 "All graph node functions."
write_py_file src/graph/edges.py                 "Conditional routing logic."
write_py_file src/graph/checkpointer.py          "SQLite persistence setup."

# memory
write_py_file src/memory/__init__.py             "Memory management."
write_py_file src/memory/short_term.py           "In-graph message window."
write_py_file src/memory/long_term.py            "SQLite user profile store."
write_py_file src/memory/vector_store.py         "ChromaDB RAG store."

# rag
write_py_file src/rag/__init__.py                "RAG pipeline."
write_py_file src/rag/ingestion.py               "Document ingestion pipeline."
write_py_file src/rag/retriever.py               "Adaptive retriever."
write_py_file src/rag/sources/github_loader.py   "GitHub source loader."
write_py_file src/rag/sources/kaggle_loader.py   "Kaggle source loader."
write_py_file src/rag/sources/document_loader.py "Generic document loader."

# tools
write_py_file src/tools/__init__.py              "Tool registry."
write_py_file src/tools/registry.py              "Central tool registry."
write_py_file src/tools/github_tools.py          "GitHub API tools."
write_py_file src/tools/kaggle_tools.py          "Kaggle API tools."
write_py_file src/tools/web_search_tool.py       "DuckDuckGo search tool."
write_py_file src/tools/resource_finder.py       "Resource finder tool."
write_py_file src/tools/code_executor.py         "Code execution sandbox."

# voice
write_py_file src/voice/__init__.py              "Voice interfaces."
write_py_file src/voice/stt.py                   "Whisper STT integration."
write_py_file src/voice/tts.py                   "Kokoro TTS integration."

# finetune
write_py_file src/finetune/__init__.py           "Fine-tuning pipeline."
write_py_file src/finetune/data_prep.py          "Convert user notes -> training pairs."
write_py_file src/finetune/trainer.py            "Unsloth + PEFT LoRA fine-tuning."
write_py_file src/finetune/export.py             "GGUF export + Ollama Modelfile."
write_py_file src/finetune/scheduler.py          "Trigger weekly fine-tuning runs."

# ui
write_py_file src/ui/__init__.py                 "UI package."
write_py_file src/ui/app.py                      "Chainlit entrypoint."
write_py_file src/ui/callbacks.py                "Message handlers."
write_py_file src/ui/components.py               "Custom Chainlit elements."

# observability
write_py_file src/observability/__init__.py      "Observability package."
write_py_file src/observability/tracer.py        "OpenTelemetry setup."
write_py_file src/observability/metrics.py       "Prometheus metrics."

# ---------- scripts/ ----------
write_py_file scripts/ingest_user.py       "One-time profile ingestion."
write_py_file scripts/run_finetune.py      "Manual fine-tune trigger."
write_py_file scripts/generate_report.py   "Manual report trigger."

# ---------- tests/ ----------
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/evals/__init__.py
write_py_file tests/evals/agent_trajectories.py "LangSmith evals for agent trajectories."

# -----------------------------------------------------------------------------
# 4. Git-keep critical empty directories (so they are tracked)
# -----------------------------------------------------------------------------
echo "[INFO] Adding .gitkeep files for empty data directories..."

touch data/chroma/.gitkeep
touch data/training/.gitkeep

# data/*.db files are created at runtime, but we can add a placeholder to .gitignore.
# We'll just create a .gitignore in data/ to ignore .db files but keep .gitkeep
cat > data/.gitignore <<'EOF'
# Ignore database files and vector store contents
*.db
chroma/*
!chroma/.gitkeep
training/*
!training/.gitkeep
EOF

# -----------------------------------------------------------------------------
# 5. Make scripts executable
# -----------------------------------------------------------------------------
chmod +x scripts/*.py

# -----------------------------------------------------------------------------
# 6. Final message
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo " AI Coach project scaffolding complete in '$PWD'"
echo "======================================================================"
echo "Next steps:"
echo "  cd $PROJECT_ROOT"
echo "  python -m venv .venv && source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  # Start building!"