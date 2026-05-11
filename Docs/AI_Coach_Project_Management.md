# 🎓 Personalized AI Learning & Career Coach — Complete Project Management Setup

> **Project:** @Maimoon-github's Personalized AI Learning & Career Coach Agent Team  
> **PM Version:** 1.0 | **Date:** May 11, 2026 | **Sprint Cadence:** 2 weeks  
> **Stack:** LangGraph v1.1.x · CrewAI v1.12 · Ollama · ChromaDB · Chainlit · Whisper · Kokoro TTS

---

## 📋 Table of Contents

1. [Executive Summary & Risk Register](#1-executive-summary--risk-register)
2. [Product Backlog — Epics & User Stories](#2-product-backlog--epics--user-stories)
3. [Priority Dashboard](#3-priority-dashboard)
4. [Team Items — Sprint Board (Sprint 1 & 2)](#4-team-items--sprint-board)
5. [Roadmap — Phased Timeline](#5-roadmap--phased-timeline)
6. [My Items — Owner Task List](#6-my-items--owner-task-list)
7. [Bug Tracker](#7-bug-tracker)
8. [Definition of Done & Quality Gates](#8-definition-of-done--quality-gates)
9. [Dependency Map](#9-dependency-map)
10. [Team Roles & RACI](#10-team-roles--raci)

---

## 1. Executive Summary & Risk Register

### Project Health Snapshot

| Metric | Status | Detail |
|--------|--------|--------|
| **Overall Status** | 🟡 Planning | Architecture complete; implementation not started |
| **Schedule Risk** | 🔴 High | Fine-tuning pipeline has GPU dependency |
| **Budget Risk** | 🟢 Low | Fully local/open-source stack; zero API costs |
| **Technical Risk** | 🟡 Medium | LangGraph HITL + CrewAI Ollama integration is new |
| **Team Readiness** | 🟡 Medium | Needs Python 3.11 env + Ollama GPU setup |
| **Scope Clarity** | 🟢 High | 21-section architecture doc is comprehensive |

### Top 5 Strategic Risks

| # | Risk | Probability | Impact | Owner | Mitigation |
|---|------|------------|--------|-------|------------|
| R1 | Ollama OOM crash on inference | Medium | 🔴 High | DevOps | Use q3_K_S quantization; set `OLLAMA_NUM_PARALLEL=1` |
| R2 | Fine-tuning degrades base model | Medium | 🟡 Medium | ML Eng | A/B test before switching; maintain model version pointers |
| R3 | LangGraph infinite loop in graph | Low | 🔴 High | Backend | `max_iterations` guard + `error_recovery_node` |
| R4 | ChromaDB corruption on shutdown | Low | 🔴 High | DevOps | `--allow-reset=false`; daily volume backup cron |
| R5 | CrewAI ↔ Ollama integration breaking change | Medium | 🟡 Medium | Backend | Pin `crewai==1.12.0`; lock `requirements.txt` |

---

## 2. Product Backlog — Epics & User Stories

> **Priority Legend:** 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Low  
> **Story Points:** XS=1 · S=2 · M=3 · L=5 · XL=8 · XXL=13

---

### 🏗️ EPIC 1 — Core Infrastructure & Environment Setup
**Goal:** Establish a reproducible, version-pinned dev environment that every engineer can spin up in under 30 minutes.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| INF-01 | As a developer, I can clone the repo and run `pip install -r requirements.txt` without version conflicts | 🔴 Critical | S=2 | S1 | Backlog |
| INF-02 | As a developer, I can install and warm-start Ollama with `llama3.1:8b` and `nomic-embed-text` via a single script | 🔴 Critical | S=2 | S1 | Backlog |
| INF-03 | As a developer, I have a `.env.example` with all required variables documented | 🟠 High | XS=1 | S1 | Backlog |
| INF-04 | As an ops engineer, I can spin up the full stack (Ollama + ChromaDB + Chainlit) via `docker compose up -d` | 🟠 High | M=3 | S1 | Backlog |
| INF-05 | As a developer, I can run `pytest tests/unit/` and get ≥90% pass rate against mocked LLM calls | 🟡 Medium | L=5 | S2 | Backlog |
| INF-06 | As a developer, I have a GitHub Actions CI pipeline that lints and runs unit tests on every PR | 🟡 Medium | M=3 | S2 | Backlog |

**Epic Total:** 16 points | **Target:** Sprint 1–2

---

### 🧠 EPIC 2 — LangGraph State Machine & Orchestration
**Goal:** Build the core directed cyclic graph with all nodes, edges, conditional routing, and SQLite checkpointing.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| LG-01 | As a backend engineer, I can define the `CoachState` TypedDict with all 20+ fields and Pydantic sub-models | 🔴 Critical | M=3 | S1 | Backlog |
| LG-02 | As a backend engineer, I can register all graph nodes (memory_ingest, rag, supervisor, agents, evaluator, hitl, memory_write, finetune_check, voice_output) | 🔴 Critical | L=5 | S1 | Backlog |
| LG-03 | As a backend engineer, I can define all conditional edges with correct routing logic (supervisor → agents, evaluator → retry/hitl/continue) | 🔴 Critical | M=3 | S1 | Backlog |
| LG-04 | As a backend engineer, the graph compiles with `SqliteSaver` checkpointing and `interrupt_before=["hitl"]` | 🔴 Critical | M=3 | S1 | Backlog |
| LG-05 | As a developer, the `max_iterations` cycle guard prevents infinite loops and falls back to `responder` | 🟠 High | S=2 | S2 | Backlog |
| LG-06 | As a developer, the `error_recovery_node` clears stale state and resets the graph when `error_count > 3` | 🟠 High | M=3 | S2 | Backlog |
| LG-07 | As a QA engineer, I can visualize the full graph as a Mermaid diagram via `graph.get_graph().draw_mermaid()` | 🟡 Medium | S=2 | S3 | Backlog |
| LG-08 | As a developer, I can swap the SQLite checkpointer for AsyncPostgresSaver with a single env var change | 🟡 Medium | M=3 | S4 | Backlog |

**Epic Total:** 24 points | **Target:** Sprint 1–2

---

### 🤖 EPIC 3 — Multi-Agent System (CrewAI + LangGraph Nodes)
**Goal:** Implement all 5 specialist agents as production-ready, role-defined CrewAI agents wrapped in LangGraph node functions.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| AG-01 | As a user, the Supervisor Agent correctly routes my request to the right specialist based on my current phase (onboarding / planning / active_learning / weekly_report) | 🔴 Critical | M=3 | S2 | Backlog |
| AG-02 | As a new user, the Profile Analyst Agent analyzes my GitHub profile and returns a structured JSON of skills and gaps within 30 seconds | 🔴 Critical | L=5 | S2 | Backlog |
| AG-03 | As a user, the Curriculum Planner designs a personalized 90-day learning plan with week-by-week topics based on my skill gaps | 🔴 Critical | L=5 | S2 | Backlog |
| AG-04 | As a user, the Project Builder generates a practice project with clear scope (4-12h), tech stack, and evaluation criteria | 🟠 High | L=5 | S3 | Backlog |
| AG-05 | As a system, the Evaluator (LLM-as-Judge) scores each agent output 0.0–1.0 and routes to retry, HITL, or continue | 🟠 High | M=3 | S3 | Backlog |
| AG-06 | As a user, the Reporter generates my weekly progress report on `/report` command within 15 seconds | 🟠 High | M=3 | S3 | Backlog |
| AG-07 | As a developer, all CrewAI agents use `llm="ollama/llama3.1:8b-instruct-q4_K_M"` and work without any OpenAI API key | 🔴 Critical | S=2 | S2 | Backlog |
| AG-08 | As a developer, each agent has `max_iter` set and respects the cycle guard | 🟡 Medium | S=2 | S3 | Backlog |

**Epic Total:** 28 points | **Target:** Sprint 2–3

---

### 🔍 EPIC 4 — Agentic RAG Pipeline
**Goal:** Build an adaptive RAG pipeline with MMR retrieval, document grading, and query rewriting.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| RAG-01 | As a developer, I can ingest PDF, Markdown, and TXT files into ChromaDB via `ingest_documents()` | 🔴 Critical | M=3 | S1 | Backlog |
| RAG-02 | As a system, the `AgenticRetriever` fetches top-6 docs using MMR (k=6, fetch_k=20) via `nomic-embed-text` embeddings | 🔴 Critical | M=3 | S2 | Backlog |
| RAG-03 | As a system, the document grader filters retrieved chunks for relevance using an LLM call before returning them | 🟠 High | M=3 | S2 | Backlog |
| RAG-04 | As a system, when < 2 relevant docs are found, the retriever rewrites the query and retries (max 2 attempts) | 🟠 High | M=3 | S2 | Backlog |
| RAG-05 | As a developer, I can load course materials from GitHub README files via `github_loader.py` | 🟡 Medium | M=3 | S3 | Backlog |
| RAG-06 | As a developer, I can load Kaggle notebook outputs via `kaggle_loader.py` | 🟡 Medium | M=3 | S3 | Backlog |
| RAG-07 | As an ops engineer, I can prune ChromaDB collections older than 30 days via a scheduled script | 🟢 Low | S=2 | S5 | Backlog |

**Epic Total:** 20 points | **Target:** Sprint 1–3

---

### 💾 EPIC 5 — Memory Architecture (3-Tier)
**Goal:** Implement working memory (in-graph), long-term SQLite store, and semantic ChromaDB memory.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| MEM-01 | As a returning user, my profile (skills, gaps, learning plan, session count) persists across sessions via SQLite | 🔴 Critical | M=3 | S2 | Backlog |
| MEM-02 | As a system, the `LongTermMemory` class initializes the SQLite schema (user_profiles, session_summaries, user_notes) on first run | 🔴 Critical | S=2 | S1 | Backlog |
| MEM-03 | As a user, my in-session notes are stored and retrievable for the fine-tuning pipeline | 🟠 High | M=3 | S3 | Backlog |
| MEM-04 | As a user, session summaries are auto-generated and saved at the end of each conversation | 🟡 Medium | M=3 | S4 | Backlog |
| MEM-05 | As a developer, the in-graph message window is limited to the last 20 messages to control context size | 🟠 High | S=2 | S2 | Backlog |

**Epic Total:** 13 points | **Target:** Sprint 1–4

---

### 🛠️ EPIC 6 — Tool Registry & External Integrations
**Goal:** Implement all tool functions with consistent signatures, error handling, and central registry.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| TOOL-01 | As a user, I can provide my GitHub username and the `analyze_github_profile` tool returns my top languages and repo complexity | 🔴 Critical | M=3 | S2 | Backlog |
| TOOL-02 | As a user, I can provide my Kaggle username and the tool returns notebook languages and competition history | 🟠 High | M=3 | S3 | Backlog |
| TOOL-03 | As a system, `duckduckgo_search` returns web results without any API key | 🟠 High | S=2 | S2 | Backlog |
| TOOL-04 | As a system, `find_learning_resources` queries the ChromaDB vector store for relevant course materials | 🔴 Critical | S=2 | S2 | Backlog |
| TOOL-05 | As a system, all tools are registered in `TOOL_MAP` and usable by any CrewAI agent | 🟡 Medium | S=2 | S2 | Backlog |
| TOOL-06 | As a developer, GitHub API calls are cached for 1 hour and retry with exponential backoff | 🟡 Medium | M=3 | S4 | Backlog |
| TOOL-07 | As a system, the tool registry is exposable as an MCP server via `langchain-mcp-adapters` | 🟢 Low | M=3 | S6 | Backlog |

**Epic Total:** 18 points | **Target:** Sprint 2–4

---

### 🎙️ EPIC 7 — Voice Interface (STT + TTS)
**Goal:** Implement fully local, offline voice I/O using Whisper STT and Kokoro ONNX TTS.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| VOICE-01 | As a user, I can type `/voice` in Chainlit to enable voice mode | 🟠 High | S=2 | S3 | Backlog |
| VOICE-02 | As a user, my microphone input is transcribed locally via Whisper `base.en` in < 3 seconds for 10s audio | 🟠 High | M=3 | S3 | Backlog |
| VOICE-03 | As a user, coach responses are spoken aloud via Kokoro ONNX TTS with the `af_sky` voice | 🟠 High | M=3 | S3 | Backlog |
| VOICE-04 | As a user, the transcribed text is shown before submitting so I can correct it | 🟡 Medium | S=2 | S4 | Backlog |
| VOICE-05 | As a developer, Kokoro ONNX model files are auto-downloaded on first `KokoroTTS()` init | 🟡 Medium | S=2 | S3 | Backlog |
| VOICE-06 | As a user, TTS output is capped at 500 characters to prevent excessively long audio responses | 🟢 Low | XS=1 | S3 | Backlog |

**Epic Total:** 13 points | **Target:** Sprint 3–4

---

### 🔧 EPIC 8 — Ollama Fine-Tuning Pipeline
**Goal:** Build the end-to-end pipeline to fine-tune a local LLM on user notes and register it with Ollama.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| FT-01 | As a system, `prepare_training_data()` converts user notes into Alpaca-format JSONL (min 50 chars per note) | 🟠 High | M=3 | S4 | Backlog |
| FT-02 | As a developer, `run_finetune()` uses Unsloth + LoRA (r=16) on `Llama-3.2-3B-Instruct` with QLoRA 4-bit | 🟠 High | XL=8 | S4 | Backlog |
| FT-03 | As a system, fine-tuned model is exported to GGUF (q4_k_m) and registered as `coach-{user_id}-v{date}` in Ollama | 🟠 High | L=5 | S4 | Backlog |
| FT-04 | As a system, `finetune_check_node` triggers a background thread fine-tune when ≥50 new notes accumulate | 🟡 Medium | M=3 | S4 | Backlog |
| FT-05 | As a developer, fine-tuning can be triggered manually via `python scripts/run_finetune.py --user_id {id}` | 🟡 Medium | S=2 | S4 | Backlog |
| FT-06 | As an ops engineer, the new model is A/B tested against the base before `PERSONALIZATION_MODEL` is updated | 🟡 Medium | L=5 | S5 | Backlog |
| FT-07 | As a user, I receive a HITL notification when my personalized model is ready | 🟢 Low | S=2 | S5 | Backlog |

**Epic Total:** 28 points | **Target:** Sprint 4–5

---

### 🖥️ EPIC 9 — Chainlit UI & UX
**Goal:** Build the conversational web UI with session management, streaming, HITL UI, and voice controls.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| UI-01 | As a user, I see a greeting message when I first connect that asks for my target role and GitHub username | 🔴 Critical | S=2 | S3 | Backlog |
| UI-02 | As a user, coach responses stream token-by-token in real time via `astream_events` | 🔴 Critical | M=3 | S3 | Backlog |
| UI-03 | As a user, when a HITL checkpoint fires I see ✅ Approve / ❌ Revise buttons inline in the chat | 🟠 High | M=3 | S3 | Backlog |
| UI-04 | As a user, I can type `/report` to request my weekly progress summary immediately | 🟠 High | S=2 | S3 | Backlog |
| UI-05 | As a developer, the graph is initialized once at startup and shared across sessions for performance | 🟠 High | S=2 | S3 | Backlog |
| UI-06 | As a user, I can upload a PDF or markdown file and it is ingested into my personal ChromaDB collection | 🟡 Medium | M=3 | S4 | Backlog |
| UI-07 | As a user, a sidebar shows my current learning plan week and skill progress | 🟡 Medium | L=5 | S5 | Backlog |
| UI-08 | As an admin, I can set `CHAINLIT_AUTH_SECRET` to protect the UI with password auth | 🟠 High | XS=1 | S3 | Backlog |

**Epic Total:** 21 points | **Target:** Sprint 3–5

---

### 🛡️ EPIC 10 — Governance, Guardrails & HITL
**Goal:** Implement input guardrails, self-healing, retry logic, and HITL breakpoints.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| GOV-01 | As a system, `apply_input_guardrails()` blocks prompt injection patterns before any LLM call | 🔴 Critical | S=2 | S2 | Backlog |
| GOV-02 | As a system, HITL fires when evaluator score < 0.5 or `escalate_to_human=True` | 🔴 Critical | M=3 | S3 | Backlog |
| GOV-03 | As a user, when HITL fires I see the content preview and can approve or request revision | 🔴 Critical | M=3 | S3 | Backlog |
| GOV-04 | As a system, `_call_llm_with_retry` retries on `OutputParserException` with exponential backoff (max 3) | 🟠 High | S=2 | S2 | Backlog |
| GOV-05 | As a developer, HITL state is persisted in the SQLite checkpointer so a page refresh doesn't lose the pause | 🟠 High | M=3 | S4 | Backlog |
| GOV-06 | As an ops engineer, user inputs are truncated at 4000 chars before LLM processing | 🟡 Medium | XS=1 | S2 | Backlog |

**Epic Total:** 14 points | **Target:** Sprint 2–4

---

### 📊 EPIC 11 — Observability, Logging & Metrics
**Goal:** Full structured logging, OpenTelemetry tracing, and Prometheus metrics.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| OBS-01 | As an ops engineer, every agent call is logged as structured JSON with user_id, agent, duration_ms, session_id | 🟠 High | M=3 | S3 | Backlog |
| OBS-02 | As an ops engineer, Prometheus counters track `agent_calls_total`, `llm_tokens_total`, and `eval_scores` | 🟡 Medium | M=3 | S4 | Backlog |
| OBS-03 | As an ops engineer, LangSmith tracing is enabled via `LANGCHAIN_TRACING_V2=true` for visual graph debugging | 🟡 Medium | S=2 | S3 | Backlog |
| OBS-04 | As an ops engineer, OpenTelemetry spans are exported to an OTLP collector when `OTEL_EXPORTER_OTLP_ENDPOINT` is set | 🟢 Low | M=3 | S5 | Backlog |

**Epic Total:** 11 points | **Target:** Sprint 3–5

---

### 🚀 EPIC 12 — Deployment & DevOps
**Goal:** Reproducible production deployment with Docker Compose, security hardening, and model upgrade path.

| ID | User Story | Priority | Points | Sprint | Status |
|----|-----------|---------|--------|--------|--------|
| DEP-01 | As an ops engineer, `docker compose up -d` starts Ollama + ChromaDB + Chainlit with correct networking | 🟠 High | M=3 | S4 | Backlog |
| DEP-02 | As an ops engineer, Ollama container has NVIDIA GPU reservation configured | 🟠 High | S=2 | S4 | Backlog |
| DEP-03 | As an ops engineer, all containers run as non-root users | 🟠 High | S=2 | S4 | Backlog |
| DEP-04 | As an ops engineer, Ollama is NOT exposed externally (bound to localhost only) | 🔴 Critical | XS=1 | S4 | Backlog |
| DEP-05 | As an ops engineer, a daily cron job backs up `data/` volumes to a compressed archive | 🟡 Medium | M=3 | S5 | Backlog |
| DEP-06 | As an ops engineer, I can upgrade models by pulling and restarting without full rebuild | 🟡 Medium | S=2 | S5 | Backlog |

**Epic Total:** 13 points | **Target:** Sprint 4–5

---

## 3. Priority Dashboard

### 🏆 Critical Path Items (Must ship before MVP)

| Rank | Item | Epic | Owner | Blocker For |
|------|------|------|-------|------------|
| 1 | `CoachState` TypedDict + sub-models | LG-01 | Backend | Everything |
| 2 | Environment setup + Ollama pull script | INF-01, INF-02 | DevOps | All dev work |
| 3 | LangGraph graph compile with checkpointer | LG-04 | Backend | Agent integration |
| 4 | ChromaDB ingestion pipeline | RAG-01 | ML Eng | RAG, Curriculum Planner |
| 5 | Supervisor routing logic | AG-01 | Backend | All agent dispatch |
| 6 | Profile Analyst (GitHub → skills JSON) | AG-02 | ML Eng | Curriculum Planner |
| 7 | Chainlit session init + streaming | UI-02 | Frontend | User-facing demo |
| 8 | Input guardrails | GOV-01 | Backend | Security gate |

### 📈 Velocity Targets

| Sprint | Target Points | Focus |
|--------|--------------|-------|
| Sprint 1 | 20 pts | Infrastructure + State Schema + ChromaDB |
| Sprint 2 | 22 pts | Core Graph + Supervisor + Profile Analyst + Tools |
| Sprint 3 | 24 pts | Remaining Agents + Chainlit UI + Voice + Guardrails |
| Sprint 4 | 22 pts | Fine-tuning + HITL + Docker + Observability |
| Sprint 5 | 18 pts | Hardening + A/B testing + Backup + Cleanup |
| Sprint 6 | 12 pts | MCP server + Polish + Docs update |

**Total Backlog:** ~161 story points across 6 sprints (~12 weeks)

### ⚡ Quick Wins (High Value, Low Effort — do these first)

- `INF-03`: Create `.env.example` — 1 point, unblocks all devs immediately
- `MEM-02`: Initialize SQLite schema — 2 points, unblocks memory layer
- `LG-01`: Define `CoachState` — 3 points, unblocks the entire graph
- `GOV-06`: Input truncation at 4000 chars — 1 point, zero-cost guardrail
- `UI-08`: Set `CHAINLIT_AUTH_SECRET` — 1 point, security before first demo

### 🔴 Dependency Red Flags

```
INF-01 ──► LG-01 ──► LG-02 ──► LG-03 ──► LG-04 ──► AG-01 ──► All Agents
                                                    ↑
RAG-01 ──────────────────────────────────────────────────► RAG-02 ──► AG-03
                                                    
FT-01 ──► FT-02 ──► FT-03 ──► FT-04    (Requires GPU — isolate this risk early)
```

---

## 4. Team Items — Sprint Board

### 🏃 SPRINT 1 (May 12 – May 25, 2026) — "Foundation"
**Goal:** Development environment, project scaffold, state schema, and ChromaDB ingestion working end-to-end.

#### 📋 Backlog → Ready

| # | Title | Size | Assignee | Labels |
|---|-------|------|----------|--------|
| INF-01 | Set up Python 3.11 venv + `requirements.txt` | S | @maimoon | `infra` `setup` |
| INF-02 | Write `setup_ollama.sh` script to pull all models | S | @maimoon | `infra` `ollama` |
| INF-03 | Create `.env.example` with all variables documented | XS | @maimoon | `infra` `docs` |
| LG-01 | Define `CoachState` TypedDict + all Pydantic sub-models | M | @backend | `core` `critical` |
| MEM-02 | `LongTermMemory._init_schema()` — SQLite table creation | S | @backend | `memory` `db` |
| RAG-01 | `ingest_documents()` — PDF, Markdown, TXT → ChromaDB | M | @ml-eng | `rag` `chromadb` |

**Sprint 1 Points:** 14 | **Risk:** Python 3.11 / bitsandbytes compatibility on Windows

#### ✅ In Progress → In Review → Done flow:
Each item moves: **Backlog → Ready → In Progress → In Review → Done**

---

### 🏃 SPRINT 2 (May 26 – Jun 8, 2026) — "Graph Core"
**Goal:** LangGraph compiles and routes correctly. Supervisor works. Profile Analyst returns real GitHub data.

| # | Title | Size | Assignee | Labels |
|---|-------|------|----------|--------|
| LG-02 | Register all graph nodes in `StateGraph` | L | @backend | `core` `langgraph` |
| LG-03 | Implement all conditional edges + routing functions | M | @backend | `core` `langgraph` |
| LG-04 | Compile graph with SqliteSaver + `interrupt_before=["hitl"]` | M | @backend | `core` `critical` |
| LG-05 | Add `max_iterations` cycle guard in `supervisor_node` | S | @backend | `guardrails` |
| AG-01 | `supervisor_node` — LLM-based routing + `_detect_phase()` | M | @ml-eng | `agents` `critical` |
| AG-02 | `profile_analyst` CrewAI agent + `run_profile_analyst()` | L | @ml-eng | `agents` `github` |
| AG-07 | Verify CrewAI uses Ollama without OpenAI key | S | @ml-eng | `agents` `ollama` |
| TOOL-01 | `analyze_github_profile()` tool | M | @backend | `tools` `github` |
| TOOL-03 | `duckduckgo_search()` tool | S | @backend | `tools` |
| TOOL-04 | `find_learning_resources()` RAG tool | S | @backend | `tools` `rag` |
| TOOL-05 | Central tool registry `TOOL_MAP` | S | @backend | `tools` |
| GOV-01 | `apply_input_guardrails()` | S | @backend | `security` `critical` |
| GOV-04 | `_call_llm_with_retry()` with tenacity | S | @backend | `resilience` |
| GOV-06 | Input truncation at 4000 chars | XS | @backend | `guardrails` |
| MEM-01 | `LongTermMemory.save_profile()` + `load_profile()` | M | @backend | `memory` |
| MEM-05 | Sliding message window (last 20 messages) | S | @backend | `memory` |
| RAG-02 | `AgenticRetriever` with MMR (k=6, fetch_k=20) | M | @ml-eng | `rag` |
| RAG-03 | Document grader — LLM relevance filter | M | @ml-eng | `rag` |
| RAG-04 | Query rewriter — retry on < 2 graded docs | M | @ml-eng | `rag` |

**Sprint 2 Points:** 44 | ⚠️ Heavy sprint — consider splitting AG-02 and RAG-03 to S3 if velocity is lower.

---

### 🏃 SPRINT 3 (Jun 9 – Jun 22, 2026) — "Agents + UI"
**Goal:** All specialist agents functional. Chainlit UI streams responses. Voice mode works locally.

| # | Title | Size | Assignee | Labels |
|---|-------|------|----------|--------|
| AG-03 | `curriculum_planner` CrewAI agent + `run_curriculum_planner()` | L | @ml-eng | `agents` `critical` |
| AG-04 | `project_builder` CrewAI agent + `run_project_builder()` | L | @ml-eng | `agents` |
| AG-05 | `evaluator_node` — LLM-as-Judge, score 0.0–1.0 | M | @ml-eng | `agents` `eval` |
| AG-06 | `run_reporter()` — weekly progress report | M | @ml-eng | `agents` |
| AG-08 | Verify all agents respect `max_iter` | S | @ml-eng | `guardrails` |
| UI-01 | Chainlit `on_chat_start` — greeting + initial state | S | @frontend | `ui` `critical` |
| UI-02 | Token streaming via `astream_events` v2 | M | @frontend | `ui` `critical` |
| UI-03 | HITL Approve/Revise action buttons | M | @frontend | `ui` `hitl` |
| UI-04 | `/report` command shortcut | S | @frontend | `ui` |
| UI-05 | Single graph instance at startup | S | @frontend | `ui` `perf` |
| UI-08 | `CHAINLIT_AUTH_SECRET` env variable | XS | @frontend | `security` |
| VOICE-01 | `/voice` command toggle | S | @ml-eng | `voice` |
| VOICE-02 | Whisper STT — record + transcribe locally | M | @ml-eng | `voice` |
| VOICE-03 | Kokoro ONNX TTS — generate + play | M | @ml-eng | `voice` |
| VOICE-05 | Auto-download Kokoro model on init | S | @ml-eng | `voice` |
| VOICE-06 | Cap TTS output at 500 chars | XS | @ml-eng | `voice` |
| GOV-02 | HITL trigger conditions in `evaluator_node` | M | @backend | `hitl` `critical` |
| GOV-03 | `hitl_node` — `interrupt()` + resume logic | M | @backend | `hitl` `critical` |
| OBS-01 | Structured JSON agent call logging | M | @devops | `observability` |
| OBS-03 | LangSmith tracing via env var | S | @devops | `observability` |
| MEM-03 | Save user notes per session | M | @backend | `memory` `finetune` |
| TOOL-02 | `analyze_kaggle_notebooks()` tool | M | @backend | `tools` |
| RAG-05 | GitHub README loader | M | @ml-eng | `rag` `github` |
| RAG-06 | Kaggle notebook loader | M | @ml-eng | `rag` `kaggle` |
| LG-06 | `error_recovery_node` — self-healing | M | @backend | `resilience` |

**Sprint 3 Points:** 56 | ⚠️ Largest sprint — split voice items to S4 if team is < 3 engineers.

---

### 🏃 SPRINT 4 (Jun 23 – Jul 6, 2026) — "Fine-Tuning + DevOps"
**Goal:** Fine-tuning pipeline end-to-end. Docker Compose production-ready. HITL state persisted.

| # | Title | Size | Assignee | Labels |
|---|-------|------|----------|--------|
| FT-01 | `prepare_training_data()` — notes → Alpaca JSONL | M | @ml-eng | `finetune` |
| FT-02 | `run_finetune()` — Unsloth + LoRA training | XL | @ml-eng | `finetune` `gpu` |
| FT-03 | GGUF export + Ollama Modelfile + `register_with_ollama()` | L | @ml-eng | `finetune` `ollama` |
| FT-04 | `finetune_check_node` — background thread trigger | M | @backend | `finetune` |
| FT-05 | `python scripts/run_finetune.py` CLI | S | @backend | `finetune` `cli` |
| VOICE-04 | Show STT transcript before submitting | S | @frontend | `voice` `ux` |
| GOV-05 | HITL state persisted in SqliteSaver | M | @backend | `hitl` `persistence` |
| DEP-01 | `docker-compose.yml` — Ollama + ChromaDB + Chainlit | M | @devops | `deploy` |
| DEP-02 | Ollama NVIDIA GPU reservation in compose | S | @devops | `deploy` `gpu` |
| DEP-03 | Non-root user in Dockerfile | S | @devops | `deploy` `security` |
| DEP-04 | Bind Ollama to localhost only | XS | @devops | `deploy` `security` `critical` |
| MEM-04 | Auto-generate session summaries at end of chat | M | @backend | `memory` |
| UI-06 | File upload → personal ChromaDB collection | M | @frontend | `ui` `rag` |
| OBS-02 | Prometheus metrics server | M | @devops | `observability` |
| TOOL-06 | GitHub API 1h TTL cache + exponential backoff | M | @backend | `tools` `resilience` |
| INF-05 | `pytest tests/unit/` with mocked LLMs | L | @qa | `testing` |

**Sprint 4 Points:** 56 | **Hard Dependency:** GPU machine must be available for FT-02

---

### 🏃 SPRINT 5 (Jul 7 – Jul 20, 2026) — "Hardening & Polish"
**Goal:** A/B testing pipeline, backup, advanced UI, and production readiness.

| # | Title | Size | Assignee | Labels |
|---|-------|------|----------|--------|
| FT-06 | A/B test new fine-tuned model vs. base before switch | L | @ml-eng | `finetune` `quality` |
| FT-07 | HITL notification when personalized model is ready | S | @frontend | `finetune` `ux` |
| DEP-05 | Daily cron backup of `data/` volumes | M | @devops | `deploy` `ops` |
| DEP-06 | Model upgrade path without full rebuild | S | @devops | `deploy` |
| RAG-07 | Prune ChromaDB collections > 30 days | S | @devops | `rag` `ops` |
| LG-08 | AsyncPostgresSaver swap via env var | M | @backend | `scalability` |
| UI-07 | Sidebar — current week + skill progress | L | @frontend | `ui` `ux` |
| OBS-04 | OpenTelemetry OTLP export | M | @devops | `observability` |
| INF-06 | GitHub Actions CI — lint + unit tests | M | @devops | `ci/cd` |
| LG-07 | Mermaid diagram export from compiled graph | S | @backend | `docs` |
| GOV-05 | HITL resume after page refresh | M | @backend | `hitl` |

**Sprint 5 Points:** 37 | **Focus:** Stability over features

---

### 🏃 SPRINT 6 (Jul 21 – Aug 3, 2026) — "Extension & Release"
**Goal:** MCP server, documentation update, load testing, and v1.0 release.

| # | Title | Size | Assignee | Labels |
|---|-------|------|----------|--------|
| TOOL-07 | Expose tool registry as MCP server | M | @backend | `mcp` `interop` |
| INF-06 | Full integration test suite | L | @qa | `testing` |
| — | Load test with 10 concurrent users | M | @devops | `perf` |
| — | Update architecture doc to match final implementation | M | @maimoon | `docs` |
| — | Record demo video and write README | M | @maimoon | `docs` |
| — | Tag `v1.0.0` release | XS | @maimoon | `release` |

**Sprint 6 Points:** 18 | **Goal:** Ship v1.0.0

---

## 5. Roadmap — Phased Timeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              AI COACH PROJECT ROADMAP — MAY–AUGUST 2026                        │
├──────────┬────────────────────────────────────────────────────────────────────┤
│          │  MAY 2026        │  JUN 2026        │  JUL 2026    │  AUG 2026   │
│ PHASE    │  W1   W2  │ W3   W4  │  W5   W6  │  W7   W8  │ W9  W10 │ W11  W12│
├──────────┼────────────────────────────────────────────────────────────────────┤
│ PHASE 1  │ ████████████████ │                                                │
│ Foundation│ Env · Schema  │                                                  │
│ (S1)     │ SQLite · ChromaDB│                                                │
├──────────┼────────────────────────────────────────────────────────────────────┤
│ PHASE 2  │               │ ████████████████ │                               │
│ Graph Core│              │ LangGraph · Graph │                               │
│ (S2)     │              │ Supervisor · RAG  │                               │
├──────────┼────────────────────────────────────────────────────────────────────┤
│ PHASE 3  │                              │ ████████████████ │                │
│ Agents +  │                             │ CrewAI Agents   │                 │
│ UI (S3)  │                             │ Chainlit · Voice │                 │
├──────────┼────────────────────────────────────────────────────────────────────┤
│ PHASE 4  │                                             │ ████████████████  │
│ Fine-tune │                                            │ LoRA · Docker     │
│ + DevOps  │                                            │ HITL · Observ.   │
│ (S4)     │                                            │                   │
├──────────┼────────────────────────────────────────────────────────────────────┤
│ PHASE 5  │                                                        │ ████████│
│ Hardening │                                                       │ A/B Test│
│ (S5-S6)  │                                                       │ v1.0 🚀  │
└──────────┴────────────────────────────────────────────────────────────────────┘
```

### 📍 Key Milestones

| Date | Milestone | Description | Success Criteria |
|------|-----------|-------------|-----------------|
| **May 25** | 🏁 M1: Foundation Complete | Env, schema, ChromaDB | `pytest tests/unit/` green; graph compiles |
| **Jun 8** | 🏁 M2: Graph Core | LangGraph routes correctly | Supervisor routes to correct agent in 3/3 test cases |
| **Jun 22** | 🏁 M3: MVP Demo | All agents + Chainlit UI | User can complete onboarding → get a learning plan |
| **Jul 6** | 🏁 M4: Fine-Tuning Live | LoRA pipeline + Docker | `ollama run coach-{id}` responds with personalized content |
| **Jul 20** | 🏁 M5: Production Ready | All hardening complete | 10 concurrent users; zero crashes in 1h soak test |
| **Aug 3** | 🚀 M6: v1.0 Release | Tagged release + docs | GitHub release published; README with demo video |

### 📊 Story Points Burndown (Target)

| Sprint | Points Planned | Points Remaining After Sprint |
|--------|---------------|-------------------------------|
| Start  | 161 | 161 |
| S1     | 14 | 147 |
| S2     | 44 | 103 |
| S3     | 56 | 47 |
| S4     | 56 | 0 (core feature complete) |
| S5     | 37 | — (hardening buffer) |
| S6     | 18 | — (release) |

---

## 6. My Items — Owner Task List

> **Filter:** `assignee:@maimoon` | This is your personal high-impact task list as the solo owner or lead.

### 🔥 This Week (May 11–17, 2026)

| Priority | Task | Points | Label | Due |
|---------|------|--------|-------|-----|
| 🔴 P1 | Create `.env.example` with all 15 variables | 1 | `setup` | May 12 |
| 🔴 P1 | Write `setup_ollama.sh` — pull 3 models + warmup | 2 | `setup` | May 12 |
| 🔴 P1 | Create `src/state/schema.py` — `CoachState` + all Pydantic models | 3 | `core` | May 14 |
| 🔴 P1 | Initialize SQLite schema via `LongTermMemory._init_schema()` | 2 | `memory` | May 14 |
| 🟠 P2 | Set up project folder structure (all `__init__.py` files) | 1 | `setup` | May 12 |
| 🟠 P2 | `ingest_documents()` — PDF/MD/TXT → ChromaDB pipeline | 3 | `rag` | May 16 |
| 🟡 P3 | Write `CONTRIBUTING.md` with setup guide | 1 | `docs` | May 17 |

### 📅 Next 2 Weeks (May 18 – Jun 1)

| Priority | Task | Points | Label |
|---------|------|--------|-------|
| 🔴 P1 | Implement LangGraph `StateGraph` with all nodes registered | 5 | `core` |
| 🔴 P1 | Conditional edge functions (`route_from_supervisor`, etc.) | 3 | `core` |
| 🔴 P1 | Graph compile with SqliteSaver + HITL interrupt | 3 | `core` |
| 🔴 P1 | `supervisor_node` — LLM routing + `_detect_phase()` | 3 | `agents` |
| 🔴 P1 | `analyze_github_profile()` tool + PyGitHub | 3 | `tools` |
| 🟠 P2 | `AgenticRetriever` — MMR + document grader + query rewriter | 6 | `rag` |
| 🟠 P2 | Input guardrails + tenacity retry | 2+2 | `security` |

### 🎯 Personal North Star Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end onboarding flow | < 60 seconds | Time from "Hello" to learning plan display |
| Profile analysis accuracy | > 80% skill match | Manual spot-check on 5 GitHub profiles |
| LLM response (8B, q4) | < 5s first token | `time curl` on Ollama API |
| Graph test coverage | > 80% | `pytest --cov` |
| Fine-tune cycle time | < 2 hours | From data prep to `ollama create` success |

---

## 7. Bug Tracker

> Pre-populated with known architectural risks as potential bugs to watch.

| ID | Title | Severity | Component | Status | Notes |
|----|-------|---------|-----------|--------|-------|
| BUG-01 | Ollama OOM crash on llama3.1:8b on 8GB VRAM | 🔴 Critical | Infra | Open | Use q3_K_S; set `OLLAMA_NUM_PARALLEL=1` |
| BUG-02 | ChromaDB collection empty error on cold start | 🟠 High | RAG | Open | Guard with existence check before query |
| BUG-03 | LangGraph `GraphRecursionError` on bad routing | 🔴 Critical | Graph | Open | `max_iterations` + cycle guard |
| BUG-04 | CrewAI Ollama `connection refused` on startup | 🟠 High | Agents | Open | `OPENAI_API_BASE` must be set before CrewAI init |
| BUG-05 | Whisper `CUDA error` when GPU is busy with Ollama | 🟡 Medium | Voice | Open | Force `WHISPER_DEVICE=cpu` |
| BUG-06 | JSON parse failure from CrewAI raw output | 🟠 High | Agents | Open | Regex fallback + default state return |
| BUG-07 | HITL state lost on page refresh | 🟡 Medium | HITL | Open | Checkpointer must persist before UI renders buttons |
| BUG-08 | Fine-tuned model GGUF path wrong in Modelfile | 🟡 Medium | Fine-tuning | Open | Use absolute path in `register_with_ollama()` |
| BUG-09 | `duckduckgo_search` rate-limited after rapid calls | 🟡 Medium | Tools | Open | Add 1s delay between searches |
| BUG-10 | GitHub API 403 on private repos | 🟢 Low | Tools | Open | Catch gracefully; only analyze public repos |

---

## 8. Definition of Done & Quality Gates

### ✅ Story Done When:

- [ ] Code written and passes `ruff` linting (0 errors)
- [ ] Unit test written and passing
- [ ] No hardcoded API keys or secrets
- [ ] PR reviewed by at least 1 other engineer (or self-reviewed with checklist for solo)
- [ ] All new env variables added to `.env.example`
- [ ] Relevant section of `AI_Coach_Architecture.md` updated if behavior changed
- [ ] Merged to `main` via PR (no direct pushes)

### 🏁 Milestone Done When:

- [ ] All sprint stories in `Done` column
- [ ] Integration test passes for the milestone's feature scope
- [ ] No open `🔴 Critical` bugs
- [ ] Demo-able to a non-technical stakeholder
- [ ] Architecture doc version bumped

### 🚀 Release v1.0 Done When:

- [ ] All 6 sprints complete
- [ ] 10 concurrent users soak test passed (60 minutes, zero crashes)
- [ ] `CHANGELOG.md` written
- [ ] GitHub Release tagged `v1.0.0`
- [ ] README has: badges, quickstart (< 5 commands), demo GIF/video
- [ ] Security checklist (Section 19 of Architecture Doc) 100% checked

---

## 9. Dependency Map

```
┌─────────────────────────────────────────────────────┐
│              DEPENDENCY GRAPH                       │
│                                                     │
│  INF-01 ──► INF-02 ──► INF-04 (Docker)            │
│     │                                               │
│     ▼                                               │
│  LG-01 (CoachState) ──────────────────────┐        │
│     │                                     │        │
│     ├──► LG-02 ──► LG-03 ──► LG-04       │        │
│     │        │                │           │        │
│     │        ▼                ▼           │        │
│     │    AG-01 (Supervisor)  HITL         │        │
│     │    AG-02 (Profile)     GOV-02       │        │
│     │    AG-03 (Curriculum)              │        │
│     │    AG-04 (Project)                 │        │
│     │    AG-05 (Evaluator)               │        │
│     │                                    │        │
│  RAG-01 ──► RAG-02 ──► RAG-03 ──► RAG-04│        │
│     │                                    │        │
│  MEM-02 ──► MEM-01 ──► MEM-03 ──► FT-01 │        │
│                              ──► FT-02 ──► FT-03   │
│                                          │        │
│  UI-01 ──► UI-02 ──► UI-03 (HITL UI)    │        │
│                                          │        │
└─────────────────────────────────────────────────────┘
```

**Hard Blockers:**
- Nothing can be tested end-to-end until `LG-04` (graph compiles)
- Fine-tuning (`FT-02`) requires GPU access — validate this **before** Sprint 4
- `AG-03` (Curriculum Planner) requires `RAG-02` (retriever) to be working

---

## 10. Team Roles & RACI

| Area | Responsible | Accountable | Consulted | Informed |
|------|------------|------------|----------|---------|
| Architecture decisions | @maimoon | @maimoon | Community / docs | — |
| LangGraph graph design | Backend | @maimoon | LangChain docs | QA |
| CrewAI agent design | ML Eng | @maimoon | CrewAI docs | Frontend |
| Ollama fine-tuning | ML Eng | @maimoon | Unsloth docs | DevOps |
| Chainlit UI | Frontend | @maimoon | Chainlit docs | ML Eng |
| Voice interface | ML Eng | @maimoon | Whisper/Kokoro | Frontend |
| Docker / DevOps | DevOps | @maimoon | — | All |
| Security review | All | @maimoon | — | — |
| Documentation | @maimoon | @maimoon | All | — |

> **Solo Project Note:** If this is a solo project, @maimoon wears all hats. In that case, the sprint structure above should be treated as a **personal Kanban** rather than a multi-person sprint, and velocity targets scaled down by ~40% (plan for ~60% of points per sprint to account for context-switching overhead).

---

## 📌 GitHub Project Setup Instructions

### Step 1: Populate the Backlog View
Add each story as a GitHub Issue using this naming convention:
```
[EPIC-ID] User story title
e.g.: [LG-01] Define CoachState TypedDict + all Pydantic sub-models
```
Labels to create: `critical`, `infra`, `core`, `agents`, `rag`, `memory`, `tools`, `voice`, `finetune`, `ui`, `hitl`, `security`, `observability`, `deploy`, `testing`, `docs`

### Step 2: Set Up Custom Fields
In GitHub Projects → Settings → Custom fields, add:
- `Story Points` (number): 1, 2, 3, 5, 8, 13
- `Epic` (single select): INF, LG, AG, RAG, MEM, TOOL, VOICE, FT, UI, GOV, OBS, DEP
- `Sprint` (single select): S1 through S6
- `Size` (single select): XS, S, M, L, XL, XXL

### Step 3: Roadmap View
Set `Date field` to `Sprint` (create a custom date field) and map:
- S1: May 12 – May 25
- S2: May 26 – Jun 8
- S3: Jun 9 – Jun 22
- S4: Jun 23 – Jul 6
- S5: Jul 7 – Jul 20
- S6: Jul 21 – Aug 3

### Step 4: Priority Board
Group by `Priority` field (Critical / High / Medium / Low) with columns matching.

### Step 5: Workflows
Enable these built-in workflows:
1. Auto-add issues labeled with project label → Backlog
2. When PR is merged → move linked issue to Done
3. When issue is reopened → move to In Progress

---

*Document version 1.0 — Generated May 11, 2026 — Ready for team execution*
