# Personalized AI Learning & Career Coach — Architecture Document

## System Overview

A production-grade multi-agent system that delivers adaptive, personalized learning paths and career coaching by orchestrating **LangGraph** (workflow state machine) with **CrewAI** (collaborative agent teams) against a locally-served **Ollama** LLM.

---

## Architectural Pillars

| Pillar | Implementation |
|---|---|
| **Reliability** | LangGraph checkpointing (Redis/Postgres), retry with exponential backoff, Pydantic output validation |
| **Debuggability** | LangSmith tracing, structured JSON logs via structlog, per-node state snapshots |
| **Maintainability** | Separation of concerns: state / nodes / crews / tools / services each in isolated modules |
| **Cost/Performance** | LLM routing by task type (small model for extraction, large for reasoning), tool result caching, async fan-out for parallel crew execution |

---

## Workflow State Machine

```
START
  └─► profile_ingestion ──(fail)──► END
              │
              ▼
      skill_assessment
              │
              ▼
       learning_path ◄────────────────────────┐
         │         │                           │
         ▼         ▼                           │
  project_generation  llm_fine_tuning          │
         │         │                           │
         └────┬────┘                           │
              ▼                                │
      progress_report                          │
              │                                │
              ▼                                │
            hitl ──(end)──► END               │
              │                                │
         (approve/revise)                      │
              ▼                                │
        advance_week ──────────────────────────┘
```

### HITL Contract
The `hitl` node uses LangGraph's `interrupt()` primitive to suspend execution. The host application resumes the graph by calling `app.astream()` with `{"hitl_action": "approve"|"revise"|"end", "user_feedback": str|None}`.

---

## Agent / Crew Design

### LangGraph Nodes → CrewAI Crews Mapping

| Node | Crew | Process | Key Agents |
|---|---|---|---|
| `profile_ingestion` | `ProfileAnalysisCrew` | Sequential | GitHub Analyst, Kaggle Analyst, Document Processor, Profile Synthesizer |
| `skill_assessment` | `SkillGapAssessmentCrew` | Sequential | Role Definition Agent, Gap Analyst |
| `learning_path` | `LearningPathGenerationCrew` | Sequential | Curriculum Designer, Resource Curator, Path Optimizer |
| `project_generation` | `ProjectGenerationCrew` ×3 (parallel) | Sequential | Project Ideator, Specification Writer, Difficulty Adjuster |
| `llm_fine_tuning` | `LLMFineTuningCrew` | Sequential | Data Preparer, Fine-tuning Orchestrator, Model Evaluator |
| `progress_report` | `ProgressReportingCrew` | Sequential | Data Aggregator, Report Generator, Motivational Coach |

### Parallel Execution
`project_generation` and `llm_fine_tuning` run concurrently via `asyncio.gather()` within their respective LangGraph nodes. Both edges from `learning_path` fire simultaneously; LangGraph waits for both before allowing `progress_report` to execute.

---

## State Schema (`AgentState`)

```python
class AgentState(TypedDict):
    # Identity
    user_id: str; target_role: str; session_id: str
    # Inputs
    github_profile_url: str | None
    kaggle_username: str | None
    uploaded_document_paths: list[str]
    session_notes: list[str]
    # Crew outputs (all serialized as dict for LangGraph compatibility)
    skill_profile: dict | None
    skill_gaps: list[dict]
    learning_path: dict | None
    practice_projects: list[dict]
    fine_tuning_status: str | None
    weekly_report: dict | None
    # Workflow control
    current_week: int; revision_cycle: int
    hitl_action: str | None; user_feedback: str | None
    error_context: dict | None
    # Conversation (add_messages reducer for append semantics)
    messages: Annotated[list[BaseMessage], add_messages]
```

---

## LLM Routing Strategy

| Task Type | Model | Rationale |
|---|---|---|
| `structured_extraction` | `llama3.2:3b` | Fast, sufficient for JSON extraction |
| `gap_analysis` | `llama3.2:3b` | Analytical comparison, structured output |
| `curriculum_design` | `llama3.1:70b` | Complex multi-week planning requires reasoning |
| `project_generation` | `llama3.1:70b` | Creative + technical spec writing |
| `report_generation` | `llama3.2:3b` | Template-driven, low complexity |
| `motivational_framing` | `llama3.2:3b` (temp=0.7) | Creative text, warm tone |

---

## Failure Handling

```
Error Type          │ Handler
────────────────────┼──────────────────────────────────────────────
ToolExecutionError  │ Retry ×3 with exponential backoff (2ˢ delay)
CrewExecutionError  │ Retry ×3; escalate to HITL on persistent failure
OllamaConnectionError│ Raise immediately; surface in error_context
ValidationError     │ Re-prompt agent with correction instruction
HITLTimeoutError    │ Auto-approve after configurable timeout (default: off)
Profile failure     │ Route to END via conditional edge
```

---

## Observability Stack

- **Tracing**: LangSmith (set `LANGCHAIN_TRACING_V2=true`) for full node + LLM call traces
- **Metrics**: Prometheus endpoint on `:9090` — latency per node, success rates, token usage
- **Logging**: structlog → JSON in production, console in development
- **Checkpoints**: LangGraph state snapshots in Redis (production) or in-memory (development)

---

## Deployment

```
docker compose up          # Full stack: app + Redis + Postgres
scripts/setup_ollama.sh    # Install Ollama + pull models (run on host)
scripts/deploy.sh all      # Build → test → push → deploy
```

### Environment Variables (required)
```
OLLAMA_BASE_URL     Ollama endpoint (default: http://localhost:11434)
GITHUB_TOKEN        GitHub PAT for profile analysis
DATABASE_URL        PostgreSQL connection string
REDIS_URL           Redis connection string
LANGCHAIN_API_KEY   LangSmith API key (optional but recommended)
```
