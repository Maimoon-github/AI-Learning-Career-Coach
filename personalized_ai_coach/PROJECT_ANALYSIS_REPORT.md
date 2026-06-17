# PROJECT ANALYSIS REPORT: Personalized AI Coach

## 1. Project Overview & Architecture

### Purpose & Goals
The **Personalized AI Coach** is a production-grade, multi-agent AI system designed to provide hyper-personalized career coaching, skill gap analysis, and tailored learning roadmaps. It leverages a student's digital footprint (GitHub, Kaggle, Resumes) to create a dynamic "L5-to-L5" or "Junior-to-Senior" upskilling experience.

### Technology Stack
*   **Core Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (v0.2.2+) for cyclical workflows and Human-in-the-Loop (HITL).
*   **Multi-Agent Framework**: [CrewAI](https://github.com/joaomaneiro/crewai) (v1.x) for specialized functional crews.
*   **LLM Provider**: [Ollama](https://ollama.com/) (Local-first) with model routing (Llama 3.2 3B for extraction, Llama 3.1 70B for strategy).
*   **Web Interfaces**: [Streamlit](https://streamlit.io/) (User Dashboard) and [FastAPI](https://fastapi.tiangolo.com/) (System API).
*   **Persistence**: [PostgreSQL](https://www.postgresql.org/) (via SQLAlchemy) for user data and LangGraph checkpointers.
*   **Voice Interface**: OpenAI Whisper (STT) and ElevenLabs/Coqui (TTS) for conversational interaction.
*   **Observability**: Structlog (JSON logging) and Prometheus metrics.

### Architectural Patterns
The system uses a **Hybrid Orchestration** pattern:
1.  **LangGraph** acts as the high-level state machine (the "Brain"), managing long-running sessions, branching logic, and persistence.
2.  **CrewAI** manages the "Hands" — execution-heavy tasks where multiple specialized agents collaborate (e.g., Profile Analysis Crew).
3.  **Asynchronous Execution**: Deeply integrated with `asyncio`, utilizing `run_in_executor` to bridge synchronous CrewAI logic into the async LangGraph event loop.

---

## 2. Directory Structure & File Inventory

```text
personalized_ai_coach/
├── app.py                  # Streamlit Dashboard (Primary User Interface)
├── main.py                 # FastAPI Entry point & Voice Bridge
├── config/                 # YAML-driven Agent & Task definitions
│   ├── agents.yaml         # Agent roles, goals, and backstories
│   ├── tasks.yaml          # Task descriptions and expected outputs
│   └── llm_config.yaml     # Model routing & hyperparameter settings
├── src/
│   ├── langgraph_workflow/ # Graph topology, state management, and nodes
│   ├── crewai_agents/      # Crew definitions (Profile, Skill Gap, etc.)
│   ├── models/             # Pydantic schemas for data validation
│   ├── services/           # Database, S3, and Voice Interface logic
│   ├── tools/              # Custom Agent tools (GitHub, Kaggle, etc.)
│   └── utils/              # Client wrappers, Logging, Error handling
├── tests/                  # Pytest suite (Unit & Integration)
├── docs/                   # Architectural drawings and specifications
└── scripts/                # Deployment and setup scripts (Docker/Ollama)
```

---

## 3. Core Components Deep Dive

### Agents & Crews
Crews are defined in `src/crewai_agents/` and configured via `config/agents.yaml`.
*   **ProfileAnalysisCrew**: Orchestrates a `GitHub Analyst`, `Kaggle Analyst`, and `Document Processor`.
*   **SkillGapAssessmentCrew**: Compares user profile against market data (fetched via Web Search).
*   **LearningPathGenerationCrew**: Creates a week-by-week curriculum with curated resources.
*   **ProjectGenerationCrew**: Proposes technical projects to bridge specific gaps.

### LangGraph Workflow
The graph (`src/langgraph_workflow/graph.py`) defines a complex lifecycle:
*   **Nodes**: Each node represents a phase (e.g., `profile_ingestion_node`).
*   **State**: `AgentState` (`src/langgraph_workflow/state.py`) persists everything from skill scores to raw messages.
*   **HITL (Human-in-the-Loop)**: The `hitl_node` (`src/langgraph_workflow/nodes/hitl_node.py`) uses `interrupt` to pause execution, allowing users to "Approve" or "Revise" the generated plan.

### State Management & Persistence
*   **AgentState**: Uses `Annotated[list[BaseMessage], add_messages]` for conversation history and field-specific keys for crew outputs.
*   **Checkpointers**: Implements `AsyncPostgresSaver` for fault-tolerant workflow resumption. If the server crashes, the user session can resume exactly where it stopped.

### Custom Tools
*   **GitHubTool**: Uses `PyGithub` to pull real repo data (`src/tools/github_tool.py`).
*   **KaggleTool**: Accesses Kaggle API for ML proficiency signals (`src/tools/kaggle_tool.py`).
*   **DocumentParser**: Handles PDF/Docx using `unstructured` and `pypdf` (`src/tools/document_parser_tool.py`).

---

## 4. Invocation & Execution Mechanisms

| Mechanism | File | Description |
| :--- | :--- | :--- |
| **Streamlit UI** | `app.py` | Direct crew kickoff for interactive sessions. Best for visualization. |
| **FastAPI API** | `main.py` | Headless execution, health monitoring, and system integration. |
| **Voice Interface** | `main.py` | Bridges STT/TTS with LangGraph. Users can talk to the coach via terminal audio. |
| **CLI / Main** | `main.py` | Traditional startup for production deployments. |

---

## 5. Data Flow & User Journey

1.  **Ingestion**: User provides GitHub URL/Resume. `ProfileAnalysisCrew` generates a `SkillProfileModel`.
2.  **Analysis**: `SkillGapAssessmentCrew` compares profile to `target_role`.
3.  **Planning**: `LearningPathGenerationCrew` generates a multi-week roadmap.
4.  **Action**: `ProjectGenerationCrew` creates specialized project specs.
5.  **Review (HITL)**: Workflow pauses. User reviews the "Progress Report" via UI or Voice.
6.  **Loop**: If "Revise", user feedback is injected back into the `LearningPath` node. If "Approve", the `current_week` counter increments.

---

## 6. Error Handling & Reliability

### Exception Hierarchy (`src/utils/error_handling.py`)
*   🔴 **CrewExecutionError**: Critical failure in multi-agent logic.
*   🟡 **ValidationError**: Schema mismatch in LLM output (common in small local models).
*   🟡 **HITLTimeoutError**: User took too long to respond to a prompt.

### Reliability Mechanisms
*   **Tenacity Retries**: Nodes utilize `@async_retry_with_backoff` to handle transient Ollama timeouts or API rate limits.
*   **Model Routing**: Tasks requiring high reasoning (Strategy, Project Gen) use the 70B model, while extraction tasks use the 3B model to ensure reliability/speed.

---

## 7. Strengths, Weaknesses & Improvement Opportunities

### Strengths
*   **Local-First Privacy**: Use of Ollama allows the system to run on private infrastructure without sending sensitive resume data to external LLMs (except optional Whisper/ElevenLabs).
*   **Modular Architecture**: Clear separation between Agent Persona (YAML), Business Logic (Crews), and Workflow Flow (LangGraph).
*   **Persistence**: Robust Postgres checkpointer allows for true "long-term coaching" sessions that last weeks.

### Weaknesses & Risks
*   **🔴 Runtime Bottleneck**: CrewAI `kickoff()` is synchronous. While wrapped in executors, high concurrency could saturate thread pools.
*   **🟡 Model Fragility**: Llama 3.2 3B can struggle with valid JSON generation for complex Pydantic schemas (mitigated by `OllamaClient` cache and retries).
*   **🟢 Voice Dependency**: Voice interface requires Python 3.11 due to `TTS` library constraints, creating environments issues for users on newer versions.

### Improvement Opportunities
1.  **Vector Memory**: Re-enable CrewAI memory using a local `ChromaDB` or `Qdrant` instance (currently disabled because it defaults to OpenAI embeddings).
2.  **Real-time Progress Hook**: Add a callback in `LearningPathGenerationCrew` to update the DB as each week is generated, rather than waiting for the whole crew to finish.
3.  **Streaming UI**: Use Streamlit fragments or custom components to show real-time agent output during the long "Analysis" phase.

---

## 8. Executive Summary & Next Steps

### Executive Summary
The project is a **highly sophisticated AI application** that successfully bridges the gap between high-level workflow management (LangGraph) and low-level task execution (CrewAI). Its structural integrity is excellent, with thorough configuration management and robust error handling. It is ready for advanced deployment, specifically in environments prioritizing local AI execution.

### Recommended Next Steps (Prioritized)
1.  **High Priority**: Implement a `SchemaCorrectionLayer` in `OllamaClient` to automatically fix malformed JSON from the 3B model.
2.  **Medium Priority**: Migrate `CrewAI` calls to `kickoff_async()` (if using newer CrewAI versions) to reduce thread overhead.
3.  **Medium Priority**: Implement the "Weekly Revision Cycle" logic to allow the coach to adjust the roadmap based on actual project completion data.
4.  **Low Priority**: Containerize the STT/TTS services to resolve the Python 3.11/3.12 version conflict.

---
*Report generated by Antigravity Technical Analyst.*
