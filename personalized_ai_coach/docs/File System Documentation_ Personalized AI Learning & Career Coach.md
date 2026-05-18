# File System Documentation: Personalized AI Learning & Career Coach

This document provides a comprehensive enumeration and detailed description of every file and directory within the Personalized AI Learning & Career Coach project. The structure is designed to support a production-grade, multi-agent system leveraging LangGraph and CrewAI [1] [2].

## Root Directory

| File | Type | Description |
| :--- | :--- | :--- |
| `.env` | Configuration | Stores sensitive environment variables including API keys (GitHub, Kaggle, OpenAI), database connection strings, and local Ollama service endpoints. |
| `README.md` | Documentation | Serves as the primary project guide, containing installation steps, environment setup, execution commands, and high-level architectural overview. |
| `requirements.txt` | Dependency | Lists all Python packages required for the project, including `langgraph`, `crewai`, `langchain`, `pydantic`, and service-specific SDKs. |
| `main.py` | Entry Point | The central execution script that initializes the LangGraph state machine, sets up the voice interface, and starts the application loop. |

## Configuration (`config/`)

| File | Type | Description |
| :--- | :--- | :--- |
| `agents.yaml` | Definition | Contains the role, goal, and backstory for every CrewAI agent, ensuring consistent persona management across different crews [4]. |
| `tasks.yaml` | Definition | Defines the descriptions and expected outputs for all CrewAI tasks, providing the operational logic for agent collaboration. |
| `llm_config.yaml` | Configuration | Specifies model parameters for the local Ollama instance, including temperature, context window size, and model versioning [5]. |
| `system_settings.yaml` | Configuration | Defines global system parameters such as HITL gate thresholds, retry limits, and observability logging levels. |

## Source Code (`src/`)

### LangGraph Workflow (`src/langgraph_workflow/`)

| File | Type | Description |
| :--- | :--- | :--- |
| `graph.py` | Logic | Defines the LangGraph state machine structure, including the registration of nodes, edges, and conditional routing logic [3]. |
| `state.py` | Schema | Implements the `AgentState` schema using Pydantic or TypedDict to maintain persistence and consistency throughout the workflow. |
| `nodes/profile_ingestion_node.py` | Node | Logic for the initial data ingestion phase; coordinates the extraction of data from GitHub and Kaggle. |
| `nodes/skill_assessment_node.py` | Node | Logic for identifying skill gaps; triggers the Skill Gap Assessment Crew. |
| `nodes/learning_path_node.py` | Node | Logic for dynamic curriculum generation; manages the iterative updates to the learning path. |
| `nodes/project_generation_node.py` | Node | Logic for creating hands-on practice projects tailored to the user's current skill level. |
| `nodes/llm_fine_tuning_node.py` | Node | Orchestrates the fine-tuning process for the local Ollama model using user-specific notes. |
| `nodes/progress_report_node.py` | Node | Aggregates metrics and generates the weekly progress report for the user. |
| `nodes/hitl_node.py` | Node | Implements Human-in-the-Loop breakpoints, allowing the system to pause for user approval or feedback. |

### CrewAI Agents (`src/crewai_agents/`)

| File | Type | Description |
| :--- | :--- | :--- |
| `profile_analysis_crew.py` | Crew | Orchestrates the GitHub Analyst, Kaggle Analyst, and Document Processor agents to create a user skill profile. |
| `skill_gap_assessment_crew.py` | Crew | Manages agents comparing the user's profile against target role requirements to identify deficiencies. |
| `learning_path_generation_crew.py` | Crew | Coordinates the Curriculum Designer and Resource Curator agents to build the weekly learning plan. |
| `project_generation_crew.py` | Crew | Manages the generation of technical specifications and difficulty adjustments for practice projects. |
| `llm_fine_tuning_crew.py` | Crew | Handles the data preparation and monitoring tasks associated with fine-tuning the local LLM. |
| `progress_reporting_crew.py` | Crew | Aggregates progress data and applies motivational framing to the final user reports. |

### Custom Tools (`src/tools/`)

| File | Type | Description |
| :--- | :--- | :--- |
| `github_tool.py` | Tool | Interface for interacting with the GitHub API to retrieve repository data, languages, and contribution history. |
| `kaggle_tool.py` | Tool | Interface for the Kaggle API to analyze notebooks and competition performance. |
| `web_search_tool.py` | Tool | A generalized search tool (e.g., DuckDuckGo or Tavily) for resource curation and market research. |
| `document_parser_tool.py` | Tool | Logic for parsing PDF, Markdown, and text files to extract structured information from user uploads. |
| `ollama_tool.py` | Tool | Direct API interface for managing local Ollama model states and triggering fine-tuning jobs. |

### Services & Utilities (`src/services/` & `src/utils/`)

| File | Type | Description |
| :--- | :--- | :--- |
| `voice_interface/stt_service.py` | Service | Implements Speech-to-Text capabilities for the full-duplex voice interface. |
| `voice_interface/tts_service.py` | Service | Implements Text-to-Speech capabilities for natural-sounding agent responses. |
| `database/db_manager.py` | Service | Manages connections and queries for the persistent storage of LangGraph checkpoints. |
| `utils/data_preprocessing.py` | Utility | Contains helper functions for cleaning and formatting raw data before agent processing. |
| `utils/llm_client.py` | Utility | Provides a centralized, singleton client for all interactions with the local Ollama LLM. |
| `utils/error_handling.py` | Utility | Defines custom exceptions and the global error taxonomy for system-wide resilience. |

## Models & Data (`src/models/`)

| File | Type | Description |
| :--- | :--- | :--- |
| `skill_profile_model.py` | Model | Pydantic schema for the structured representation of a user's skills and experience. |
| `learning_path_model.py` | Model | Pydantic schema for the dynamic, week-by-week learning curriculum. |
| `project_model.py` | Model | Pydantic schema for practice project specifications and success criteria. |

## Support & Infrastructure

| File | Type | Description |
| :--- | :--- | :--- |
| `tests/unit/` | Test | Contains unit tests for individual nodes, agents, and tools to ensure isolated correctness. |
| `tests/integration/` | Test | Contains end-to-end tests for full LangGraph cycles and CrewAI collaboration flows. |
| `scripts/setup_ollama.sh` | Script | Automates the installation and configuration of the Ollama service on the local machine. |
| `scripts/deploy.sh` | Script | Handles the containerization and deployment of the application to production environments. |
| `docs/architecture_document.md` | Document | The foundational design document detailing the agentic workflow and system pillars. |
| `notebooks/ollama_fine_tuning_experiment.ipynb` | Notebook | A research environment for testing fine-tuning hyperparameters and data preparation strategies. |

## References

[1] Khatib, M. (2025). *Combining LangGraph and CrewAI*. Medium. [https://medium.com/@mayadakhatib/combining-langgraph-and-crewai-bf38c719ab27](https://medium.com/@mayadakhatib/combining-langgraph-and-crewai-bf38c719ab27)
[2] myengineeringpath.dev. (2026). *LangGraph vs CrewAI — Graph Orchestration or Role-Based Teams*. [https://myengineeringpath.dev/tools/langgraph-vs-crewai/](https://myengineeringpath.dev/tools/langgraph-vs-crewai/)
[3] LangChain. *Persistence - Docs by LangChain*. [https://docs.langchain.com/oss/python/langgraph/persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
[4] CrewAI. *Agents - CrewAI Documentation*. [https://docs.crewai.com/en/concepts/agents](https://docs.crewai.com/en/concepts/agents)
[5] Pankaj, P. (2025). *Building a Dynamic, Parallel Tool-Calling Agent with LangGraph + Ollama*. LinkedIn. [https://www.linkedin.com/pulse/building-dynamic-parallel-tool-calling-agent-langgraph-prabhat-pankaj-wdhxc](https://www.linkedin.com/pulse/building-dynamic-parallel-tool-calling-agent-langgraph-prabhat-pankaj-wdhxc)
