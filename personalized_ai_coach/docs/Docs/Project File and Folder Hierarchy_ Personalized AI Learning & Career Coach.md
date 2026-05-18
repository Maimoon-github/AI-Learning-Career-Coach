# Project File and Folder Hierarchy: Personalized AI Learning & Career Coach

Based on the comprehensive architecture document, the following file and folder hierarchy is proposed for the Personalized AI Learning & Career Coach system. This structure aims to promote modularity, maintainability, scalability, and clear separation of concerns, aligning with production-grade development practices.

```
personalized_ai_coach/
├── .env                           # Environment variables (API keys, database connections, etc.)
├── README.md                      # Project overview, setup instructions, and deployment guide
├── requirements.txt               # Python dependencies
├── main.py                        # Main application entry point, initializes LangGraph and voice interface
├── config/                        # Configuration files for agents, tools, and system settings
│   ├── __init__.py
│   ├── agents.yaml                # CrewAI agent definitions (roles, backstories, goals)
│   ├── tasks.yaml                 # CrewAI task definitions (descriptions, expected outputs)
│   ├── llm_config.yaml            # Ollama model settings, API endpoints
│   └── system_settings.yaml       # General system parameters, HITL thresholds
├── src/                           # Source code for the core application logic
│   ├── __init__.py
│   ├── langgraph_workflow/        # LangGraph orchestration and state management
│   │   ├── __init__.py
│   │   ├── graph.py               # LangGraph definition (nodes, edges, state)
│   │   ├── state.py               # Pydantic model/TypedDict for AgentState
│   │   └── nodes/                 # Individual LangGraph node implementations
│   │       ├── __init__.py
│   │       ├── profile_ingestion_node.py
│   │       ├── skill_assessment_node.py
│   │       ├── learning_path_node.py
│   │       ├── project_generation_node.py
│   │       ├── llm_fine_tuning_node.py
│   │       ├── progress_report_node.py
│   │       └── hitl_node.py
│   ├── crewai_agents/             # CrewAI agent definitions and crew compositions
│   │   ├── __init__.py
│   │   ├── profile_analysis_crew.py
│   │   ├── skill_gap_assessment_crew.py
│   │   ├── learning_path_generation_crew.py
│   │   ├── project_generation_crew.py
│   │   ├── llm_fine_tuning_crew.py
│   │   └── progress_reporting_crew.py
│   ├── tools/                     # Custom tools used by CrewAI agents and LangGraph nodes
│   │   ├── __init__.py
│   │   ├── github_tool.py         # GitHub API interaction
│   │   ├── kaggle_tool.py         # Kaggle API interaction
│   │   ├── web_search_tool.py     # General web search capabilities
│   │   ├── document_parser_tool.py# PDF/document parsing and extraction
│   │   └── ollama_tool.py         # Direct Ollama interactions (if not via langchain client)
│   ├── services/                  # External service integrations (voice, database, storage)
│   │   ├── __init__.py
│   │   ├── voice_interface/       # Full-duplex voice interface components
│   │   │   ├── __init__.py
│   │   │   ├── stt_service.py     # Speech-to-Text implementation
│   │   │   ├── tts_service.py     # Text-to-Speech implementation
│   │   │   └── audio_stream_handler.py # Manages audio input/output streams (e.g., WebRTC/Twilio integration)
│   │   ├── database/              # Database interaction layer (e.g., PostgreSQL for LangGraph checkpoints)
│   │   │   ├── __init__.py
│   │   │   └── db_manager.py
│   │   └── storage/               # Cloud storage integration (e.g., S3 for uploaded documents)
│   │       ├── __init__.py
│   │       └── s3_manager.py
│   ├── utils/                     # Utility functions and helpers
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py  # Data cleaning and formatting
│   │   ├── llm_client.py          # Centralized LLM client for Ollama
│   │   └── error_handling.py      # Custom exception classes and error handlers
│   └── models/                    # Pydantic models for structured data (e.g., skill profiles, learning paths)
│       ├── __init__.py
│       ├── skill_profile_model.py
│       ├── learning_path_model.py
│       └── project_model.py
├── tests/                         # Unit and integration tests
│   ├── __init__.py
│   ├── unit/                      # Unit tests for individual components
│   │   ├── __init__.py
│   │   ├── test_langgraph_nodes.py
│   │   ├── test_crewai_agents.py
│   │   └── test_tools.py
│   └── integration/               # Integration tests for workflows and crews
│       ├── __init__.py
│       └── test_full_workflow.py
├── scripts/                       # Helper scripts (e.g., setup, deployment, data migration)
│   ├── __init__.py
│   ├── setup_ollama.sh            # Script to set up Ollama locally
│   └── deploy.sh                  # Deployment script
├── docs/                          # Project documentation (API docs, design docs)
│   └── architecture_document.md   # The architecture document itself
└── notebooks/                     # Jupyter notebooks for experimentation, data analysis, or demos
    ├── __init__.py
    └── ollama_fine_tuning_experiment.ipynb
```

## Rationale for Hierarchy Design

*   **Top-Level Clarity**: The root directory contains essential project files (`.env`, `README.md`, `requirements.txt`, `main.py`) for quick project understanding and setup.
*   **Configuration Management (`config/`)**: Centralizes all configuration parameters, making it easy to manage agent definitions, LLM settings, and system-wide variables. Using YAML files promotes human readability and version control.
*   **Core Logic (`src/`)**: Encapsulates all primary application code. This directory is further subdivided to reflect the architectural components:
    *   **`langgraph_workflow/`**: Dedicated to LangGraph's state machine definition, state management, and individual node implementations. This separation ensures that the orchestration logic is distinct and manageable.
    *   **`crewai_agents/`**: Houses all CrewAI-related code, including the definition of agents and their respective crews. This aligns with CrewAI's role-based design and promotes reusability of agent definitions.
    *   **`tools/`**: Contains all custom tools that agents (both LangGraph and CrewAI) can utilize. This promotes a clear interface for tool development and integration.
    *   **`services/`**: Manages integrations with external services such as the voice interface, database, and cloud storage. This abstraction layer ensures that core logic is decoupled from specific service implementations.
    *   **`utils/`**: Provides a home for common utility functions, data preprocessing routines, and a centralized LLM client, preventing code duplication.
    *   **`models/`**: Stores Pydantic models or other structured data definitions, ensuring data consistency and validation across the system.
*   **Testing (`tests/`)**: Separates unit and integration tests, facilitating thorough testing and continuous integration practices.
*   **Scripts (`scripts/`)**: Contains executable scripts for common development and deployment tasks, enhancing automation.
*   **Documentation (`docs/`)**: A dedicated place for all project documentation, including the architecture document itself.
*   **Notebooks (`notebooks/`)**: Provides a flexible environment for experimentation, data analysis, and prototyping without cluttering the main codebase.

This hierarchy provides a clear roadmap for development, allowing different teams or individuals to work on distinct components with minimal interference, while ensuring a cohesive and well-organized project structure.
