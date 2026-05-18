#!/bin/bash
# -----------------------------------------------------------------------------
# Script: create_ai_coach_structure.sh
# Description: Generates the complete project hierarchy for the Personalized AI Coach.
#              Creates all directories and empty placeholder files as defined
#              in the architectural layout.
# Usage: ./create_ai_coach_structure.sh
# -----------------------------------------------------------------------------

set -e  # Exit immediately if any command fails

# Root directory of the project
PROJECT_ROOT="personalized_ai_coach"

# -----------------------------------------------------------------------------
# Create all required directories
# -----------------------------------------------------------------------------
create_directories() {
    local dirs=(
        "$PROJECT_ROOT/config"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes"
        "$PROJECT_ROOT/src/crewai_agents"
        "$PROJECT_ROOT/src/tools"
        "$PROJECT_ROOT/src/services/voice_interface"
        "$PROJECT_ROOT/src/services/database"
        "$PROJECT_ROOT/src/services/storage"
        "$PROJECT_ROOT/src/utils"
        "$PROJECT_ROOT/src/models"
        "$PROJECT_ROOT/tests/unit"
        "$PROJECT_ROOT/tests/integration"
        "$PROJECT_ROOT/scripts"
        "$PROJECT_ROOT/docs"
        "$PROJECT_ROOT/notebooks"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
}

# -----------------------------------------------------------------------------
# Create all files (empty placeholders)
# -----------------------------------------------------------------------------
create_files() {
    local files=(
        "$PROJECT_ROOT/.env"
        "$PROJECT_ROOT/README.md"
        "$PROJECT_ROOT/requirements.txt"
        "$PROJECT_ROOT/main.py"
        "$PROJECT_ROOT/config/__init__.py"
        "$PROJECT_ROOT/config/agents.yaml"
        "$PROJECT_ROOT/config/tasks.yaml"
        "$PROJECT_ROOT/config/llm_config.yaml"
        "$PROJECT_ROOT/config/system_settings.yaml"
        "$PROJECT_ROOT/src/__init__.py"
        "$PROJECT_ROOT/src/langgraph_workflow/__init__.py"
        "$PROJECT_ROOT/src/langgraph_workflow/graph.py"
        "$PROJECT_ROOT/src/langgraph_workflow/state.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/__init__.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/profile_ingestion_node.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/skill_assessment_node.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/learning_path_node.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/project_generation_node.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/llm_fine_tuning_node.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/progress_report_node.py"
        "$PROJECT_ROOT/src/langgraph_workflow/nodes/hitl_node.py"
        "$PROJECT_ROOT/src/crewai_agents/__init__.py"
        "$PROJECT_ROOT/src/crewai_agents/profile_analysis_crew.py"
        "$PROJECT_ROOT/src/crewai_agents/skill_gap_assessment_crew.py"
        "$PROJECT_ROOT/src/crewai_agents/learning_path_generation_crew.py"
        "$PROJECT_ROOT/src/crewai_agents/project_generation_crew.py"
        "$PROJECT_ROOT/src/crewai_agents/llm_fine_tuning_crew.py"
        "$PROJECT_ROOT/src/crewai_agents/progress_reporting_crew.py"
        "$PROJECT_ROOT/src/tools/__init__.py"
        "$PROJECT_ROOT/src/tools/github_tool.py"
        "$PROJECT_ROOT/src/tools/kaggle_tool.py"
        "$PROJECT_ROOT/src/tools/web_search_tool.py"
        "$PROJECT_ROOT/src/tools/document_parser_tool.py"
        "$PROJECT_ROOT/src/tools/ollama_tool.py"
        "$PROJECT_ROOT/src/services/__init__.py"
        "$PROJECT_ROOT/src/services/voice_interface/__init__.py"
        "$PROJECT_ROOT/src/services/voice_interface/stt_service.py"
        "$PROJECT_ROOT/src/services/voice_interface/tts_service.py"
        "$PROJECT_ROOT/src/services/voice_interface/audio_stream_handler.py"
        "$PROJECT_ROOT/src/services/database/__init__.py"
        "$PROJECT_ROOT/src/services/database/db_manager.py"
        "$PROJECT_ROOT/src/services/storage/__init__.py"
        "$PROJECT_ROOT/src/services/storage/s3_manager.py"
        "$PROJECT_ROOT/src/utils/__init__.py"
        "$PROJECT_ROOT/src/utils/data_preprocessing.py"
        "$PROJECT_ROOT/src/utils/llm_client.py"
        "$PROJECT_ROOT/src/utils/error_handling.py"
        "$PROJECT_ROOT/src/models/__init__.py"
        "$PROJECT_ROOT/src/models/skill_profile_model.py"
        "$PROJECT_ROOT/src/models/learning_path_model.py"
        "$PROJECT_ROOT/src/models/project_model.py"
        "$PROJECT_ROOT/tests/__init__.py"
        "$PROJECT_ROOT/tests/unit/__init__.py"
        "$PROJECT_ROOT/tests/unit/test_langgraph_nodes.py"
        "$PROJECT_ROOT/tests/unit/test_crewai_agents.py"
        "$PROJECT_ROOT/tests/unit/test_tools.py"
        "$PROJECT_ROOT/tests/integration/__init__.py"
        "$PROJECT_ROOT/tests/integration/test_full_workflow.py"
        "$PROJECT_ROOT/scripts/__init__.py"
        "$PROJECT_ROOT/scripts/setup_ollama.sh"
        "$PROJECT_ROOT/scripts/deploy.sh"
        "$PROJECT_ROOT/docs/architecture_document.md"
        "$PROJECT_ROOT/notebooks/__init__.py"
        "$PROJECT_ROOT/notebooks/ollama_fine_tuning_experiment.ipynb"
    )

    for file in "${files[@]}"; do
        # Ensure parent directory exists (redundant but safe)
        mkdir -p "$(dirname "$file")"
        touch "$file"
    done
}

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
echo "Creating project structure for Personalized AI Coach..."
create_directories
create_files
echo "✅ Project structure created successfully under ./$PROJECT_ROOT"