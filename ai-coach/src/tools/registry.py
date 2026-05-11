"""Central tool registry."""

# src/tools/registry.py

from src.tools.github_tools import analyze_github_profile
from src.tools.kaggle_tools import analyze_kaggle_notebooks
from src.tools.web_search_tool import duckduckgo_search
from src.tools.resource_finder import find_learning_resources
from src.tools.code_executor import validate_project_spec

ALL_TOOLS = [
    analyze_github_profile,
    analyze_kaggle_notebooks,
    duckduckgo_search,
    find_learning_resources,
    validate_project_spec,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}