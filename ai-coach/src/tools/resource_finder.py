"""Resource finder tool."""

# src/tools/resource_finder.py

from crewai_tools import tool
from src.memory.vector_store import AgenticRetriever


@tool("find_learning_resources")
def find_learning_resources(topic: str, level: str = "intermediate") -> list[dict]:
    """
    Search the internal knowledge base for learning resources on a topic.
    topic: The subject to find resources for.
    level: 'beginner', 'intermediate', or 'advanced'.
    Returns matching resources from the RAG vector store.
    """
    retriever = AgenticRetriever()
    query = f"{level} {topic} tutorial course resource"
    docs = retriever.retrieve(query)
    return docs[:5]