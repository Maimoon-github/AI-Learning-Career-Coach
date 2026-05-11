"""In-graph message window."""

# src/memory/short_term.py

from src.state.schema import CoachState


def sliding_window_memory(state: CoachState) -> CoachState:
    """
    Keep only the last N messages for the LLM context.
    N = 10 (5 user, 5 assistant).
    """
    messages = state["messages"]
    window_size = 10

    if len(messages) > window_size:
        # Keep last N messages (alternating user/assistant)
        recent = messages[-window_size:]
    else:
        recent = messages

    return {**state, "messages": recent}

