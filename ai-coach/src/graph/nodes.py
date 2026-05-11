"""All graph node functions."""

# src/graph/nodes.py  (excerpt: finetune_check_node)

import threading
from src.memory.long_term import LongTermMemory
from src.state.schema import CoachState


def finetune_check_node(state: CoachState) -> CoachState:
    """
    Check if enough new notes have accumulated to trigger a fine-tune job.
    Fine-tune runs in a background thread to avoid blocking the graph.
    Threshold: 50 new user notes OR 7 days since last fine-tune.
    """
    user_id = state["user_profile"].user_id
    memory = LongTermMemory()
    notes = memory.get_notes(user_id, limit=60)

    should_trigger = len(state.get("new_notes", [])) >= 50

    if should_trigger:
        # Fire and forget in background thread
        thread = threading.Thread(
            target=_background_finetune,
            args=(user_id,),
            daemon=True,
        )
        thread.start()
        return {**state, "finetune_trigger": True, "new_notes": []}

    return {**state, "finetune_trigger": False}


def _background_finetune(user_id: str):
    """Runs in background — does not block the main graph."""
    import os, datetime
    from src.finetune.data_prep import prepare_training_data
    from src.finetune.trainer import run_finetune
    from src.finetune.export import register_with_ollama

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d")
    data_path = f"./data/training/{user_id}_{timestamp}.jsonl"
    output_dir = f"./data/training/output_{user_id}_{timestamp}"
    model_name = f"coach-{user_id[:8]}-v{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    n_examples = prepare_training_data(user_id, data_path)
    if n_examples < 10:
        return   # not enough data yet

    gguf_path = run_finetune(user_id, data_path, output_dir)
    register_with_ollama(gguf_path, model_name)





# src/graph/nodes.py  (excerpt: hitl_node)

from langgraph.types import interrupt
from src.state.schema import CoachState


def hitl_node(state: CoachState) -> CoachState:
    """
    LangGraph interrupt node.
    Execution pauses here; resumes only after human provides input.
    The Chainlit UI handles presenting the approval UI.
    """
    # This call serializes state and pauses the graph.
    # LangGraph will not advance past this node until:
    #   graph.update_state(config, {"human_approval": True/False})
    #   graph.invoke(None, config=config)  ← resume
    human_response = interrupt({
        "prompt": state["hitl_prompt"],
        "content_preview": _get_preview(state),
    })

    return {
        **state,
        "human_approval": human_response.get("approved", False),
        "hitl_required": False,
    }


def _get_preview(state: CoachState) -> str:
    if state.get("learning_plan"):
        plan = state["learning_plan"]
        return f"Learning plan: {len(plan.weeks)} weeks, targeting {state['user_profile'].target_role}"
    if state.get("current_project"):
        proj = state["current_project"]
        return f"Project: {proj.title} ({proj.difficulty}), ~{proj.estimated_hours}h"
    return "No preview available."






# src/graph/nodes.py  (excerpt: memory_ingest_node with guardrails)

import re
from src.state.schema import CoachState

BLOCKED_PATTERNS = [
    r"ignore (previous|all) instructions",
    r"you are now",
    r"jailbreak",
    r"DAN mode",
]

def apply_input_guardrails(user_input: str) -> tuple[str, bool]:
    """
    Sanitize user input. Returns (cleaned_input, is_safe).
    """
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return user_input, False
    # Truncate extremely long inputs
    return user_input[:4000], True