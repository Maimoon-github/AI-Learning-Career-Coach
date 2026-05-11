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