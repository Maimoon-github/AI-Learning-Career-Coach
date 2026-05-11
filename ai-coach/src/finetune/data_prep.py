"""Convert user notes -> training pairs."""

# src/finetune/data_prep.py

import json
from src.memory.long_term import LongTermMemory


ALPACA_TEMPLATE = {
    "instruction": "",
    "input": "",
    "output": "",
}


def prepare_training_data(user_id: str, output_path: str) -> int:
    """
    Convert user notes and session summaries into Alpaca-format JSONL
    for LoRA fine-tuning. Returns number of training examples created.
    """
    memory = LongTermMemory()
    notes = memory.get_notes(user_id, limit=500)

    examples = []
    for note in notes:
        if len(note["note"]) < 50:
            continue   # skip trivially short notes
        examples.append({
            "instruction": (
                f"Explain the concept of {note['topic']} in the way this specific user "
                f"understands best, matching their vocabulary and background."
            ),
            "input": "",
            "output": note["note"],
        })

    # Add Q&A pairs from session summaries
    summaries = []  # load from DB similarly
    for summary in summaries:
        examples.append({
            "instruction": "Summarize today's learning session with key takeaways.",
            "input": "",
            "output": summary,
        })

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return len(examples)