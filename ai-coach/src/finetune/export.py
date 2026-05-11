"""GGUF export + Ollama Modelfile."""

# src/finetune/export.py

import subprocess
import os


MODELFILE_TEMPLATE = """FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM \"""
You are a personalized AI learning coach. You deeply understand this specific learner's 
background, learning style, and goals. You explain concepts using examples and analogies 
that resonate with their particular knowledge base. You are encouraging, precise, and 
always guide them toward hands-on practice.
\"""
"""


def register_with_ollama(gguf_path: str, model_name: str) -> bool:
    """
    Create and register a fine-tuned model with Ollama.
    Returns True if successful.
    """
    model_dir = os.path.dirname(gguf_path)
    modelfile_path = os.path.join(model_dir, "Modelfile")

    with open(modelfile_path, "w") as f:
        f.write(MODELFILE_TEMPLATE.format(gguf_path=gguf_path))

    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Ollama error: {result.stderr}")
        return False

    print(f"✅ Model '{model_name}' registered with Ollama.")
    return True