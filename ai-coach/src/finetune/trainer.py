"""Unsloth + PEFT LoRA fine-tuning."""

# src/finetune/trainer.py

import os
from pathlib import Path


def run_finetune(user_id: str, data_path: str, output_dir: str) -> str:
    """
    Fine-tune a small local model using Unsloth + LoRA.
    Requires GPU. Returns path to GGUF file.

    NOTE: Unsloth must be imported in a GPU-enabled environment.
    This function is designed to be called from a separate process
    or background job scheduler.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # ── Load base model via Unsloth ──────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        dtype=None,        # auto-detect
        load_in_4bit=True, # QLoRA
    )

    # ── Apply LoRA adapters ───────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Load dataset ──────────────────────────────────────────────
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_alpaca(examples):
        texts = []
        for instr, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if inp:
                text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
            texts.append(text + tokenizer.eos_token)
        return {"text": texts}

    dataset = dataset.map(format_alpaca, batched=True)

    # ── Training arguments ────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )
    trainer.train()

    # ── Export to GGUF for Ollama ─────────────────────────────────
    gguf_path = os.path.join(output_dir, "model.gguf")
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")

    return gguf_path