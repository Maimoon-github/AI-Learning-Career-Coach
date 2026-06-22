from __future__ import annotations

import asyncio
import os
import json
import logging
from typing import Any, Optional, Union, List
from pathlib import Path

import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ollama import AsyncClient, ResponseError

# Existing error handling
from src.utils.error_handling import OllamaConnectionError

# Optional dependencies for fine-tuning
try:
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    HAS_TRAINING_LIBS = True
except ImportError:
    HAS_TRAINING_LIBS = False

log = structlog.get_logger(__name__)

class OllamaToolInput(BaseModel):
    action: str = Field(
        description="Action to perform: list_models | pull_model | delete_model | start_lora_training | create_model"
    )
    model_name: str = Field(
        default="", 
        description="Name of the model to pull, delete, or create (e.g. 'custom-coach')"
    )
    base_model: str = Field(
        default="llama3.1:8b", 
        description="Base model for training or creating (e.g. 'llama3.1:8b', 'unsloth/llama-3-8b-bnb-4bit')"
    )
    training_data: Optional[Union[str, List[str], List[dict]]] = Field(
        default=None, 
        description="Path to JSONL file or list of instruction-response pairs/notes for training"
    )
    output_path: str = Field(
        default="./fine_tuned_model", 
        description="Directory to save the trained adapter and Modelfile"
    )
    training_params: Optional[dict] = Field(
        default=None, 
        description="Optional overrides for training (rank, alpha, epochs, batch_size, etc.)"
    )

class OllamaTool(BaseTool):
    name: str = "ollama_manager"
    description: str = (
        "Advanced manager for local Ollama instances. Supports model lifecycle (list, pull, delete) "
        "and production-ready LoRA fine-tuning workflows using Unsloth/HF stack."
    )
    args_schema: type[BaseModel] = OllamaToolInput

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """Sync wrapper for CrewAI compatibility."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import threading
            
            result = {}
            def run_in_thread():
                nonlocal result
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(self._async_run(**kwargs))
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            return result
        else:
            return loop.run_until_complete(self._async_run(**kwargs))

    async def _async_run(self, **kwargs: Any) -> dict[str, Any]:
        """Main async execution logic using official Ollama SDK."""
        action = kwargs.get("action")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        client = AsyncClient(host=base_url)

        try:
            if action == "list_models":
                resp = await client.list()
                # Handle both SDK object response and raw dict
                models = []
                model_list = getattr(resp, 'models', resp.get('models', []))
                for m in model_list:
                    # Model name can be in 'model' or 'name' attribute/key
                    name = getattr(m, 'model', None) or getattr(m, 'name', None) or m.get('model') or m.get('name')
                    if name:
                        models.append(name)
                return {"status": "success", "models": models}

            elif action == "pull_model":
                model_name = kwargs.get("model_name")
                if not model_name:
                    return {"status": "error", "message": "model_name is required for pull_model"}
                
                log.info("pulling_model", model=model_name)
                # We use progress reporting if possible, but for simple tool output we just await
                await client.pull(model=model_name)
                return {"status": "success", "message": f"Model {model_name} pulled successfully"}

            elif action == "delete_model":
                model_name = kwargs.get("model_name")
                if not model_name:
                    return {"status": "error", "message": "model_name is required for delete_model"}
                
                await client.delete(model=model_name)
                return {"status": "success", "message": f"Model {model_name} deleted"}

            elif action == "start_lora_training":
                return await self._handle_training(kwargs)

            elif action == "create_model":
                return await self._handle_create_model(client, kwargs)

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except ResponseError as e:
            log.error("ollama_sdk_error", error=str(e))
            return {"status": "error", "message": f"Ollama SDK error: {e.error}"}
        except Exception as e:
            log.error("ollama_tool_unexpected_error", error=str(e))
            if "Connection" in str(e):
                raise OllamaConnectionError(f"Cannot connect to Ollama at {base_url}")
            return {"status": "error", "message": str(e)}

    async def _handle_training(self, kwargs: dict) -> dict[str, Any]:
        """Orchestrates the LoRA fine-tuning process."""
        if not HAS_TRAINING_LIBS:
            return {
                "status": "failed",
                "message": "Training dependencies (unsloth, torch, transformers) not found. Fine-tuning aborted."
            }

        training_data = kwargs.get("training_data")
        if not training_data:
            return {"status": "error", "message": "training_data is required for start_lora_training"}

        # Hardware Check
        if not torch.cuda.is_available():
            return {"status": "error", "message": "CUDA not available. GPU is required for fine-tuning."}
        
        gpu_stats = torch.cuda.get_device_properties(0)
        vram_gb = gpu_stats.total_memory / 1024**3
        log.info("gpu_detected", name=gpu_stats.name, vram=f"{vram_gb:.2f}GB")

        if vram_gb < 6:
            return {"status": "error", "message": f"Insufficient VRAM ({vram_gb:.2f}GB). Minimum 6GB required for 4-bit LoRA."}

        # Data Preparation
        dataset = self._prepare_dataset(training_data)
        if len(dataset) < 5: # Arbitrary small limit for example
            return {"status": "error", "message": f"Insufficient training data. Need at least 5 examples, got {len(dataset)}."}

        # Training Parameters
        params = kwargs.get("training_params") or {}
        model_name = kwargs.get("base_model", "unsloth/llama-3-8b-bnb-4bit")
        output_dir = kwargs.get("output_path", "./fine_tuned_model")

        try:
            # Load Model & Tokenizer
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = params.get("max_seq_length", 2048),
                load_in_4bit = True,
            )

            # Add LoRA Adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r = params.get("r", 16),
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha = params.get("lora_alpha", 16),
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
            )

            # SFT Trainer
            trainer = SFTTrainer(
                model = model,
                train_dataset = dataset,
                dataset_text_field = "text",
                max_seq_length = params.get("max_seq_length", 2048),
                args = TrainingArguments(
                    per_device_train_batch_size = params.get("batch_size", 2),
                    gradient_accumulation_steps = params.get("grad_accum", 4),
                    warmup_steps = 5,
                    max_steps = params.get("epochs", 1) * 10, # Simplified for example
                    learning_rate = 2e-4,
                    fp16 = not torch.cuda.is_bf16_supported(),
                    bf16 = torch.cuda.is_bf16_supported(),
                    logging_steps = 1,
                    output_dir = output_dir,
                ),
            )

            trainer.train()

            # Save LoRA Adapter
            adapter_path = os.path.join(output_dir, "adapter")
            model.save_pretrained_merged(adapter_path, tokenizer, save_method = "lora")
            
            # Generate Modelfile
            base_model = kwargs.get('base_model', 'llama3.1:8b')
            modelfile_content = f"FROM {base_model}\nADAPTER {os.path.abspath(adapter_path)}\n"
            modelfile_path = os.path.join(output_dir, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)

            # Register in Ollama
            new_model_name = kwargs.get("model_name") or f"fine-tuned-{base_model.replace(':', '-')}"
            log.info("registering_fine_tuned_model", name=new_model_name)
            
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            async with AsyncClient(host=base_url) as client:
                await client.create(model=new_model_name, modelfile=modelfile_content)

            return {
                "status": "success",
                "message": f"Fine-tuning completed and model '{new_model_name}' registered in Ollama.",
                "adapter_path": adapter_path,
                "modelfile": modelfile_path,
                "steps": trainer.state.global_step
            }

        except Exception as e:
            log.error("training_failed", error=str(e))
            return {"status": "error", "message": f"Training failed: {str(e)}"}

    async def _handle_create_model(self, client: AsyncClient, kwargs: dict) -> dict[str, Any]:
        """Registers a model in Ollama using a Modelfile."""
        model_name = kwargs.get("model_name")
        output_path = kwargs.get("output_path", "./fine_tuned_model")
        modelfile_path = os.path.join(output_path, "Modelfile")

        if not model_name:
            return {"status": "error", "message": "model_name is required for create_model"}

        if not os.path.exists(modelfile_path):
            # Try to generate one if it doesn't exist but we have an adapter
            adapter_path = os.path.join(output_path, "adapter")
            if os.path.exists(adapter_path):
                content = f"FROM {kwargs.get('base_model', 'llama3.1:8b')}\nADAPTER {os.path.abspath(adapter_path)}\n"
                with open(modelfile_path, "w") as f:
                    f.write(content)
            else:
                return {"status": "error", "message": f"Modelfile not found at {modelfile_path}"}

        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()

        log.info("creating_model_in_ollama", name=model_name)
        await client.create(model=model_name, modelfile=modelfile_content)
        return {"status": "success", "message": f"Model {model_name} registered in Ollama"}

    def _prepare_dataset(self, data: Union[str, List[str], List[dict]]) -> Dataset:
        """Parses training data into a Hugging Face Dataset."""
        processed_data = []

        if isinstance(data, str):
            # Assume path to JSONL
            path = Path(data)
            if path.exists():
                with open(path, "r") as f:
                    for line in f:
                        processed_data.append(json.loads(line))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    processed_data.append({"text": item})
                elif isinstance(item, dict):
                    # Expect instruction/input/output or text
                    if "text" in item:
                        processed_data.append(item)
                    elif "instruction" in item:
                        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item.get('output', '')}"
                        processed_data.append({"text": text})

        return Dataset.from_list(processed_data)