import os
from typing import Any, AsyncIterator

import httpx
from langchain_community.llms import Ollama
from langchain_core.language_models.llms import BaseLLM
# Add near the top
from cachetools import TTLCache
import hashlib

from src.services.metrics_exporter import llm_token_usage, llm_latency

async def generate(self, prompt: str, task_type: str = "structured_extraction") -> str:
    model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
    start = time.perf_counter()
    response = await self._get_llm_for_task(task_type).ainvoke(prompt)
    duration = time.perf_counter() - start
    llm_latency.labels(model=model, task_type=task_type).observe(duration)
    # token counting (approx)
    llm_token_usage.labels(model=model, task_type=task_type).set(len(prompt.split()) + len(response.split()))
    return response

class OllamaClient:
    def __init__(self):
        # ... existing init ...
        self._response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour

    async def generate(self, prompt: str, task_type: str = "structured_extraction") -> str:
        llm = self.get_llm_for_task(task_type)
        # Create cache key
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        temp = 0.7 if task_type == "motivational_framing" else 0.2
        key = hashlib.md5(f"{model}:{temp}:{prompt}".encode()).hexdigest()
        if key in self._response_cache:
            return self._response_cache[key]
        response = await llm.ainvoke(prompt)
        self._response_cache[key] = response
        return response

# Task type to model routing (mirrors config/llm_config.yaml)
TASK_MODEL_MAP = {
    "structured_extraction": "llama3.2:3b",
    "gap_analysis": "llama3.2:3b",
    "curriculum_design": "llama3.1:70b",
    "resource_curation": "llama3.2:3b",
    "project_generation": "llama3.1:70b",
    "fine_tuning_eval": "llama3.2:3b",
    "report_generation": "llama3.2:3b",
    "motivational_framing": "llama3.2:3b",  # creative temp handled separately
}

class OllamaClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._clients: dict[str, BaseLLM] = {}
        self._initialized = True

    def _get_llm(self, model: str, temperature: float = 0.2) -> BaseLLM:
        key = f"{model}:{temperature}"
        if key not in self._clients:
            self._clients[key] = Ollama(
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                num_ctx=8192 if "3b" in model else 32768,
            )
        return self._clients[key]

    def get_llm_for_task(self, task_type: str) -> BaseLLM:
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        temperature = 0.7 if task_type == "motivational_framing" else 0.2
        return self._get_llm(model, temperature)

    async def generate(self, prompt: str, task_type: str = "structured_extraction") -> str:
        llm = self.get_llm_for_task(task_type)
        return await llm.ainvoke(prompt)

    async def stream(self, prompt: str, task_type: str = "structured_extraction") -> AsyncIterator[str]:
        llm = self.get_llm_for_task(task_type)
        async for chunk in llm.astream(prompt):
            yield chunk

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False