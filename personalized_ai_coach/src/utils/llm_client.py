import os
import time
import hashlib
from typing import Any, AsyncIterator, Optional

import httpx
import structlog
from cachetools import TTLCache
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

log = structlog.get_logger(__name__)

# Task type to model routing (mirrors config/llm_config.yaml)
TASK_MODEL_MAP = {
    "structured_extraction": "llama3.2:3b",
    "gap_analysis": "llama3.2:3b",
    "curriculum_design": "llama3.1:70b",
    "resource_curation": "llama3.2:3b",
    "project_generation": "llama3.1:70b",
    "fine_tuning_eval": "llama3.2:3b",
    "report_generation": "llama3.2:3b",
    "motivational_framing": "llama3.2:3b",
}

class OllamaClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._clients: dict[str, BaseChatModel] = {}
        self._response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        self._initialized = True

    def _get_llm(self, model: str, temperature: float = 0.2) -> BaseChatModel:
        key = f"{model}:{temperature}"
        if key not in self._clients:
            self._clients[key] = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                num_ctx=8192 if "3b" in model else 32768,
            )
        return self._clients[key]

    def get_llm_for_task(self, task_type: str) -> BaseChatModel:
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        temperature = 0.7 if task_type == "motivational_framing" else 0.2
        return self._get_llm(model, temperature)

    async def generate(self, prompt: str, task_type: str = "structured_extraction") -> str:
        llm = self.get_llm_for_task(task_type)
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        
        # Cache check
        temp = 0.7 if task_type == "motivational_framing" else 0.2
        key = hashlib.md5(f"{model}:{temp}:{prompt}".encode()).hexdigest()
        if key in self._response_cache:
            log.debug("llm_cache_hit", task_type=task_type)
            return self._response_cache[key]

        start = time.perf_counter()
        response = await llm.ainvoke(prompt)
        duration = time.perf_counter() - start
        
        log.debug("llm_generated", model=model, duration=duration)
        
        self._response_cache[key] = response
        return response

    async def stream(self, prompt: str, task_type: str = "structured_extraction") -> AsyncIterator[str]:
        llm = self.get_llm_for_task(task_type)
        async for chunk in llm.astream(prompt):
            yield chunk

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            log.error("ollama_health_failed", error=str(e))
            return False

def get_llm(task_type: str = "structured_extraction") -> Any:
    """Helper for CrewAI agents to get a configured LLM identifier."""
    model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
    # Return string format for CrewAI compatibility to avoid type validation issues
    return f"ollama/{model}"