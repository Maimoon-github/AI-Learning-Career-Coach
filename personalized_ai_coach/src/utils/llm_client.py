import hashlib
import os
import time
from typing import Any, AsyncIterator, Type, TypeVar

import httpx
from openai import AsyncOpenAI
import instructor
import structlog
from cachetools import TTLCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

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
    "embeddings": "nomic-embed-text:latest",
}

class OllamaClient:
    """
    Production-ready Ollama client for local-first AI.
    Features:
    - Singleton pattern.
    - Task-based model routing.
    - Instructor integration for structured extraction.
    - LRU Caching for deterministic prompts.
    - Async/Sync health checks.
    """
    _instance = None

    def __new__(cls) -> "OllamaClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            cls._instance._clients = {}
            # Use AsyncOpenAI for better Instructor compatibility with Ollama
            openai_client = AsyncOpenAI(
                base_url=f"{cls._instance.base_url}/v1",
                api_key="ollama",  # Required but ignored by Ollama
            )
            cls._instance._instructor_client = instructor.from_openai(
                openai_client,
                mode=instructor.Mode.JSON,
            )
            cls._instance._response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        return cls._instance

    def __init__(self):
        # Already initialized in __new__
        pass

    def _get_llm(self, model: str, temperature: float = 0.2) -> BaseChatModel:
        key = f"{model}:{temperature}"
        if key not in self._clients:
            # Dynamic context window based on model scale
            num_ctx = 32768 if "70b" in model else 8192
            self._clients[key] = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                num_ctx=num_ctx,
            )
        return self._clients[key]

    def get_llm_for_task(self, task_type: str) -> BaseChatModel:
        """Route to appropriate LangChain model based on task type."""
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        temperature = 0.7 if task_type == "motivational_framing" else 0.2
        return self._get_llm(model, temperature)

    async def generate(self, prompt: str, task_type: str = "structured_extraction") -> str:
        """Standard async generation with caching."""
        llm = self.get_llm_for_task(task_type)
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        
        # Cache lookup
        temp = 0.7 if task_type == "motivational_framing" else 0.2
        cache_key = hashlib.md5(f"{model}:{temp}:{prompt}".encode()).hexdigest()
        if cache_key in self._response_cache:
            log.debug("llm_cache_hit", task_type=task_type, model=model)
            return self._response_cache[cache_key]

        start = time.perf_counter()
        try:
            response = await llm.ainvoke(prompt)
            content = getattr(response, "content", str(response))
            duration = time.perf_counter() - start
            
            log.info("llm_generated", model=model, task_type=task_type, duration=f"{duration:.2f}s")
            self._response_cache[cache_key] = content
            return content
        except Exception as e:
            log.error("llm_generation_failed", model=model, error=str(e))
            raise

    async def extract_structured(self, prompt: str, response_model: Type[T], task_type: str = "structured_extraction") -> T:
        """Use Instructor for reliable structured output extraction."""
        model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
        start = time.perf_counter()
        try:
            result = await self._instructor_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
            )
            duration = time.perf_counter() - start
            log.info("structured_extraction_success", model=model, type=response_model.__name__, duration=f"{duration:.2f}s")
            return result
        except Exception as e:
            log.error("structured_extraction_failed", model=model, type=response_model.__name__, error=str(e))
            raise

    async def stream(self, prompt: str, task_type: str = "structured_extraction") -> AsyncIterator[str]:
        """Stream response chunks."""
        llm = self.get_llm_for_task(task_type)
        async for chunk in llm.astream(prompt):
            # Handle different chunk types (LangChain BaseMessage vs string)
            content = getattr(chunk, "content", str(chunk))
            if content:
                yield content

    def health_check(self) -> bool:
        """Synchronous check if Ollama service is reachable."""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            log.error("ollama_health_check_failed", error=str(e))
            return False

# --- Factory Functions ---

def get_llm(task_type: str = "structured_extraction") -> Any:
    """
    Helper for CrewAI/LangChain compatibility. 
    Returns the string identifier for the model.
    """
    model = TASK_MODEL_MAP.get(task_type, "llama3.2:3b")
    return f"ollama/{model}"

def get_embeddings() -> Embeddings:
    """Get LangChain-compatible Ollama embeddings."""
    return OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=TASK_MODEL_MAP.get("embeddings", "nomic-embed-text:latest"),
    )

def get_embedder_config() -> dict[str, Any]:
    """Get CrewAI-compatible embedder configuration."""
    return {
        "provider": "ollama",
        "config": {
            "model": TASK_MODEL_MAP.get("embeddings", "nomic-embed-text:latest"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
    }