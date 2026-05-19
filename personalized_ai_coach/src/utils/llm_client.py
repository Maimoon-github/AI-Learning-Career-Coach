from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import structlog
import yaml
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.utils.error_handling import OllamaConnectionError

log = structlog.get_logger(__name__)

_CLIENTS: dict[str, ChatOllama] = {}


def _load_llm_config() -> dict[str, Any]:
    with open("config/llm_config.yaml") as f:
        raw = f.read()
    import os
    raw = raw.replace("${OLLAMA_BASE_URL}", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    return yaml.safe_load(raw)


def get_llm(task_type: str = "primary") -> ChatOllama:
    """Return a cached ChatOllama client routed by task type."""
    config = _load_llm_config()
    route = config.get("routing", {}).get(task_type, "primary")
    model_cfg = config.get(route, config["primary"])

    cache_key = f"{model_cfg['model']}:{model_cfg['base_url']}"
    if cache_key not in _CLIENTS:
        try:
            client = ChatOllama(
                model=model_cfg["model"],
                base_url=model_cfg["base_url"],
                temperature=model_cfg.get("temperature", 0.2),
                num_ctx=model_cfg.get("num_ctx", 8192),
                num_predict=model_cfg.get("num_predict", 2048),
                top_p=model_cfg.get("top_p", 0.9),
                repeat_penalty=model_cfg.get("repeat_penalty", 1.1),
            )
            _CLIENTS[cache_key] = client
            log.info("llm_client_created", model=model_cfg["model"], route=route)
        except Exception as exc:
            raise OllamaConnectionError(
                f"Failed to initialize Ollama client for model {model_cfg['model']}",
                context={"model_cfg": model_cfg, "error": str(exc)},
            ) from exc
    return _CLIENTS[cache_key]


def get_openai_fallback(model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Return OpenAI client as fallback when Ollama is unavailable."""
    import os
    return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY", ""))