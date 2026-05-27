from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.error_handling import OllamaConnectionError

log = structlog.get_logger(__name__)


class OllamaToolInput(BaseModel):
    action: str = Field(description="Action: list_models | pull_model | delete_model | fine_tuning_dry_run")
    model_name: str = Field(default="", description="Model name for pull/delete/info actions")
    user_notes: list[str] = Field(default=[], description="User notes for fine-tuning dry run")


class OllamaTool(BaseTool):
    name: str = "ollama_manager"
    description: str = "Manages local Ollama models: list, pull, delete, and dry-run fine-tuning."
    args_schema: type[BaseModel] = OllamaToolInput

    def _run(self, action: str, model_name: str = "", user_notes: list[str] = None) -> dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(
            self._async_run(action, model_name, user_notes or [])
        )

    async def _async_run(self, action: str, model_name: str, user_notes: list[str]) -> dict[str, Any]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                if action == "list_models":
                    r = await client.get(f"{base_url}/api/tags")
                    return r.json()
                elif action == "pull_model":
                    if not model_name:
                        return {"error": "model_name required for pull_model"}
                    r = await client.post(f"{base_url}/api/pull", json={"name": model_name})
                    return {"status": "pulling", "model": model_name, "response": r.json()}
                elif action == "delete_model":
                    if not model_name:
                        return {"error": "model_name required for delete_model"}
                    r = await client.delete(f"{base_url}/api/delete", json={"name": model_name})
                    return {"status": "deleted", "model": model_name}
                elif action == "fine_tuning_dry_run":
                    # Validate that required models exist
                    models_r = await client.get(f"{base_url}/api/tags")
                    models = [m["name"] for m in models_r.json().get("models", [])]
                    required = ["llama3.2:3b", "llama3.1:70b"]
                    missing = [m for m in required if m not in models]
                    if missing:
                        return {"status": "failed", "missing_models": missing}
                    # Simulate fine-tuning preparation (dry run)
                    # In real implementation, would call Ollama's fine-tuning endpoint
                    sample_count = len(user_notes)
                    return {
                        "status": "dry_run_success",
                        "base_model": "llama3.2:3b",
                        "sample_notes_used": sample_count,
                        "estimated_time_minutes": max(5, sample_count // 10),
                        "required_examples": 50,
                        "ready": sample_count >= 50,
                        "message": f"Dry run: {sample_count} notes prepared. {'Ready for fine-tuning' if sample_count >= 50 else 'Need more notes.'}"
                    }
                else:
                    return {"error": f"Unknown action: {action}"}
            except httpx.ConnectError as exc:
                raise OllamaConnectionError(f"Cannot connect to Ollama at {base_url}") from exc