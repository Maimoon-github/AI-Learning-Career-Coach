#!/usr/bin/env python3
"""Integration entry point – wires all layers together."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage

# Import all services
from src.services.database.db_manager import init_db, health_check as db_health
from src.services.storage.s3_manager import S3Manager
from src.services.voice_interface.stt_service import STTService
from src.services.voice_interface.tts_service import TTSService
from src.services.voice_interface.audio_stream_handler import AudioStreamHandler

# Import LangGraph workflow
from src.langgraph_workflow.graph import create_app
from src.langgraph_workflow.state import initial_state
from src.utils.llm_client import OllamaClient
from src.utils.error_handling import async_retry_with_backoff

# At top, after imports
from src.utils.structlog_config import configure_structlog, bind_context
from src.services.metrics_exporter import start_metrics_server
from langsmith import Client as LangSmithClient

# Environment & logging
load_dotenv()
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# After load_dotenv()
configure_structlog()
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    # LangSmith automatically patches LangChain calls if env vars are set.
    # We'll also set a callback handler for CrewAI.
    from langsmith.wrappers import wrap_openai
    # CrewAI uses LangChain LLMs; tracing will propagate.

# Start Prometheus metrics server in a background thread
start_metrics_server(port=int(os.getenv("PROMETHEUS_PORT", "9090")))

# Inside lifespan, after creating graph_app, bind context for each session
# (Stub removed, full implementation is at line 115)


# Global references for lifespan
graph_app = None
voice_handler = None
current_thread_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app, voice_handler

    # 1. Initialise database and services
    await init_db()
    s3 = S3Manager()
    stt = STTService()
    tts = TTSService()
    voice_handler = AudioStreamHandler(stt, tts)
    logger.info("Services initialised")

    # 2. Compile LangGraph with production checkpointer
    backend = os.getenv("CHECKPOINT_BACKEND", "postgres")
    graph_app = create_app(backend=backend)
    logger.info("LangGraph compiled", backend=backend)

    # 3. Wire voice bridge callbacks
    voice_handler.on_transcript = on_user_transcript
    voice_handler.on_error = on_audio_error

    yield

    # 4. Graceful shutdown
    await voice_handler.close()
    logger.info("Shutdown complete")


async def on_user_transcript(transcript: str):
    """Bridge from STT to LangGraph: inject user message."""
    global graph_app, current_thread_config
    if not graph_app or not current_thread_config:
        logger.warning("Graph not ready, dropping transcript")
        return

    logger.info("Voice transcript received", text=transcript)
    # Update state with user message
    await graph_app.ainvoke(
        {"messages": [HumanMessage(content=transcript)]},
        config=current_thread_config,
    )


async def on_audio_error(error: Exception):
    logger.error("Audio handler error", error=str(error))


async def run_coaching_session_with_voice(user_id: str, target_role: str):
    """Start a coaching session and connect voice interface."""
    global current_thread_config

    # Initialise state
    state = initial_state(
        user_id=user_id,
        target_role=target_role,
        session_id=os.urandom(8).hex(),
    )
    thread_config = {"configurable": {"thread_id": state["session_id"]}}
    current_thread_config = thread_config

    # Stream the workflow – when hitl_node interrupts, voice handler will present options
    async for event in graph_app.astream(state, config=thread_config, stream_mode="values"):
        logger.debug("Workflow event", event_type=type(event))
        if "__interrupt__" in event:
            # HITL interrupt – use voice to ask for approve/revise/end
            await voice_handler.prompt_hitl(event["__interrupt__"][0].value)


# FastAPI app with lifespan
fastapi_app = FastAPI(lifespan=lifespan)


@fastapi_app.get("/health")
async def healthcheck():
    """Aggregate health status for all services."""
    status = {"status": "healthy", "services": {}}
    # Database
    status["services"]["database"] = await db_health()
    # S3 (if configured)
    s3 = S3Manager()
    status["services"]["s3"] = await s3.health_check()
    # Ollama
    llm = OllamaClient()
    status["services"]["ollama"] = llm.health_check()
    # STT/TTS stubs (just check env presence)
    status["services"]["stt"] = bool(os.getenv("OPENAI_API_KEY") or os.getenv("WHISPER_MODEL"))
    status["services"]["tts"] = bool(os.getenv("ELEVENLABS_API_KEY"))

    if all(status["services"].values()):
        status["status"] = "healthy"
    else:
        status["status"] = "degraded"
    return status


async def main():
    """Run FastAPI server (healthcheck) and optionally a demo session."""
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())