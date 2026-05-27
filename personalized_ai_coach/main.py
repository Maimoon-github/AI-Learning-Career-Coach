#!/usr/bin/env python3
"""Foundation & Infrastructure entry point.

Loads environment, initialises LangGraph with Postgres/Redis checkpointing,
bootstraps voice interface stub, and exposes /health endpoint.
"""

import asyncio
import os
from contextlib import asynccontextmanager

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import uvicorn

# Load environment from .env
load_dotenv()

# Configure structured JSON logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# ---------- LangGraph state definition ----------
class AgentState(TypedDict):
    user_input: str
    echo: Optional[str]

# Simple echo node to demonstrate checkpointing
async def echo_node(state: AgentState) -> AgentState:
    state["echo"] = f"Echo: {state['user_input']}"
    logger.info("echo_node_executed", input=state["user_input"])
    return state

async def build_graph(backend: str):
    """Build and compile LangGraph with the specified checkpoint backend."""
    if backend == "postgres":
        conn_string = os.environ["DATABASE_URL"]
        checkpointer = PostgresSaver.from_conn_string(conn_string)
        # Setup schema (idempotent)
        await checkpointer.setup()
        logger.info("Postgres checkpointer initialised", url=conn_string)
    elif backend == "redis":
        redis_url = os.environ["REDIS_URL"]
        checkpointer = RedisSaver.from_conn_string(redis_url)
        await checkpointer.setup()
        logger.info("Redis checkpointer initialised", url=redis_url)
    else:
        raise ValueError(f"Unknown checkpoint backend: {backend}")

    workflow = StateGraph(AgentState)
    workflow.add_node("echo", echo_node)
    workflow.set_entry_point("echo")
    workflow.add_edge("echo", END)

    app = workflow.compile(checkpointer=checkpointer)
    return app

# ---------- Voice interface stub ----------
async def voice_loop():
    """Stub for full-duplex voice interface (STT/TTS)."""
    logger.info("Voice loop started (stub mode)")
    while True:
        await asyncio.sleep(10)
        logger.debug("Voice loop heartbeat – replace with real STT/TTS")

# ---------- Healthcheck HTTP server ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global graph_app, voice_task
    backend = os.getenv("CHECKPOINT_BACKEND", "postgres")
    graph_app = await build_graph(backend)
    logger.info("LangGraph compiled and checkpointer ready")

    # Start voice loop in background
    voice_task = asyncio.create_task(voice_loop())
    yield
    # Shutdown
    voice_task.cancel()
    await voice_task

fastapi_app = FastAPI(lifespan=lifespan)

@fastapi_app.get("/health")
async def healthcheck():
    """Verify connectivity to Postgres and Redis."""
    status = {"status": "healthy", "postgres": False, "redis": False}
    # Check Postgres
    try:
        import asyncpg
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        await conn.execute("SELECT 1")
        await conn.close()
        status["postgres"] = True
    except Exception as e:
        logger.error("Postgres health check failed", error=str(e))
    # Check Redis
    try:
        import redis.asyncio as redis
        r = redis.from_url(os.environ["REDIS_URL"])
        await r.ping()
        await r.close()
        status["redis"] = True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))

    if not (status["postgres"] and status["redis"]):
        status["status"] = "degraded"
    logger.info("Healthcheck requested", status=status["status"])
    return status

async def main():
    """Run the FastAPI server (healthcheck) and voice loop concurrently."""
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())