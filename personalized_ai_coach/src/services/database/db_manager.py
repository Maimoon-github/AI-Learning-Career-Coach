from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import structlog
from sqlalchemy import text, Column, String, Integer, DateTime, JSON, Index
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

log = structlog.get_logger(__name__)

Base = declarative_base()
_engine = None
_session_factory = None

# SQLAlchemy model for checkpoint storage
class Checkpoint(Base):
    __tablename__ = "langgraph_checkpoints"
    thread_id = Column(String(255), primary_key=True)
    checkpoint_ns = Column(String(255), primary_key=True)
    checkpoint_id = Column(String(255), primary_key=True)
    parent_checkpoint_id = Column(String(255), nullable=True)
    type = Column(String(50))
    checkpoint = Column(JSON, nullable=False)  # serialized AgentState
    checkpoint_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, server_default=text("NOW()"))
    __table_args__ = (Index("idx_thread", "thread_id"),)

def _get_engine():
    global _engine
    if _engine is None:
        db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_coach")
        _engine = create_async_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=os.getenv("APP_ENV") == "development",
        )
    return _engine

def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _session_factory

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def init_db() -> None:
    """Create tables if they don't exist."""
    async with _get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_tables_created")

# ----- LangGraph checkpoint persistence -----
async def save_checkpoint(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    checkpoint_data: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
    parent_checkpoint_id: Optional[str] = None,
) -> None:
    """Atomically store a state checkpoint."""
    async with get_session() as session:
        await session.execute(
            text("""
                INSERT INTO langgraph_checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :parent, 'agent_state', :checkpoint, :metadata)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                DO UPDATE SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata
            """),
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "parent": parent_checkpoint_id,
                "checkpoint": json.dumps(checkpoint_data),
                "metadata": json.dumps(metadata or {}),
            }
        )
        log.debug("checkpoint_saved", thread=thread_id, checkpoint=checkpoint_id)

async def get_latest_checkpoint(thread_id: str, checkpoint_ns: str = "") -> Optional[dict[str, Any]]:
    """Retrieve the most recent checkpoint for a given thread."""
    async with get_session() as session:
        row = await session.execute(
            text("""
                SELECT checkpoint FROM langgraph_checkpoints
                WHERE thread_id = :thread_id AND checkpoint_ns = :ns
                ORDER BY created_at DESC LIMIT 1
            """),
            {"thread_id": thread_id, "ns": checkpoint_ns}
        )
        result = row.first()
        if result:
            return json.loads(result[0])
        return None

async def list_checkpoints(thread_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """List checkpoints for a thread (for debugging)."""
    async with get_session() as session:
        rows = await session.execute(
            text("""
                SELECT checkpoint_id, created_at, metadata
                FROM langgraph_checkpoints
                WHERE thread_id = :thread_id
                ORDER BY created_at DESC LIMIT :limit
            """),
            {"thread_id": thread_id, "limit": limit}
        )
        return [
            {"checkpoint_id": r[0], "created_at": r[1].isoformat(), "metadata": json.loads(r[2]) if r[2] else {}}
            for r in rows.fetchall()
        ]

async def delete_checkpoints(thread_id: str, older_than_days: int = 30) -> int:
    """Garbage collect old checkpoints."""
    async with get_session() as session:
        result = await session.execute(
            text("""
                DELETE FROM langgraph_checkpoints
                WHERE thread_id = :thread_id AND created_at < NOW() - INTERVAL ':days days'
                RETURNING checkpoint_id
            """),
            {"thread_id": thread_id, "days": older_than_days}
        )
        deleted = len(result.fetchall())
        log.info("checkpoints_garbage_collected", thread=thread_id, count=deleted)
        return deleted

# ----- User progress methods (original) -----
async def save_user_progress(user_id: str, week_number: int, metrics: dict, report: dict) -> None:
    async with get_session() as session:
        await session.execute(
            text("""
                INSERT INTO user_progress (user_id, week_number, metrics, report, created_at)
                VALUES (:user_id, :week_number, :metrics, :report, NOW())
                ON CONFLICT (user_id, week_number) DO UPDATE SET metrics = EXCLUDED.metrics, report = EXCLUDED.report
            """),
            {"user_id": user_id, "week_number": week_number, "metrics": json.dumps(metrics), "report": json.dumps(report)},
        )

async def get_user_progress_history(user_id: str, last_n_weeks: int = 4) -> list[dict]:
    async with get_session() as session:
        rows = await session.execute(
            text("""
                SELECT week_number, metrics, report, created_at
                FROM user_progress WHERE user_id = :user_id ORDER BY week_number DESC LIMIT :limit
            """),
            {"user_id": user_id, "limit": last_n_weeks}
        )
        return [
            {"week_number": r[0], "metrics": json.loads(r[1]), "report": json.loads(r[2]), "created_at": r[3].isoformat()}
            for r in rows.fetchall()
        ]

async def health_check() -> bool:
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error("db_health_failed", error=str(e))
        return False