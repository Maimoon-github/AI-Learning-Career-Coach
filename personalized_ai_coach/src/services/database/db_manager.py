from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from sqlalchemy import DateTime, Index, String, Text, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Configure structured logging
log = structlog.get_logger(__name__)

# SQLAlchemy 2.0 Declarative Base
class Base(DeclarativeBase):
    """Base class for all models."""
    pass

# Models
class Checkpoint(Base):
    """
    Storage for LangGraph state checkpoints.
    Supports versioning and point-in-time recovery for agentic workflows.
    """
    __tablename__ = "langgraph_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    parent_checkpoint_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    checkpoint: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    checkpoint_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        index=True
    )

    __table_args__ = (
        Index("idx_checkpoint_thread", "thread_id"),
        Index("idx_checkpoint_created", "created_at"),
    )

class UserProgress(Base):
    """
    Tracks user learning metrics and generated reports over time.
    """
    __tablename__ = "user_progress"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    week_number: Mapped[int] = mapped_column(primary_key=True)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    report: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )

    __table_args__ = (
        Index("idx_user_progress_lookup", "user_id", "week_number"),
    )

# Global lazy-initialized engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

def _get_engine() -> AsyncEngine:
    """Initialize or return the global async engine with optimized pooling."""
    global _engine
    if _engine is None:
        db_url = os.getenv(
            "DATABASE_URL", 
            "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_coach"
        )
        _engine = create_async_engine(
            db_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=os.getenv("APP_ENV") == "development",
        )
    return _engine

def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Initialize or return the global session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False
        )
    return _session_factory

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    Handles automatic commit on success and rollback on failure.
    """
    session = _get_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        log.error("database_session_error", error=str(e))
        raise
    finally:
        await session.close()

async def init_db() -> None:
    """Initialize database schemas asynchronously."""
    try:
        async with _get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("database_tables_initialized")
    except Exception as e:
        log.critical("database_init_failed", error=str(e))
        raise

# ----- LangGraph Checkpoint Persistence -----

async def save_checkpoint(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    checkpoint_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    parent_checkpoint_id: Optional[str] = None,
) -> None:
    """
    Atomically store a state checkpoint.
    Uses SQLAlchemy 2.0 UPSERT semantics logic (or native SQL for performance).
    """
    async with get_session() as session:
        # Using text for raw UPSERT which is more efficient for this specific case
        query = text("""
            INSERT INTO langgraph_checkpoints 
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
            VALUES (:thread_id, :ns, :cid, :parent, 'agent_state', :checkpoint, :metadata)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
            DO UPDATE SET 
                checkpoint = EXCLUDED.checkpoint, 
                metadata = EXCLUDED.metadata,
                parent_checkpoint_id = EXCLUDED.parent_checkpoint_id
        """)
        
        await session.execute(query, {
            "thread_id": thread_id,
            "ns": checkpoint_ns,
            "cid": checkpoint_id,
            "parent": parent_checkpoint_id,
            "checkpoint": checkpoint_data,  # SQLAlchemy handles dict to JSON
            "metadata": metadata or {},
        })
        log.debug("checkpoint_persisted", thread=thread_id, checkpoint=checkpoint_id)

async def get_latest_checkpoint(thread_id: str, checkpoint_ns: str = "") -> Optional[Dict[str, Any]]:
    """Retrieve the most recent checkpoint for a given thread namespace."""
    async with get_session() as session:
        stmt = (
            select(Checkpoint.checkpoint)
            .where(Checkpoint.thread_id == thread_id)
            .where(Checkpoint.checkpoint_ns == checkpoint_ns)
            .order_by(Checkpoint.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

async def list_checkpoints(thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """List recent checkpoints for debugging and audit trails."""
    async with get_session() as session:
        stmt = (
            select(Checkpoint.checkpoint_id, Checkpoint.created_at, Checkpoint.checkpoint_metadata)
            .where(Checkpoint.thread_id == thread_id)
            .order_by(Checkpoint.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return [
            {
                "checkpoint_id": r.checkpoint_id,
                "created_at": r.created_at.isoformat(),
                "metadata": r.checkpoint_metadata or {}
            }
            for r in result.all()
        ]

async def delete_checkpoints(thread_id: str, older_than_days: int = 30) -> int:
    """ Garbage collect checkpoints older than a specific threshold."""
    async with get_session() as session:
        # Fixing the interval parameterization issue
        query = text("""
            DELETE FROM langgraph_checkpoints
            WHERE thread_id = :thread_id 
            AND created_at < NOW() - (:days * INTERVAL '1 day')
            RETURNING checkpoint_id
        """)
        result = await session.execute(query, {"thread_id": thread_id, "days": older_than_days})
        deleted_count = len(result.all())
        log.info("checkpoints_vacuumed", thread=thread_id, count=deleted_count)
        return deleted_count

# ----- User Progress Tracking -----

async def save_user_progress(
    user_id: str, 
    week_number: int, 
    metrics: Dict[str, Any], 
    report: Dict[str, Any]
) -> None:
    """Save or update weekly user progress snapshots."""
    async with get_session() as session:
        query = text("""
            INSERT INTO user_progress (user_id, week_number, metrics, report, created_at)
            VALUES (:user_id, :week, :metrics, :report, NOW())
            ON CONFLICT (user_id, week_number) 
            DO UPDATE SET 
                metrics = EXCLUDED.metrics, 
                report = EXCLUDED.report,
                created_at = NOW()
        """)
        await session.execute(query, {
            "user_id": user_id,
            "week": week_number,
            "metrics": metrics,
            "report": report
        })
        log.info("user_progress_saved", user=user_id, week=week_number)

async def get_user_progress_history(user_id: str, last_n_weeks: int = 4) -> List[Dict[str, Any]]:
    """Retrieve historical progress for a specific user."""
    async with get_session() as session:
        stmt = (
            select(UserProgress.week_number, UserProgress.metrics, UserProgress.report, UserProgress.created_at)
            .where(UserProgress.user_id == user_id)
            .order_by(UserProgress.week_number.desc())
            .limit(last_n_weeks)
        )
        result = await session.execute(stmt)
        return [
            {
                "week_number": r.week_number,
                "metrics": r.metrics,
                "report": r.report,
                "created_at": r.created_at.isoformat()
            }
            for r in result.all()
        ]

# ----- Utility -----

async def health_check() -> bool:
    """Verify database connectivity and responsiveness."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error("database_health_check_failed", error=str(e))
        return False