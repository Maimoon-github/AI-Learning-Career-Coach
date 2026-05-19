from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

log = structlog.get_logger(__name__)

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ai_coach.db")
        _engine = create_async_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=os.getenv("APP_ENV", "development") == "development",
        )
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
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


async def save_user_progress(
    user_id: str,
    week_number: int,
    metrics: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Persist a weekly progress snapshot for analytics and retrieval."""
    async with get_session() as session:
        import json
        await session.execute(
            text("""
                INSERT INTO user_progress (user_id, week_number, metrics, report, created_at)
                VALUES (:user_id, :week_number, :metrics, :report, NOW())
                ON CONFLICT (user_id, week_number)
                DO UPDATE SET metrics = EXCLUDED.metrics, report = EXCLUDED.report
            """),
            {
                "user_id": user_id,
                "week_number": week_number,
                "metrics": json.dumps(metrics),
                "report": json.dumps(report),
            },
        )
        log.info("progress_saved", user_id=user_id, week=week_number)


async def get_user_progress_history(user_id: str, last_n_weeks: int = 4) -> list[dict]:
    """Retrieve recent progress history for trend analysis."""
    async with get_session() as session:
        import json
        result = await session.execute(
            text("""
                SELECT week_number, metrics, report, created_at
                FROM user_progress
                WHERE user_id = :user_id
                ORDER BY week_number DESC
                LIMIT :limit
            """),
            {"user_id": user_id, "limit": last_n_weeks},
        )
        rows = result.fetchall()
        return [
            {
                "week_number": r[0],
                "metrics": json.loads(r[1]) if r[1] else {},
                "report": json.loads(r[2]) if r[2] else {},
                "created_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]


async def health_check() -> bool:
    """Verify database connectivity."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.error("db_health_check_failed", error=str(exc))
        return False