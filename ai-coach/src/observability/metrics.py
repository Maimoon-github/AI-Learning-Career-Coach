"""Prometheus metrics."""

# src/observability/metrics.py

import logging
import json
from datetime import datetime
from src.state.schema import CoachState


logging.basicConfig(
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}',
    level=logging.INFO,
)
logger = logging.getLogger("ai-coach")


def log_agent_call(agent: str, state: CoachState, duration_ms: float):
    logger.info(json.dumps({
        "event": "agent_call",
        "agent": agent,
        "user_id": state["user_profile"].user_id,
        "session_id": state["session_id"],
        "iteration": state["iteration_count"],
        "duration_ms": duration_ms,
    }))


def log_evaluation(state: CoachState):
    if state.get("evaluation"):
        logger.info(json.dumps({
            "event": "evaluation",
            "score": state["evaluation"].score,
            "passed": state["evaluation"].passed,
            "user_id": state["user_profile"].user_id,
        }))