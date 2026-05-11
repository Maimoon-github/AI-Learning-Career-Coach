"""SQLite persistence setup."""

# src/graph/checkpointer.py

import os
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer() -> SqliteSaver:
    """
    SQLite checkpointer for development.
    Swap for AsyncPostgresSaver in production for horizontal scaling:

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver.from_conn_string(os.environ["POSTGRES_URI"])
    """
    db_path = os.environ.get("CHECKPOINTER_DB_URI", "sqlite:///./data/checkpoints.db")
    # SqliteSaver expects the raw file path, not the URI form
    file_path = db_path.replace("sqlite:///", "")
    return SqliteSaver.from_conn_string(file_path)