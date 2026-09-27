import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from config import DATABASE_URL

_TASK_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS task_memory (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,
    outcome TEXT NOT NULL,
    steps JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_db() -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(_TASK_MEMORY_DDL)


def get_checkpointer() -> PostgresSaver:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver
