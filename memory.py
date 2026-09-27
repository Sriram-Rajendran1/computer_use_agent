import json

import psycopg

from config import DATABASE_URL, RECALL_LIMIT


def save_task_result(goal: str, outcome: str, steps: list[dict]) -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO task_memory (goal, outcome, steps) VALUES (%s, %s, %s)",
            (goal, outcome, json.dumps(steps)),
        )


def recall_similar_tasks(goal: str) -> list[str]:
    """Plain LIKE-based recall on keywords from the goal. Good enough for v1."""
    keywords = [w for w in goal.lower().split() if len(w) > 3]
    if not keywords:
        return []
    pattern = f"%{keywords[0]}%"
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            """
            SELECT goal, outcome, created_at
            FROM task_memory
            WHERE lower(goal) LIKE %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (pattern, RECALL_LIMIT),
        ).fetchall()
    return [f"[{row[2]}] goal={row[0]!r} outcome={row[1]!r}" for row in rows]
