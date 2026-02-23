"""Import seed questions from JSON into SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "questions.db"
SRC = ROOT / "data" / "questions.json"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    short_summary TEXT NOT NULL,
    full_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    difficulty INTEGER NOT NULL,
    tags TEXT NOT NULL,
    qtype TEXT NOT NULL,
    est_time_seconds INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS student_records (
    student_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    correct INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    time_spent_seconds INTEGER NOT NULL,
    answer_text TEXT
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def import_questions() -> int:
    with SRC.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conn = sqlite3.connect(DB)
    try:
        init_db(conn)
        cur = conn.cursor()
        for q in data:
            tags = ",".join(q.get("tags", []))
            cur.execute(
                """
                INSERT OR REPLACE INTO questions
                (id, short_summary, full_text, answer, difficulty, tags, qtype, est_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    q["id"],
                    q["short_summary"],
                    q["full_text"],
                    q["answer"],
                    q["difficulty"],
                    tags,
                    q["qtype"],
                    q["est_time_seconds"],
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return len(data)


if __name__ == "__main__":
    count = import_questions()
    print(f"imported {count} questions into {DB}")
