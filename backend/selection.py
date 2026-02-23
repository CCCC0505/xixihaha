"""Question selection logic for the MVP."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "questions.db"


@dataclass
class StudentProfile:
    student_id: str
    ability: float = 5.0
    tag_stats: dict[str, tuple[int, int]] = field(default_factory=dict)


def ability_match(student_ability: float, q_diff: int, max_diff: int = 10) -> float:
    return max(0.0, 1.0 - abs(student_ability - q_diff) / max_diff)


def compute_weakness_bonus(student_profile: StudentProfile, q_tags: list[str]) -> float:
    bonus = 0.0
    for tag in q_tags:
        stats = student_profile.tag_stats.get(tag)
        if not stats:
            continue
        correct, total = stats
        if total == 0:
            continue
        acc = correct / total
        if acc < 0.6:
            bonus += 0.2
    return bonus


def query_candidates(target_tags: list[str], difficulty: int | None = None) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        clauses = []
        params: list[object] = []

        if target_tags:
            clauses.append("(" + " OR ".join(["tags LIKE ?" for _ in target_tags]) + ")")
            params.extend([f"%{tag}%" for tag in target_tags])

        if difficulty is not None:
            clauses.append("difficulty BETWEEN ? AND ?")
            params.extend([max(1, difficulty - 1), min(10, difficulty + 1)])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(
            f"SELECT id, short_summary, difficulty, tags, qtype FROM questions {where_sql}",
            params,
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": row["id"],
            "short_summary": row["short_summary"],
            "difficulty": row["difficulty"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "qtype": row["qtype"],
        }
        for row in rows
    ]


def score_and_sample(
    candidates: list[dict],
    student: StudentProfile,
    n: int = 5,
    top_factor: int = 3,
) -> list[dict]:
    if not candidates:
        return []

    scored: list[tuple[dict, float]] = []
    for q in candidates:
        score = ability_match(student.ability, q["difficulty"])
        score += compute_weakness_bonus(student, q["tags"])
        score += random.uniform(0.0, 0.15)  # 保持一定随机性
        scored.append((q, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    shortlist_count = min(len(scored), max(n, n * top_factor))
    shortlist = [q for q, _ in scored[:shortlist_count]]

    return random.sample(shortlist, min(n, len(shortlist)))
