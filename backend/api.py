"""FastAPI service for MVP question selection."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.selection import StudentProfile, query_candidates, score_and_sample

app = FastAPI(title="Smart Quiz MVP API", version="0.1.0")


class SelectRequest(BaseModel):
    student_id: str = Field(..., description="Student unique identifier")
    target_tags: list[str] = Field(default_factory=list)
    n: int = Field(default=5, ge=1, le=20)
    difficulty: int | None = Field(default=None, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/select")
def select_questions(req: SelectRequest) -> dict[str, Any]:
    # MVP中先用默认画像；下一步可接入student_records动态计算ability/tag_stats
    student = StudentProfile(student_id=req.student_id)

    candidates = query_candidates(req.target_tags, req.difficulty)
    selected = score_and_sample(candidates, student, n=req.n)

    return {
        "student_id": req.student_id,
        "candidate_count": len(candidates),
        "selected": selected,
    }
