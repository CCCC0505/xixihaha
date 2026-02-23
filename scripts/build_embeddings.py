"""Build embeddings from question short summaries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "questions.db"
OUT_EMB = ROOT / "data" / "embeddings.npy"
OUT_IDS = ROOT / "data" / "ids.npy"


def load_questions() -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(DB)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, short_summary FROM questions ORDER BY id")
        rows = cur.fetchall()
    finally:
        conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return ids, texts


def main() -> None:
    ids, texts = load_questions()
    if not texts:
        raise RuntimeError("No questions found. Run scripts/import_questions.py first.")

    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(OUT_EMB, embs)
    np.save(OUT_IDS, np.array(ids))
    print(f"saved embeddings {embs.shape} -> {OUT_EMB}")


if __name__ == "__main__":
    main()
