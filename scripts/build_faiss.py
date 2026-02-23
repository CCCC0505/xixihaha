"""Build a FAISS index from saved embeddings."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EMB_FILE = DATA / "embeddings.npy"
IDS_FILE = DATA / "ids.npy"
INDEX_FILE = DATA / "faiss.index"
INDEX_IDS_FILE = DATA / "faiss_ids.npy"


def main() -> None:
    emb = np.load(EMB_FILE).astype("float32")
    ids = np.load(IDS_FILE)

    d = emb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb)
    faiss.write_index(index, str(INDEX_FILE))

    np.save(INDEX_IDS_FILE, ids)
    print(f"FAISS index built with {index.ntotal} vectors -> {INDEX_FILE}")


if __name__ == "__main__":
    main()
