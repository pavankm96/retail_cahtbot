"""Hugging Face embeddings and a persistent SQLite-backed vector store."""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the configured Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)


def _pack(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32))


def _unpack(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


class SQLiteVectorStore:
    """Minimal persistent cosine-similarity vector store on top of sqlite3."""

    def __init__(self, db_path: Path | str = settings.sqlite_db_path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text TEXT NOT NULL,
                       metadata TEXT,
                       embedding BLOB NOT NULL
                   )"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add_texts(self, texts: Iterable[str], metadatas: Iterable[dict] | None = None,
                  embeddings: HuggingFaceEmbeddings | None = None) -> None:
        emb = embeddings or get_embeddings()
        texts = list(texts)
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        vectors = emb.embed_documents(texts)
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO documents (text, metadata, embedding) VALUES (?, ?, ?)",
                [(t, json.dumps(m), _pack(np.asarray(v))) for t, m, v in zip(texts, metas, vectors)],
            )

    def similarity_search(self, query: str, k: int = settings.top_k,
                          embeddings: HuggingFaceEmbeddings | None = None) -> List[Tuple[str, dict, float]]:
        emb = embeddings or get_embeddings()
        q = np.asarray(emb.embed_query(query), dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        rows = self._conn().execute("SELECT text, metadata, embedding FROM documents").fetchall()
        scored: List[Tuple[str, dict, float]] = []
        for text, meta, blob in rows:
            v = _unpack(blob)
            v = v / (np.linalg.norm(v) + 1e-9)
            scored.append((text, json.loads(meta or "{}"), float(np.dot(q_norm, v))))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]
