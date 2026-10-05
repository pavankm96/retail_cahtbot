"""Environment configuration and runtime settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    hf_chat_model: str = field(
        default_factory=lambda: os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    )
    hf_embedding_model: str = field(
        default_factory=lambda: os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    sqlite_db_path: Path = field(
        default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", "./data/retail_vectors.db"))
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))

    def validate(self) -> None:
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
