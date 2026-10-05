"""Load the pavankm96/KB Hugging Face dataset into the SQLite vector store."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Iterable, Tuple

from .vectorstore import SQLiteVectorStore

DATASET_REPO = "pavankm96/KB"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/main?recursive=true"

DOMAIN_MAP = {
    "customer support": "customer_support",
    "dynamic pricing": "pricing",
    "store inventory operations": "inventory",
    "supply chain logistics": "supply_chain",
}


def _chunk_markdown(text: str, max_chars: int = 800) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += ("\n\n" if buf else "") + p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _iter_remote_docs() -> Iterable[Tuple[str, str]]:
    """Yield (path, markdown_text) for every .md file without downloading the repo."""
    with urllib.request.urlopen(API_URL, timeout=30) as resp:
        tree = json.loads(resp.read().decode("utf-8"))
    for entry in tree:
        if entry.get("type") == "file" and entry["path"].endswith(".md") and entry["path"] != "README.md":
            raw_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{urllib.parse.quote(entry['path'])}"
            with urllib.request.urlopen(raw_url, timeout=30) as r:
                yield entry["path"], r.read().decode("utf-8", errors="ignore")


def ingest_kb(store: SQLiteVectorStore | None = None) -> int:
    store = store or SQLiteVectorStore()
    texts, metas = [], []
    for path, content in _iter_remote_docs():
        parent = path.rsplit("/", 1)[0].lower() if "/" in path else ""
        domain = next((v for k, v in DOMAIN_MAP.items() if k in parent), "customer_support")
        for chunk in _chunk_markdown(content):
            texts.append(chunk)
            metas.append({"domain": domain, "source": path.rsplit("/", 1)[-1]})
    if texts:
        store.add_texts(texts, metas)
    return len(texts)
