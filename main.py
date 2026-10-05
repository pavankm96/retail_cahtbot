"""End-to-end demo: seed SQLite vector store, answer a query, run RAGAS."""
from __future__ import annotations

from src.config import settings
from src.graph import run_query
from src.kb_loader import ingest_kb
from src.vectorstore import SQLiteVectorStore


def main() -> None:
    settings.validate()
    store = SQLiteVectorStore()
    samples = store.similarity_search("returns", k=1)
    if not samples:
        n = ingest_kb(store)
        print(f"Ingested {n} KB chunks.")

    result = run_query("My order arrived damaged, how do I get a refund?", store)
    print("Domain:", result["domain"])
    print("Answer:", result["final_answer"])


if __name__ == "__main__":
    main()
