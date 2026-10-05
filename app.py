"""Streamlit chat frontend for the Walmart multi-agent retail platform."""
from __future__ import annotations

import streamlit as st

from src.config import settings
from src.graph import run_query
from src.kb_loader import ingest_kb
from src.vectorstore import SQLiteVectorStore

st.set_page_config(page_title="Walmart Retail AI Orchestrator", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .domain-badge {display:inline-block;padding:4px 12px;border-radius:999px;
        background:#0071CE;color:white;font-weight:600;font-size:13px;}
      .source-card {background:#f6f7f9;border-left:4px solid #0071CE;
        padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 Walmart Retail AI Orchestrator")
st.caption("LangGraph + CrewAI + Hugging Face RAG over the pavankm96/KB knowledge base")


@st.cache_resource
def get_store() -> SQLiteVectorStore:
    store = SQLiteVectorStore()
    if not store.similarity_search("policy", k=1):
        n = ingest_kb(store)
        st.sidebar.success(f"Ingested {n} KB chunks from pavankm96/KB")
    return store


with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"Model: `{settings.hf_chat_model}`")
    st.write(f"Embeddings: `{settings.hf_embedding_model}`")
    if st.button("🔄 Re-ingest KB"):
        import os

        st.cache_resource.clear()
        db = settings.sqlite_db_path
        try:
            os.remove(db)
        except FileNotFoundError:
            pass
        store = SQLiteVectorStore()  # recreates the table
        n = ingest_kb(store)
        st.success(f"Knowledge base re-ingested ({n} chunks).")
    st.divider()
    st.markdown("**Domains:** Supply Chain · Pricing · Inventory · Customer Support")

try:
    store = get_store()
except Exception as e:
    st.error(f"Failed to initialize store: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

query = st.chat_input("Ask about supply chain, pricing, inventory, or customer support…")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Agents deliberating…"):
            try:
                result = run_query(query, store)
                st.markdown(
                    f"<span class='domain-badge'>{result['domain']}</span>", unsafe_allow_html=True
                )
                st.markdown(result["final_answer"])
                if result["retrieved_context"]:
                    with st.expander("📚 Retrieved KB sources"):
                        for i, ctx in enumerate(result["retrieved_context"], 1):
                            st.markdown(f"<div class='source-card'><b>Chunk {i}</b><br>{ctx[:400]}…</div>",
                                        unsafe_allow_html=True)
                st.session_state.history.append(("assistant", result["final_answer"]))
            except Exception as e:
                st.error(f"Error: {e}")
