"""LangGraph stateful RAG workflow: classify -> retrieve -> crew -> answer."""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from .config import settings
from .crews import run_domain_crew
from .state import RetailState
from .vectorstore import SQLiteVectorStore

DOMAIN_KEYWORDS = {
    "supply_chain": [
        "shipment", "delivery", "deliver", "carrier", "routing", "eta", "logistics",
        "warehouse", "shipping", "package", "freight", "truck", "dispatch", "track",
        "arrive", "distribution", "last-mile", "last mile", "cross-docking",
    ],
    "pricing": [
        "price", "pricing", "discount", "markdown", "promotion", "competitor",
        "clearance", "edlp", "promo", "deal", "coupon", "price match",
    ],
    "inventory": [
        "stock", "inventory", "replenishment", "restock", "on-shelf", "shrink",
        "cycle count", "planogram", "shelf", "stockout", "otif", "in stock", "in-stock",
    ],
    "customer_support": [
        "refund", "return", "complaint", "order status", "support", "warranty",
        "exchange", "escalation", "damaged", "broken", "membership", "cancel",
    ],
}

DOMAIN_PROTOTYPES = {
    "supply_chain": (
        "supply chain logistics shipment delivery carrier routing eta warehouse "
        "distribution center freight tracking last mile dispatch"
    ),
    "pricing": (
        "dynamic pricing price discount markdown promotion competitor "
        "price matching edlp clearance coupon"
    ),
    "inventory": (
        "store inventory operations stock replenishment on-shelf availability "
        "shrink cycle count planogram stockout restocking"
    ),
    "customer_support": (
        "customer support refund return complaint order status warranty "
        "escalation damaged exchange cancel membership"
    ),
}

_prototype_vectors = None


def _get_prototype_vectors() -> dict:
    """Lazily embed domain prototypes with the local HF embedding model."""
    global _prototype_vectors
    if _prototype_vectors is None:
        import numpy as np

        from .vectorstore import get_embeddings

        emb = get_embeddings()
        _prototype_vectors = {}
        for domain, text in DOMAIN_PROTOTYPES.items():
            v = np.asarray(emb.embed_query(text), dtype=np.float32)
            _prototype_vectors[domain] = v / (np.linalg.norm(v) + 1e-9)
    return _prototype_vectors


def _keyword_hits(q: str) -> dict:
    import re

    return {
        domain: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw), q))
        for domain, kws in DOMAIN_KEYWORDS.items()
    }


def _embedding_scores(q: str) -> dict:
    import numpy as np

    from .vectorstore import get_embeddings

    qv = np.asarray(get_embeddings().embed_query(q), dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    return {
        domain: float(qv @ v) for domain, v in _get_prototype_vectors().items()
    }


def classify_node(state: RetailState) -> dict:
    q = state["query"].lower()
    hits = _keyword_hits(q)
    scores = _embedding_scores(q)
    # Semantic similarity decides; a keyword hit adds a small boost (capped at one
    # per domain so repeated/prefix-overlapping keywords can't dominate).
    final = {
        d: scores[d] + (0.1 if hits[d] else 0.0) for d in scores
    }
    domain = max(final, key=lambda d: final[d])
    return {"domain": domain}


def retrieve_node(state: RetailState, store: SQLiteVectorStore | None = None) -> dict:
    store = store or SQLiteVectorStore()
    results = store.similarity_search(state["query"], k=settings.top_k)
    return {"retrieved_context": [r[0] for r in results]}


SMALL_TALK = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye", "goodbye", "how are you"}


def crew_node(state: RetailState) -> dict:
    if state["query"].strip().lower().rstrip("!?.") in SMALL_TALK:
        return {"crew_result": "Hi there! 👋 How can I help you with supply chain, pricing, inventory, or customer support today?"}
    context = "\n---\n".join(state["retrieved_context"])
    result = run_domain_crew(state["domain"], state["query"], context)
    return {"crew_result": result}


def answer_node(state: RetailState) -> dict:
    if state["crew_result"].startswith("Hi there!"):
        answer = state["crew_result"]
    else:
        answer = f"**{state['domain'].replace('_', ' ').title()}**\n\n{state['crew_result']}"
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def build_graph(store: SQLiteVectorStore | None = None):
    def _retrieve(state: RetailState) -> dict:
        return retrieve_node(state, store)

    graph = StateGraph(RetailState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("crew", crew_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "crew")
    graph.add_edge("crew", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_query(query: str, store: SQLiteVectorStore | None = None) -> RetailState:
    app = build_graph(store)
    initial: RetailState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "domain": "unknown",
        "retrieved_context": [],
        "crew_result": "",
        "final_answer": "",
        "metadata": {},
    }
    return app.invoke(initial)
