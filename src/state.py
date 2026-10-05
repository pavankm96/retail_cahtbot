"""Immutable LangGraph conversational state definition."""
from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RetailState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    domain: str  # supply_chain | pricing | inventory | customer_support | unknown
    retrieved_context: List[str]
    crew_result: str
    final_answer: str
    metadata: Dict[str, str]
