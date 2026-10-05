"""CrewAI domain crews backed by Hugging Face LLM endpoints."""
from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process, Task

from .config import settings


def get_hf_llm() -> LLM:
    """CrewAI LLM routed through Hugging Face Inference Providers."""
    os.environ.setdefault("HUGGINGFACE_API_KEY", settings.hf_token)
    return LLM(
        model=f"huggingface/{settings.hf_chat_model}",
        api_key=settings.hf_token,
        base_url="https://router.huggingface.co/v1",
    )


def _build_crew(role: str, goal: str, backstory: str, task_desc: str) -> Crew:
    llm = get_hf_llm()
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory + " You chat like a friendly, concise customer-service agent: match the user's intent and length, keep greetings and small talk to one short sentence, and only give detailed, structured answers for substantive operational questions.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            task_desc
            + "\n\nStyle rules: respond conversationally like a real chatbot. If the user just greets or makes small talk, reply with a single short sentence and offer help — no lists, no policy dumps. For real questions, answer directly in 2-4 sentences, citing retrieved context only when relevant."
        ),
        expected_output="A natural, human-length chatbot reply sized to the user's message.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def run_domain_crew(domain: str, query: str, context: str) -> str:
    ctx = context or "No retrieved context available."
    if domain == "supply_chain":
        crew = _build_crew(
            "Supply Chain Logistics Analyst",
            "Optimize last-mile routing, ETA, and carrier performance",
            "Senior Walmart logistics planner with 15 years of network operations experience.",
            f"Query: {query}\nContext:\n{ctx}\nAnswer with routing, carrier, and SLA guidance.",
        )
    elif domain == "pricing":
        crew = _build_crew(
            "Dynamic Pricing Strategist",
            "Recommend competitive, policy-compliant price adjustments",
            "Pricing science lead specializing in elasticity and markdown optimization.",
            f"Query: {query}\nContext:\n{ctx}\nAnswer with price change rationale and guardrails.",
        )
    elif domain == "inventory":
        crew = _build_crew(
            "Store Inventory Operations Manager",
            "Improve on-shelf availability and replenishment accuracy",
            "Store operations leader focused on OTIF, shrink, and planogram compliance.",
            f"Query: {query}\nContext:\n{ctx}\nAnswer with replenishment and cycle-count actions.",
        )
    else:
        crew = _build_crew(
            "Customer Support Resolution Specialist",
            "Resolve customer issues with empathy and policy compliance",
            "Customer care expert trained on returns, refunds, and service recovery.",
            f"Query: {query}\nContext:\n{ctx}\nAnswer with a resolution and next steps.",
        )
    return str(crew.kickoff())
