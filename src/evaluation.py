"""RAGAS evaluation harness mapped to Hugging Face models."""
from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from datasets import Dataset

from .config import settings


def build_ragas_llm() -> LangchainLLMWrapper:
    chat = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=settings.hf_token,
        model=settings.hf_chat_model,
        temperature=0.1,
        max_tokens=512,
    )
    return LangchainLLMWrapper(chat)


def evaluate_rag(questions: List[str], answers: List[str], contexts: List[List[str]]):
    """Run faithfulness + answer_relevancy over a batch of QA examples."""
    llm = build_ragas_llm()
    embeddings = HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings
    return evaluate(ds, metrics=[faithfulness, answer_relevancy])
