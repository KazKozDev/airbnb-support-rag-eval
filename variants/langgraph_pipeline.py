"""LangGraph rewrite of the pipeline — comparison branch material.

Purpose: same retrieve -> generate flow expressed as a LangGraph state graph,
to compare against the direct-SDK implementation in src/. Reuses the existing
Retriever, so retrieval metrics are identical by construction; the comparison
is about generation behavior, code ergonomics, and observability.

Install (separate from core requirements on purpose):
    pip install -r variants/requirements-langgraph.txt

Run:
    python -m variants.langgraph_pipeline "When do I get a full refund if I cancel?"

Evaluate it with the same golden dataset by swapping the generate call in
eval/run_eval.py for `variants.langgraph_pipeline.ask` (or add a --engine flag —
left as the exercise that makes the comparison honest).

Talking points the rewrite surfaces:
- The graph buys you checkpointing, retries and visual tracing for free, at the
  cost of an abstraction layer between you and the API call.
- Prompt and citation contract are IDENTICAL to src/generation.py — the
  framework must not silently change behavior, or the comparison is invalid.
"""
import sys
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src import config
from src.generation import PROMPTS, CITATION_RE, build_context
from src.retrieval import Retriever


class QAState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str
    citations: list[str]
    refused: bool


_retriever: Retriever | None = None
_llm = ChatAnthropic(model=config.LLM_MODEL, max_tokens=config.MAX_TOKENS)


def retrieve_node(state: QAState) -> dict:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    chunks = _retriever.retrieve(state["question"], k=config.TOP_K,
                                 mode=config.RETRIEVAL_MODE, rerank=config.RERANK)
    return {"chunks": chunks}


def generate_node(state: QAState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content=PROMPTS["grounded"]),
        HumanMessage(content=(
            f"Context chunks:\n\n{build_context(state['chunks'])}\n\n"
            f"Question: {state['question']}"
        )),
    ])
    answer = resp.content
    return {
        "answer": answer,
        "citations": list(dict.fromkeys(CITATION_RE.findall(answer))),
        "refused": config.REFUSAL_MARKER in answer,
    }


def build_graph():
    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def ask(question: str) -> dict:
    return build_graph().invoke({"question": question})


if __name__ == "__main__":
    result = ask(" ".join(sys.argv[1:]) or "When do I get a full refund if I cancel?")
    print(result["answer"])
    print("citations:", result["citations"], "| refused:", result["refused"])
