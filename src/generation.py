"""LLM answer generation.

Two prompt variants, switchable per call — this is what makes v3 an ablation
instead of a vibe: run the same eval with `naive` and `grounded` and the delta
in hallucination rate is attributable to the prompt + citation contract alone.
"""
import re
import time

from src import config
from src.providers import chat

PROMPTS = {
    # v0 baseline: what most tutorials ship. No refusal contract, no citations.
    "naive": (
        "You are a helpful assistant. Answer the user's question using the "
        "provided context chunks."
    ),
    # v3: grounding contract + mandatory citations + explicit refusal token.
    "grounded": f"""You are a question-answering assistant restricted to a single document.

Rules:
1. Answer ONLY using the context chunks provided. Never use outside knowledge.
2. After every claim, cite the supporting chunk like [c0012]. Citations are mandatory.
3. If the context does not contain the answer, reply with exactly: {config.REFUSAL_MARKER}
   followed by one short sentence explaining that the document does not cover this.
4. Be concise and precise. Quote the document where wording matters.""",
}

CITATION_RE = re.compile(r"\[(c\d{4})\]")


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens / 1e6 * config.PRICE_IN_PER_MTOK
        + output_tokens / 1e6 * config.PRICE_OUT_PER_MTOK,
        6,
    )


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] (page {c['page']})\n{c['text']}" for c in chunks)


def generate_answer(question: str, chunks: list[dict],
                    prompt_variant: str = config.PROMPT_VARIANT) -> dict:
    """Returns answer text + citation report + latency/cost telemetry."""
    model = config.active_model()
    t0 = time.time()
    res = chat(
        user=f"Context chunks:\n\n{build_context(chunks)}\n\nQuestion: {question}",
        system=PROMPTS[prompt_variant],
        model=model,
        max_tokens=config.MAX_TOKENS,
    )
    latency = time.time() - t0
    answer = res["text"]
    in_tok, out_tok = res["input_tokens"], res["output_tokens"]

    retrieved_ids = {c["id"] for c in chunks}
    cited = list(dict.fromkeys(CITATION_RE.findall(answer)))  # unique, ordered
    invalid = [c for c in cited if c not in retrieved_ids]
    refused = config.REFUSAL_MARKER in answer

    return {
        "answer": answer,
        "provider": config.LLM_PROVIDER,
        "model": model,
        "prompt_variant": prompt_variant,
        "refused": refused,
        "citations": cited,
        "invalid_citations": invalid,          # v3: hallucinated citations caught by code
        "citations_valid": not invalid and (bool(cited) or refused),
        "latency_s": round(latency, 2),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": estimate_cost_usd(in_tok, out_tok),
    }
