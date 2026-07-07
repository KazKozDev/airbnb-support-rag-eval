"""Evaluation runner.

Golden dataset format (eval/golden_dataset.jsonl), one JSON object per line:
{
  "id": "q001",
  "type": "factual" | "synthesis" | "adversarial",
  "question": "...",
  "reference_answer": "...",          # empty for adversarial
  "relevant_chunk_ids": ["c0012"]     # empty for adversarial
}

Metrics:
- Retrieval: recall@k, MRR (deterministic, no LLM, adversarial excluded)
- Generation: faithfulness (LLM judge, 1-5), answer correctness (LLM judge, 1-5),
  hallucination rate on adversarial questions (answered instead of refusing)

Usage:
  python -m eval.run_eval --k 5 --mode hybrid --tag "v1-hybrid"
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from src import config
from src.generation import generate_answer
from src.providers import chat
from src.retrieval import Retriever

RUBRIC = (Path(__file__).parent / "judge_rubric.md").read_text(encoding="utf-8")


def load_golden() -> list[dict]:
    with open(config.GOLDEN_DATASET, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- Retrieval metrics ----------
def recall_at_k(retrieved: list[str], relevant: list[str]) -> float | None:
    if not relevant:
        return None
    return len(set(retrieved) & set(relevant)) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


# ---------- LLM-as-judge ----------
def judge(question: str, context: str, answer: str, reference: str) -> dict:
    """Returns {"faithfulness": 1-5, "correctness": 1-5, "rationale": str}."""
    prompt = f"""{RUBRIC}

Question: {question}

Context given to the system:
{context}

Reference answer (gold): {reference or "(none)"}

System answer: {answer}

Respond with ONLY a JSON object: {{"faithfulness": <1-5>, "correctness": <1-5>, "rationale": "<one sentence>"}}"""
    # Reasoning judges (e.g. deepseek-v4-pro) spend most of their output budget
    # "thinking" before emitting the JSON, so a low cap truncates the object and
    # the score is silently dropped — the #1 cause of a biased, tiny-sample metric.
    # 2000 leaves room for the reasoning trace plus the JSON.
    res = chat(user=prompt, model=config.active_judge_model(), max_tokens=2000,
               host=config.OLLAMA_JUDGE_HOST, api_key=config.OLLAMA_JUDGE_API_KEY)
    return _parse_judge(res["text"])


def _parse_judge(text: str) -> dict:
    """Robustly pull the score object out of the judge's reply.

    Some judge models wrap the JSON in prose or reasoning, so a plain json.loads
    over the whole string fails. Fall back to extracting the first {...} block
    that contains the expected keys.
    """
    cleaned = text.replace("```json", "").replace("```", "").strip()
    for candidate in (cleaned, _extract_json_block(cleaned)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if "faithfulness" in obj or "correctness" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return {"faithfulness": None, "correctness": None, "rationale": "judge parse error"}


def _extract_json_block(text: str) -> str | None:
    m = re.search(r'\{[^{}]*"(?:faithfulness|correctness)"[^{}]*\}', text, re.S)
    return m.group(0) if m else None


# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--mode", default=config.RETRIEVAL_MODE)
    ap.add_argument("--tag", default="untagged", help="iteration label, e.g. v1-hybrid")
    ap.add_argument("--no-judge", action="store_true", help="retrieval metrics only")
    ap.add_argument("--rerank", action="store_true", help="v2: cross-encoder rerank")
    ap.add_argument("--prompt", default=config.PROMPT_VARIANT,
                    choices=["naive", "grounded"], help="v3 ablation: prompt variant")
    args = ap.parse_args()

    retriever = Retriever()
    golden = load_golden()
    rows = []

    for item in golden:
        chunks = retriever.retrieve(item["question"], k=args.k, mode=args.mode,
                                    rerank=args.rerank)
        retrieved_ids = [c["id"] for c in chunks]
        row = {
            "id": item["id"],
            "type": item["type"],
            "recall": recall_at_k(retrieved_ids, item.get("relevant_chunk_ids", [])),
            "rr": reciprocal_rank(retrieved_ids, item.get("relevant_chunk_ids", []))
                  if item.get("relevant_chunk_ids") else None,
        }
        if not args.no_judge:
            gen = generate_answer(item["question"], chunks, prompt_variant=args.prompt)
            row["model"] = gen["model"]
            row["refused"] = gen["refused"]
            row["citations_valid"] = gen["citations_valid"]
            row["answer"] = gen["answer"]
            row["cost_usd"] = gen["cost_usd"]
            if item["type"] == "adversarial":
                row["hallucinated"] = not gen["refused"]
            else:
                context = "\n".join(c["text"] for c in chunks)
                row.update(judge(item["question"], context, gen["answer"],
                                 item.get("reference_answer", "")))
        rows.append(row)
        print(f"  {item['id']} done")

    # ---------- Aggregate ----------
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 3) if vals else None

    report = {
        "tag": args.tag,
        "k": args.k,
        "mode": args.mode,
        "rerank": args.rerank,
        "prompt": args.prompt,
        "n": len(rows),
        "recall_at_k": mean([r["recall"] for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
    }
    if not args.no_judge:
        # Self-document the run: which model generated, which judged.
        report["provider"] = config.LLM_PROVIDER
        report["generator"] = config.active_model()
        report["judge"] = config.active_judge_model()
        if config.LLM_PROVIDER == "ollama":
            report["generator_host"] = config.OLLAMA_HOST
            report["judge_host"] = config.OLLAMA_JUDGE_HOST
        adv = [r for r in rows if r["type"] == "adversarial"]
        judged = [r for r in rows if r["type"] != "adversarial"]
        scored = [r for r in judged if r.get("correctness") is not None]
        # Judge health: if scored << judged, the judge failed to return parseable
        # verdicts and faithfulness/correctness are means over a biased sub-sample.
        report["judge_scored"] = f"{len(scored)}/{len(judged)}"
        report["faithfulness"] = mean([r.get("faithfulness") for r in rows])
        report["correctness"] = mean([r.get("correctness") for r in rows])
        report["hallucination_rate"] = mean([float(r["hallucinated"]) for r in adv])
        report["citation_validity"] = mean([float(r["citations_valid"]) for r in rows])
        report["total_generation_cost_usd"] = round(
            sum(r.get("cost_usd", 0) for r in rows), 4)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{args.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / f"{args.tag}.summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
