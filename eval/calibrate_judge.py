"""Judge calibration: compare human labels vs LLM-judge labels.

Workflow:
1. Run eval, open eval/results/<tag>.jsonl
2. Copy ~30 rows into eval/human_labels.jsonl adding your own scores:
   {"id": "q001", "human_faithfulness": 5, "human_correctness": 4}
3. Run: python -m eval.calibrate_judge --tag v1-hybrid

Reports exact agreement, within-1 agreement, and Spearman correlation.
If agreement is low -> fix the rubric, re-run, repeat. (Same loop as
calibration sessions with human annotators.)
"""
import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def agreement(human: list[int], judge: list[int]) -> dict:
    n = len(human)
    exact = sum(h == j for h, j in zip(human, judge)) / n
    within1 = sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / n
    rho = spearmanr(human, judge).statistic if n > 2 else None
    return {"n": n, "exact": round(exact, 3), "within_1": round(within1, 3),
            "spearman": round(rho, 3) if rho is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    base = Path(__file__).parent
    judge_rows = {r["id"]: r for r in load_jsonl(base / "results" / f"{args.tag}.jsonl")}
    human_rows = load_jsonl(base / "human_labels.jsonl")

    for dim in ("faithfulness", "correctness"):
        pairs = [(h[f"human_{dim}"], judge_rows[h["id"]].get(dim))
                 for h in human_rows
                 if h.get(f"human_{dim}") is not None
                 and judge_rows.get(h["id"], {}).get(dim) is not None]
        if not pairs:
            print(f"{dim}: no overlapping labels")
            continue
        human, judge = zip(*pairs)
        print(f"{dim}: {agreement(list(human), list(judge))}")


if __name__ == "__main__":
    main()
