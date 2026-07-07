"""Remap relevant_chunk_ids in the golden dataset after re-chunking.

Chunk ids are positional, so changing the chunking strategy invalidates the
annotation. This tool maps each old relevant chunk to the new chunk(s) with the
highest word-overlap (Jaccard), deterministically — no LLM involved.

Workflow for v2:
  cp data/chunks.jsonl data/chunks.v0.jsonl        # keep the old chunking
  python -m src.ingest doc.pdf --strategy structure
  python -m src.index
  python -m eval.remap_chunk_ids \
      --old data/chunks.v0.jsonl --new data/chunks.jsonl \
      --golden eval/golden_dataset.jsonl --out eval/golden_dataset.v2.jsonl

Then MANUALLY verify low-confidence mappings (flagged in stderr) before running
eval — automated remapping is a shortlist, not ground truth.
"""
import argparse
import json
import sys


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="old chunks.jsonl")
    ap.add_argument("--new", required=True, help="new chunks.jsonl")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-overlap", type=float, default=0.3,
                    help="flag mappings below this Jaccard for manual review")
    args = ap.parse_args()

    old_chunks = {c["id"]: set(c["text"].lower().split()) for c in load_jsonl(args.old)}
    new_chunks = [(c["id"], set(c["text"].lower().split())) for c in load_jsonl(args.new)]
    golden = load_jsonl(args.golden)

    flagged = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for item in golden:
            new_ids = []
            for old_id in item.get("relevant_chunk_ids", []):
                old_words = old_chunks.get(old_id)
                if old_words is None:
                    print(f"WARN {item['id']}: {old_id} not in old chunks", file=sys.stderr)
                    continue
                best_id, best_score = max(
                    ((nid, jaccard(old_words, nwords)) for nid, nwords in new_chunks),
                    key=lambda x: x[1],
                )
                if best_score < args.min_overlap:
                    flagged += 1
                    print(f"REVIEW {item['id']}: {old_id} -> {best_id} "
                          f"(jaccard={best_score:.2f})", file=sys.stderr)
                if best_id not in new_ids:
                    new_ids.append(best_id)
            f.write(json.dumps({**item, "relevant_chunk_ids": new_ids},
                               ensure_ascii=False) + "\n")

    print(f"Remapped {len(golden)} items -> {args.out}; {flagged} need manual review")


if __name__ == "__main__":
    main()
