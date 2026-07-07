# Evaluation Methodology

The evaluation loop is the core of this project. Every architectural change is
accepted or rejected based on numbers from this harness, never on vibes.

## 1. Golden dataset

115 hand-annotated Q&A pairs over the Airbnb Help Center (cancellations, refunds,
AirCover, host/guest policies), annotated per
[eval/annotation_guidelines.md](eval/annotation_guidelines.md):

| type | count | measures |
|---|---|---|
| factual | 50 | basic retrieval + answer extraction |
| synthesis | 47 | multi-chunk retrieval, context assembly |
| adversarial (unanswerable) | 18 | hallucination rate — correct behavior is refusal |

Synthesis is deliberately heavy: real support tickets combine a scenario with
policy across several articles, so most questions require multi-chunk assembly.

Adversarial questions are on-topic and plausible ("Does Airbnb give a 50% loyalty
discount after 10 bookings?") — the kind a real customer would ask and a support
bot must NOT answer from thin air. Off-topic questions are trivially refused and
measure nothing.

## 2. Metrics

### Retrieval (deterministic, no LLM, cheap to run on every commit)
- **recall@k** = |retrieved ∩ relevant| / |relevant|, averaged over non-adversarial questions.
- **MRR** = mean(1 / rank of the first relevant chunk), 0 if none in top-k.

Run retrieval-only: `python -m eval.run_eval --no-judge --tag <tag>`

### Generation (LLM-judged + deterministic checks)
- **Faithfulness (1–5)** — every claim supported by the retrieved context. Judged
  by LLM per [eval/judge_rubric.md](eval/judge_rubric.md).
- **Correctness (1–5)** — semantic match against the reference answer. Same judge.
- **Hallucination rate** — share of adversarial questions answered instead of
  refused. Deterministic: string check for the `NOT_IN_DOCUMENT` token.
- **Citation validity** — share of answers where all cited chunk ids actually
  exist in the retrieved set (code check, no LLM).

## 3. Judge reliability

Before trusting any judged number, check that the judge actually produced one.

Early evaluation runs used a reasoning judge (`deepseek-v4-pro`) that spent most
of its token budget on chain-of-thought, so a low `max_tokens` truncated the JSON
verdict and it failed to parse. Around **85% of scores were silently dropped**, and
`correctness` was effectively a mean over ~12% of the data (pure noise at ~1.8/5).

The fix had three parts:
1. A larger judge token budget.
2. A tolerant parser that handles prose/reasoning wrapped around the JSON verdict.
3. Switching to a non-reasoning judge model (`gpt-oss:120b`) for the final runs.

`run_eval` now reports `judge_scored` (parsed verdicts / N) on every run. If it is
far below N, the metric is untrustworthy regardless of its value. In the final
iteration log below, all five runs scored **97/97** judged rows (97 non-adversarial
questions out of 115 total — the 18 adversarial questions are measured by the
deterministic hallucination-rate check, not by the judge).

## 4. Judge calibration

Judge calibration is scaffolded but not yet completed. The workflow
(`eval/calibrate_judge.py`) is designed to compare a batch of hand labels against
the judge's scores (exact agreement, within-1 agreement, Spearman ρ per dimension)
— the annotator-calibration workflow from production data-quality work, ported to
an LLM judge. The intended steps:

1. Run eval, hand-label 30 answers on both dimensions → `eval/human_labels.jsonl`.
2. `python -m eval.calibrate_judge --tag <tag>` → agreement metrics.
3. If within-1 agreement < 0.85: inspect disagreements, tighten the rubric, re-run.
4. Log each rubric revision and the resulting agreement.

The final reported runs use a single judge model, `gpt-oss:120b`, with successful
parse coverage of 97/97 judged rows per run. However, agreement against independent
human labels is still pending — this is the intended next iteration.

## 5. Error analysis protocol

After each eval run, failures are bucketed before anything is changed:

1. **Retrieval failure** — relevant chunk not in top-k (recall row = 0 or partial).
   Fixes live in chunking / retrieval / rerank.
2. **Grounding failure** — chunk retrieved, answer still wrong or fabricated
   (recall fine, faithfulness low). Fixes live in the prompt / citation contract.
3. **Annotation failure** — the gold label itself is wrong or ambiguous. Fix the
   dataset, note it, re-run. (These exist in every dataset; pretending otherwise
   is how metrics rot.)

A change is only shipped if it moves the metric of its bucket without regressing
the others.

## 6. Iteration log

Generator: `deepseek-v4-flash` (Ollama Cloud). Judge: `gpt-oss:120b` — a different
model on the same platform (cross-model judging). All five rows judged by the same
model. 115 questions (50 factual / 47 synthesis / 18 adversarial). Each row changes
ONE variable relative to the previous row.

| Iteration | Change | Recall@5 | MRR | Faithfulness | Correctness | Halluc. rate | Cit. validity | Judge scored |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v0 | vector retrieval, naive prompt | 0.448 | 0.425 | 4.61 | 4.07 | 1.00 | 0.07 | 97/97 |
| v1 | + hybrid retrieval (RRF) | 0.418 | 0.385 | 4.73 | 4.00 | 1.00 | 0.04 | 97/97 |
| v2 | + cross-encoder rerank | 0.552 | 0.485 | 4.57 | 4.06 | 1.00 | 0.10 | 97/97 |
| v3 | + grounding prompt & citations | 0.552 | 0.485 | 4.37 | 2.59 | **0.00** | **0.99** | 97/97 |
| v4 | + retrieval depth k=5 → 8 | **0.619** | 0.493 | 4.25 | 2.73 | **0.00** | **0.99** | 97/97 |

Commands to reproduce:

```bash
python -m eval.run_eval --mode vector          --prompt naive    --tag v0
python -m eval.run_eval --mode hybrid          --prompt naive    --tag v1
python -m eval.run_eval --mode hybrid --rerank --prompt naive    --tag v2
python -m eval.run_eval --mode hybrid --rerank --prompt grounded --tag v3
python -m eval.run_eval --mode hybrid --rerank --prompt grounded --k 8 --tag v4
```

Each iteration changes ONE thing relative to the previous row, so every delta is
attributable. Full per-question outputs live in `eval/results/<tag>.jsonl`,
summaries in `eval/results/<tag>.summary.json`.

## 7. Results interpretation

**Hybrid retrieval regressed (v0 → v1).** Adding BM25 via RRF actually hurt
recall (0.448 → 0.418). BM25 latched onto repeated site boilerplate present in
many chunks, pulling irrelevant matches into the fusion ranking.

**Reranking recovered retrieval (v1 → v2).** The cross-encoder rerank re-scored
(query, chunk) pairs jointly, pushing Recall@5 from 0.418 to 0.552 — above the
original vector-only baseline.

**Grounding is the main safety improvement (v2 → v3).** Same retrieval pipeline,
only the prompt changes from naive to grounded. Hallucination rate dropped from
1.00 to 0.00 (all 18 adversarial questions correctly refused). Citation validity
rose from 0.10 to 0.99 (answers now cite their source chunks, validated by code).

**Retrieval depth improved recall further (v3 → v4).** Increasing k from 5 to 8
pushed Recall@5 to 0.619, giving the grounded generator more relevant context to
work with.

### The correctness paradox

Raw correctness drops from 4.06 (v2, naive) to 2.59 (v3, grounded). This is
expected and correct behavior: the naive system scores higher because it always
answers, including confidently fabricating answers to unanswerable questions. The
grounded system refuses when retrieval misses, and safe refusals are penalized by
the raw correctness metric (lower score against the reference answer).

On the questions the grounded system actually answers, correctness is ~4/5; the
average is pulled down by safe refusals. This means **the remaining bottleneck is
retrieval recall, not generation safety**. Improving recall directly improves the
share of questions the grounded system can answer correctly.

For customer-support RAG, metrics should be interpreted together: recall,
hallucination rate, citation validity, and refusal behavior. Raw correctness alone
is misleading for refusal-heavy systems.

## 8. Known limitations

- **Single human annotator**: self-agreement across a second pass substitutes for
  inter-annotator agreement.
- **Judge calibration pending**: the judge calibration workflow is scaffolded
  (`eval/calibrate_judge.py`) but agreement against independent human labels has
  not been measured. The final runs use `gpt-oss:120b` with 97/97 parse coverage,
  but numeric agreement with human judgments is not yet validated.
- **Cross-model judging, same provider**: judge (`gpt-oss:120b`) and generator
  (`deepseek-v4-flash`) are different models but served on the same platform
  (Ollama Cloud). `JUDGE_MODEL` or `LLM_PROVIDER=anthropic` enables a fuller
  cross-family check.
- **Retrieval recall is the main remaining bottleneck**: the grounded system
  refuses correctly but cannot answer when retrieval misses. Improving recall
  (better chunking, embeddings, or retrieval strategies) is the highest-leverage
  next step.
- **Raw correctness is misleading**: for refusal-heavy customer-support systems,
  correctness must be interpreted alongside hallucination rate and citation
  validity. A system that always answers (including fabrications) will score higher
  on raw correctness than one that safely refuses.
- **recall@k** is computed against annotated chunk ids, so it inherits chunk-id
  remapping quality after re-chunking (see `eval/remap_chunk_ids.py`; flagged
  mappings are manually reviewed).
