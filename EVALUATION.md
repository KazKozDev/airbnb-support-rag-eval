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

## 3. Judge health (instrument the meter first)

Before trusting any judged number, check that the judge actually produced one.
A reasoning judge (`deepseek-v4-pro`) spends most of its token budget thinking, so
a low `max_tokens` truncated the JSON verdict and it failed to parse — **~85% of
scores were silently dropped**, and `correctness` was a mean over ~12% of the data
(pure noise at ≈1.8/5). The fix: a larger judge token budget + a parser that
tolerates prose/reasoning around the JSON, which cut parse errors 85% → 5% and
revealed the true correctness (2.67/5). `run_eval` reports the judged-row count;
if it is far below N, the metric is untrustworthy regardless of its value.

## 4. Judge calibration

The LLM judge is treated like a new annotator: it doesn't grade production data
until it agrees with the gold rater (me).

1. Run eval, hand-label 30 answers on both dimensions → `eval/human_labels.jsonl`.
2. `python -m eval.calibrate_judge --tag <tag>` → exact agreement, within-1
   agreement, Spearman ρ per dimension.
3. If within-1 agreement < 0.85: inspect disagreements, tighten the rubric
   (usually: an underspecified boundary between adjacent scores), re-run.
4. Log each rubric revision and the resulting agreement here:

| rubric version | faithfulness within-1 | correctness within-1 | change made |
|---|---|---|---|
| r1 | — | — | initial rubric |

This is the annotator-calibration workflow from production data-quality work,
ported to an LLM judge.

## 4. Error analysis protocol

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

## 5. Iteration log

| Iteration | command | Recall@5 | MRR | Faithfulness | Halluc. rate | Cit. validity |
|---|---|---|---|---|---|---|
| v0 naive | `--mode vector --prompt naive --tag v0` | — | — | — | — | — |
| v1 hybrid | `--mode hybrid --prompt naive --tag v1` | — | — | — | — | — |
| v2 chunking+rerank | `--mode hybrid --rerank --prompt naive --tag v2` | — | — | — | — | — |
| v3 grounding | `--mode hybrid --rerank --prompt grounded --tag v3` | — | — | — | — | — |

Each iteration changes ONE thing relative to the previous row, so every delta is
attributable. Full per-question outputs live in `eval/results/<tag>.jsonl`,
summaries in `eval/results/<tag>.summary.json`.

## 6. Known limitations

- Single human annotator: self-agreement across a second pass substitutes for
  inter-annotator agreement.
- Judge and generator are different models (`deepseek-v4-pro` judging
  `deepseek-v4-flash`) to reduce self-preference bias, and correctness is anchored
  to a written reference. Both are still from one family/provider; `JUDGE_MODEL`
  (or `LLM_PROVIDER=anthropic` for the judge run) enables a fuller cross-family
  check.
- recall@k is computed against annotated chunk ids, so it inherits chunk-id
  remapping quality after re-chunking (see `eval/remap_chunk_ids.py`; flagged
  mappings are manually reviewed).
