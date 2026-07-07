# Golden Dataset Annotation Guidelines

## Purpose
The golden dataset is the ground truth for measuring retrieval and generation quality.
Every metric in EVALUATION.md depends on the quality of this annotation. Treat this
document the way you would treat guidelines for a team of annotators.

The document under test is the Airbnb Help Center (cancellations, refunds,
AirCover, host/guest policies, check-in, payments). Questions should sound like
real customer support tickets, not like the help article's own headings.

## Dataset composition (115 items)
- 50 factual: answer is stated in one place in the document (e.g. "how long do I
  have to report a listing problem?").
- 47 synthesis: answer requires combining information from 2+ chunks (e.g. a
  refund that depends on the cancellation policy AND the AirCover reporting window).
- 18 adversarial: plausible-sounding customer questions the Help Center does NOT
  answer (invented discounts, guarantees, direct contacts, out-of-policy demands,
  or topics simply not in the knowledge base). These measure hallucination rate.
  The correct system behavior is refusal / escalation, NOT inventing a policy.

## Fields
| field | rule |
|---|---|
| id | q + zero-padded number, unique |
| type | factual / synthesis / adversarial |
| question | natural phrasing a real user would type; do not copy document wording verbatim (that inflates BM25 scores) |
| reference_answer | complete, self-contained, verifiable against the document; empty for adversarial |
| relevant_chunk_ids | ALL chunks that contain the answer, not just one; empty for adversarial |

## How to find relevant_chunk_ids
Annotate AFTER ingestion, against data/chunks.jsonl for the chunking strategy under
test. Use `python -m eval.find_chunks "your question"` to shortlist candidates, then
verify by reading the chunk text. If the answer spans a chunk boundary, include both
chunks.

IMPORTANT: chunk ids change when the chunking strategy changes. Keep a mapping per
strategy or re-verify ids after re-chunking (see eval/README note in EVALUATION.md).

## Quality bar
- A second pass on your own annotations after 1+ day (self inter-annotator agreement).
- Ambiguous questions are rewritten or dropped, not "resolved by feeling".
- Adversarial questions must be on-topic and plausible; off-topic questions ("what
  is the capital of France") are trivially refused and measure nothing. A good
  adversarial item is one a support agent could be *tempted* to answer confidently
  but that has no basis in the Help Center (e.g. "can I get a free night for a
  cleanliness complaint?").
