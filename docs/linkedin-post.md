# LinkedIn Launch Post

> **Image attachment:** `docs/social-preview.png` (1280x640, 746 KB — meets LinkedIn's requirements)

---

## Recommended version

```text
The biggest bug in my RAG project wasn't in the bot. It was in the evaluator.

I built a customer-support Q&A system over real help-center documentation — retrieval, grounding, citations, refusal when the answer isn't there. Standard RAG stuff.

But the real project is the measurement loop around it.

The first "results" were a trap: my LLM judge spent its token budget on chain-of-thought reasoning, silently truncating its own JSON verdicts. 85% of scores were dropped. The correctness metric looked fine — it was just a mean over 12% of the data.

I only caught it because I instrumented the evaluator itself, not just the system under test.

Once the meter worked, I ran 5 iterations — each changing exactly one variable — over a 115-question hand-annotated dataset (factual, synthesis, and adversarial/unanswerable questions):

- Hybrid retrieval (vector + BM25 + cross-encoder rerank) pushed recall from 0.45 to 0.62
- A grounding prompt dropped hallucination on trick questions from 100% to 0%
- Citation validity went from 0.10 to 0.99

The grounded system scores lower on raw "correctness" — because it refuses to answer when it shouldn't, instead of confidently making things up. That's the right trade for anything customer-facing.

Built with Python, no LangChain in the main pipeline, direct SDK calls. A LangGraph variant is included for comparison.

Repo, dataset, eval harness, and all iteration results are open:
https://github.com/KazKozDev/airbnb-support-rag-eval

#GenerativeAI #RAG #LLMEvaluation #Python #AIEngineering
```

---

## Shorter version (recruiter-friendly)

```text
The biggest bug in my RAG project wasn't in the bot — it was in the evaluator.

I built a Q&A system over customer-support docs with retrieval, grounding, mandatory citations, and refusal when the answer isn't there.

The real work: a measured evaluation loop. 115 hand-annotated questions, 5 iterations each changing one variable, deterministic + LLM-judged metrics.

Key results:
- Hallucination on trick questions: 100% → 0% (one prompt change)
- Citation validity: 0.10 → 0.99
- Caught an 85%-silent-failure bug in my own evaluator before trusting any numbers

Open-source — code, dataset, and all eval results:
https://github.com/KazKozDev/airbnb-support-rag-eval

#GenerativeAI #RAG #LLMEvaluation #AIEngineering
```

---

## Technical version (RAG evaluation focus)

```text
Your RAG eval is probably lying to you. Mine was.

I built a customer-support RAG over help-center documentation — hybrid retrieval (RRF over vector + BM25), cross-encoder rerank, grounding contract with code-validated citations, explicit refusal token for unanswerable queries.

Then I built the evaluation harness: 115 hand-annotated questions (50 factual, 47 multi-chunk synthesis, 18 adversarial), recall@k, MRR, LLM-judged faithfulness/correctness, deterministic hallucination rate and citation validity.

First lesson: instrument the judge. My LLM judge (a reasoning model) burned its token budget on chain-of-thought and truncated the JSON verdict. 85% of scores silently dropped. The "correctness" number was a mean over ~12% of the data — noise with a confidence interval. Fixed by monitoring judge parse-rate and switching to a non-reasoning judge.

Second lesson: one variable per iteration. Five runs, each changing one thing:
- BM25 alone regressed vs. pure vector (latched onto repeated boilerplate) — reranking recovered it
- Grounding prompt: hallucination 1.00 → 0.00, citation validity 0.10 → 0.99
- Correctness drops with grounding — because safe refusals are "wrong" by the metric. The remaining bottleneck is retrieval recall, not generation

Judge calibration workflow (inter-annotator agreement between me and the LLM judge) is scaffolded in the repo as the next iteration.

Everything is open — code, golden dataset, annotation guidelines, judge rubric, per-question results:
https://github.com/KazKozDev/airbnb-support-rag-eval

#RAG #LLMEvaluation #GenerativeAI #Python
```

---

## Pre-posting checklist

- [ ] Attach `docs/social-preview.png` as the image when posting (1280x640, 746 KB)
- [ ] Verify GitHub repo is public: https://github.com/KazKozDev/airbnb-support-rag-eval
- [ ] Verify release `v1.0.0` exists: https://github.com/KazKozDev/airbnb-support-rag-eval/releases/tag/v1.0.0
- [ ] Paste the GitHub URL in the post body (already included in all versions above)
- [ ] Optionally pin the repo on your GitHub profile
- [ ] Upload `docs/social-preview.png` as the GitHub social preview (Settings → General → Social preview) if not already done
- [ ] Review and personalize the post — add or remove details based on what you want to emphasize
- [ ] Consider posting on a Tuesday–Thursday morning for better LinkedIn reach
