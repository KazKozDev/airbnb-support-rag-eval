"""Optional observability via Langfuse (self-hosted or cloud).

Design: the pipeline never depends on Langfuse being installed or reachable.
If LANGFUSE_ENABLED=false, langfuse isn't installed, or init fails — a no-op
tracer is used and the app behaves identically. Enable with:

  pip install "langfuse>=2,<3"
  export LANGFUSE_ENABLED=true LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
         LANGFUSE_HOST=http://localhost:3000   # or cloud
"""
import sys

from src import config


class NoopTracer:
    def log_qa(self, result: dict) -> None:
        pass


class LangfuseTracer:
    def __init__(self):
        from langfuse import Langfuse  # imported lazily, optional dependency
        self.client = Langfuse()

    def log_qa(self, result: dict) -> None:
        try:
            trace = self.client.trace(
                name="docqa.ask",
                input={"question": result["question"],
                       "mode": result["retrieval_mode"], "k": result["k"]},
                output={"answer": result["answer"], "refused": result["refused"]},
                metadata={
                    "retrieved_chunks": result["retrieved_chunks"],
                    "citations": result["citations"],
                    "invalid_citations": result["invalid_citations"],
                    "prompt_variant": result.get("prompt_variant"),
                    "rerank": result.get("rerank"),
                },
            )
            trace.generation(
                name="answer",
                model=config.LLM_MODEL,
                usage={"input": result["input_tokens"],
                       "output": result["output_tokens"], "unit": "TOKENS"},
                metadata={"latency_s": result["latency_s"],
                          "cost_usd": result["cost_usd"]},
            )
        except Exception as e:  # tracing must never break the request path
            print(f"langfuse trace failed: {e}", file=sys.stderr)


_tracer = None


def get_tracer():
    global _tracer
    if _tracer is None:
        if config.LANGFUSE_ENABLED:
            try:
                _tracer = LangfuseTracer()
            except Exception as e:
                print(f"langfuse init failed, tracing disabled: {e}", file=sys.stderr)
                _tracer = NoopTracer()
        else:
            _tracer = NoopTracer()
    return _tracer
