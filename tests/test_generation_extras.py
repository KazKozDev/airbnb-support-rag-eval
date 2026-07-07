"""Tests for v3 additions: prompt variants, cost telemetry, tracing fallback."""
from src import config
from src.generation import PROMPTS, estimate_cost_usd
from src.tracing import NoopTracer, get_tracer


def test_prompt_variants_exist_and_differ():
    assert set(PROMPTS) == {"naive", "grounded"}
    assert config.REFUSAL_MARKER in PROMPTS["grounded"]
    assert config.REFUSAL_MARKER not in PROMPTS["naive"]
    assert "cite" in PROMPTS["grounded"].lower()


def test_cost_estimate():
    # 1M in + 1M out at default $3/$15 per MTok
    assert estimate_cost_usd(1_000_000, 1_000_000) == config.PRICE_IN_PER_MTOK + config.PRICE_OUT_PER_MTOK
    assert estimate_cost_usd(0, 0) == 0.0


def test_tracer_defaults_to_noop():
    # LANGFUSE_ENABLED is false by default -> tracing must be a harmless no-op
    tracer = get_tracer()
    assert isinstance(tracer, NoopTracer)
    tracer.log_qa({"anything": True})  # must not raise
