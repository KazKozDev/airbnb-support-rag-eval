"""Provider abstraction: Ollama Cloud request/response mapping and model listing.

Network is mocked, so these run offline and assert we build the right payload and
read token counts back correctly."""
from unittest.mock import MagicMock, patch

from src import config, providers


def test_ollama_chat_builds_payload_and_parses_tokens():
    fake = MagicMock()
    fake.json.return_value = {
        "message": {"role": "assistant", "content": "Full refund 24h before check-in."},
        "prompt_eval_count": 120,
        "eval_count": 17,
    }
    fake.raise_for_status = MagicMock()

    with patch.object(config, "LLM_PROVIDER", "ollama"), \
         patch.object(providers, "requests") as req:
        req.post.return_value = fake
        out = providers.chat(user="When do I get a refund?", system="Be grounded.",
                             model="gpt-oss:120b", max_tokens=256)

    url = req.post.call_args.args[0]
    payload = req.post.call_args.kwargs["json"]
    assert url.endswith("/api/chat")
    assert payload["model"] == "gpt-oss:120b"
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 256
    assert payload["messages"][0] == {"role": "system", "content": "Be grounded."}
    assert payload["messages"][1]["role"] == "user"
    assert out == {"text": "Full refund 24h before check-in.",
                   "input_tokens": 120, "output_tokens": 17}


def test_ollama_missing_token_counts_default_to_zero():
    fake = MagicMock()
    fake.json.return_value = {"message": {"content": "hi"}}  # no *_eval_count
    with patch.object(config, "LLM_PROVIDER", "ollama"), \
         patch.object(providers, "requests") as req:
        req.post.return_value = fake
        out = providers.chat(user="q", model="m", max_tokens=8)
    assert out["input_tokens"] == 0 and out["output_tokens"] == 0


def test_list_models_reads_server_tags():
    fake = MagicMock()
    fake.json.return_value = {"models": [{"name": "gpt-oss:120b"}, {"name": "qwen3:8b"}]}
    with patch.object(config, "LLM_PROVIDER", "ollama"), \
         patch.object(providers, "requests") as req:
        req.get.return_value = fake
        names = providers.list_models()
    assert names == ["gpt-oss:120b", "qwen3:8b"]
    assert req.get.call_args.args[0].endswith("/api/tags")
