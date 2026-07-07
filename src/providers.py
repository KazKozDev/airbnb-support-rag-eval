"""LLM provider abstraction.

Both the answer generator (src/generation.py) and the eval judge
(eval/run_eval.py) call through here, so switching between Ollama Cloud and
Anthropic is a config flag, not a code change.

chat() returns a uniform dict: {"text", "input_tokens", "output_tokens"}.
list_models() returns the model names the provider's server actually offers,
which is what powers the interactive picker in src/select_model.py.
"""
import requests

from src import config

_anthropic_client = None


def _ollama_headers(api_key: str | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    key = config.OLLAMA_API_KEY if api_key is None else api_key
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # imported lazily so Ollama users need no anthropic key
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


# ---------- model discovery ----------
def list_models() -> list[str]:
    """Model names the server exposes. Ollama: GET /api/tags. Anthropic: /v1/models."""
    if config.LLM_PROVIDER == "ollama":
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags",
                            headers=_ollama_headers(), timeout=30)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names = [m.get("name") or m.get("model") for m in models]
        return [n for n in names if n]
    return [m.id for m in _get_anthropic().models.list().data]


# ---------- chat ----------
def chat(user: str, *, model: str, max_tokens: int, system: str | None = None,
         host: str | None = None, api_key: str | None = None) -> dict:
    """host/api_key override the generator endpoint (used to route the judge to a
    different Ollama server than the generator)."""
    if config.LLM_PROVIDER == "ollama":
        return _ollama_chat(user, system, model, max_tokens, host, api_key)
    return _anthropic_chat(user, system, model, max_tokens)


def _ollama_chat(user: str, system: str | None, model: str, max_tokens: int,
                 host: str | None = None, api_key: str | None = None) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    base = (host or config.OLLAMA_HOST).rstrip("/")
    resp = requests.post(f"{base}/api/chat",
                         headers=_ollama_headers(api_key), json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return {
        "text": data.get("message", {}).get("content", ""),
        # Ollama reports token counts only when the model backend provides them.
        "input_tokens": data.get("prompt_eval_count") or 0,
        "output_tokens": data.get("eval_count") or 0,
    }


def _anthropic_chat(user: str, system: str | None, model: str, max_tokens: int) -> dict:
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    msg = _get_anthropic().messages.create(**kwargs)
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {
        "text": text,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }
