"""Interactive model picker.

Asks the provider's server which models it offers (Ollama Cloud: GET /api/tags)
and lets you choose one from a numbered menu. The choice is saved to
data/selected_model.txt and picked up automatically by /ask and the eval runner.

    python -m src.select_model

Override without the menu at any time with the OLLAMA_MODEL env var.
"""
import sys

from src import config
from src.providers import list_models


def main() -> None:
    try:
        models = list_models()
    except Exception as e:  # noqa: BLE001 - report cleanly instead of a traceback
        print(f"Could not fetch models from {config.OLLAMA_HOST}: {e}")
        print("Check OLLAMA_API_KEY and OLLAMA_HOST in your .env, then retry.")
        sys.exit(1)

    if not models:
        print("The server returned no models.")
        sys.exit(1)

    current = config.active_model()
    print(f"Provider: {config.LLM_PROVIDER}    Host: {config.OLLAMA_HOST}\n")
    print("Models offered by the server:\n")
    for i, name in enumerate(models, 1):
        marker = "   <- current" if name == current else ""
        print(f"  {i:2}. {name}{marker}")

    raw = input(f"\nPick a model [1-{len(models)}] (Enter to keep current): ").strip()
    if not raw:
        print(f"Kept current model: {current}")
        return
    try:
        model = models[int(raw) - 1]
        if int(raw) < 1:
            raise IndexError
    except (ValueError, IndexError):
        print("Invalid choice, nothing changed.")
        sys.exit(1)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SELECTED_MODEL_PATH.write_text(model, encoding="utf-8")
    print(f"\nSaved '{model}' -> {config.SELECTED_MODEL_PATH}")
    print("It will be used by POST /ask and `python -m eval.run_eval` automatically.")


if __name__ == "__main__":
    main()
