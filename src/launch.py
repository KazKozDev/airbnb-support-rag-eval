"""Interactive launcher for docqa-eval.

Run by double-clicking launch.command (which opens Terminal and calls this).
Walks you through: provider -> model (from the server's list) -> what to run.
Choices are saved to .env so the next launch remembers them.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"


def set_env_var(key: str, value: str) -> None:
    """Update or append KEY=value in .env and set it for the current process."""
    lines, found = [], False
    if ENV.exists():
        for ln in ENV.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s and not s.startswith("#") and s.split("=", 1)[0].strip() == key:
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(ln)
    if not found:
        lines.append(f"{key}={value}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value


def choose(title: str, options: list[str]) -> int:
    """Print a numbered menu, return the chosen index (0-based)."""
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i:2}. {o}")
    while True:
        raw = input(f"Enter a number [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  Invalid choice, try again.")


def main() -> None:
    print("\n=== docqa-eval launcher ===")

    # 1) provider ---------------------------------------------------------
    provider = ["ollama", "anthropic"][choose(
        "Choose a provider:", ["Ollama Cloud", "Anthropic"])]
    set_env_var("LLM_PROVIDER", provider)

    # import config AFTER the provider env is set, so its constants are right
    from src import config

    # 2) key check --------------------------------------------------------
    if provider == "ollama" and not config.OLLAMA_API_KEY:
        key = input("\nOLLAMA_API_KEY is not set. Paste it now (or Enter to skip): ").strip()
        if key:
            set_env_var("OLLAMA_API_KEY", key)
            config.OLLAMA_API_KEY = key
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        key = input("\nANTHROPIC_API_KEY is not set. Paste it now (or Enter to skip): ").strip()
        if key:
            set_env_var("ANTHROPIC_API_KEY", key)

    # 3) model (from the server's list) ----------------------------------
    from src.providers import list_models
    try:
        models = list_models()
    except Exception as e:  # noqa: BLE001
        print(f"\nCouldn't fetch the model list: {e}")
        models = []
    if models:
        model = models[choose("Choose a model:", models)]
    else:
        model = input("\nType a model id manually: ").strip()
    if provider == "ollama":
        set_env_var("OLLAMA_MODEL", model)
    else:
        set_env_var("LLM_MODEL", model)
    print(f"\n-> provider={provider}  model={model}")

    # 4) what to run ------------------------------------------------------
    action = choose("What do you want to run?", [
        "Start the API server (POST /ask on :8000)",
        "Ask questions right here in the terminal",
        "Run the full evaluation (all 115 questions)",
    ])

    if action == 0:
        import uvicorn
        print("\nStarting server on http://localhost:8000 ... (Ctrl+C to stop)\n")
        uvicorn.run("src.api:app", host="0.0.0.0", port=8000)
    elif action == 1:
        from src.pipeline import ask
        print("\nAsk a question (empty line to quit):")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            r = ask(q, mode="hybrid", rerank=True, prompt_variant="grounded")
            print("\n" + r["answer"])
            print(f"\n[model={r['model']} | {r['latency_s']}s | "
                  f"cites={r['citations']} | refused={r['refused']}]")
    else:
        import subprocess
        subprocess.run([sys.executable, "-m", "eval.run_eval",
                        "--mode", "hybrid", "--rerank",
                        "--prompt", "grounded", "--tag", "launch-run"], cwd=ROOT)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
