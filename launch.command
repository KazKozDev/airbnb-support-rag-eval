#!/bin/bash
# Double-click this file in Finder to launch docqa-eval.
# It opens in Terminal, then asks you to pick a provider, a model, and what to run.

cd "$(dirname "$0")" || exit 1

# Use the project's virtualenv if one exists, otherwise fall back to system python3.
if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

PY="${PYTHON:-python3}"

echo "Working dir: $(pwd)"
"$PY" -m src.launch

echo
read -n 1 -s -r -p "Done. Press any key to close this window..."
echo
