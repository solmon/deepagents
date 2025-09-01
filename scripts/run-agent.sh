#!/usr/bin/env bash
# Run the FastAPI agent server using the workspace virtualenv
# Usage: ./scripts/run-agent.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
else
  echo "Warning: virtualenv not found at $VENV. Proceeding without activating venv."
fi

# Run the FastAPI server (main FastAPI app is in src/api.py)
python "$REPO_ROOT/src/api.py"
