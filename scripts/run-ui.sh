#!/usr/bin/env bash
# Start the Next.js UI dev server from the ui/ folder.
# Ensures the workspace python venv is activated first (if present), as requested.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
else
  echo "Warning: virtualenv not found at $VENV. Proceeding without activating venv."
fi

cd "$REPO_ROOT/ui"
# Use npm; if you prefer pnpm or yarn update this script
npm run dev
