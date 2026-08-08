#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV="$ROOT/venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "GymForm — setup and start"
echo "Project: $ROOT"
echo

if [[ ! -d "$VENV" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi

echo "Installing dependencies..."
"$PIP" install -q -r requirements.txt

echo
echo "Starting API + UI at http://localhost:8080"
echo "Press Ctrl+C to stop."
echo

exec "$PYTHON" api_server.py
