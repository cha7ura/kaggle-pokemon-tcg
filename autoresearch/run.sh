#!/usr/bin/env bash
# Run one evaluation experiment inside the linux/amd64 engine container.
# Usage: ./run.sh [eval.py args...]
#   ./run.sh --games 400                       # challenger vs random baseline
#   ./run.sh --games 600 --champion champion_agent.py
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec docker run --rm --platform linux/amd64 \
  -v "$REPO":/app -w /app/autoresearch \
  -e PYTHONPATH=/app/sdk \
  python:3.11-slim python eval.py "$@"
