#!/usr/bin/env bash
# Push the working tree to the GPU box (code only; data lives remotely).
set -euo pipefail
REMOTE=${REMOTE:-root@<SERVER_A_HOST>}
PORT=${PORT:-<SERVER_A_PORT>}
rsync -az --delete -e "ssh -p $PORT" \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude .ruff_cache --exclude .env \
  "$(dirname "$0")/../" "$REMOTE:/workspace/simbig/repo/"
