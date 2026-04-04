#!/usr/bin/env bash
# Run on a GPU node via Slurm (see ollama_proxy.py). One request per job.
set -euo pipefail

REQ_FILE="${1:?usage: ollama_infer.sh /path/to/request.json}"
OLLAMA_BASE="${OLLAMA_BASE:?}"

OLLAMA_BIN="$OLLAMA_BASE/ollama_bin/ollama"
export OLLAMA_MODELS="$OLLAMA_BASE/ollama_models"
export OLLAMA_HOST=127.0.0.1:11436

if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "{\"error\":{\"message\":\"Ollama binary not found or not executable: $OLLAMA_BIN\"}}" >&2
  exit 1
fi

"$OLLAMA_BIN" serve >/dev/null 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT

# Cold start can be slow on first load
sleep "${OLLAMA_SERVE_WAIT:-4}"

curl -sS -f -X POST "http://127.0.0.1:11436/api/chat" \
  -H "Content-Type: application/json" \
  --data-binary @"$REQ_FILE" \
  || { echo "{\"error\":{\"message\":\"curl to ollama failed (exit $?)\"}}"; exit 1; }
