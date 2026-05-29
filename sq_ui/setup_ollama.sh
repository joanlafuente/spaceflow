#!/usr/bin/env bash
#
# One-time setup: Ollama binary + model weights in shared course storage (cluster).
# After this, start the proxy from the printed path, then run the SQ UI dev server.
#
# Usage:
#   cd /work/courses/3dv/team3/spaceflow
#   bash sq_ui/setup_ollama.sh
#
# Optional environment:
#   SQ_OLLAMA_BASE    — install root (default <spaceflow>/spaceflow_runtime/superquadric_ui)
#   SQ_OLLAMA_SCRATCH — legacy alias for SQ_OLLAMA_BASE
#   OLLAMA_VERSION   — release tag (default v0.20.2)
#   OLLAMA_MODEL     — model to pull (default gemma4:e2b)
#   SKIP_DOWNLOAD    — if set to 1, skip tarball download (reuse existing bin)
#   SKIP_PULL        — if set to 1, skip model pull (reuse existing models dir)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEFLOW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE="${SQ_OLLAMA_BASE:-${SQ_OLLAMA_SCRATCH:-$SPACEFLOW_ROOT/spaceflow_runtime/superquadric_ui}}"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.20.2}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:e2b}"
TARBALL_URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"

echo "Install root: $BASE"
mkdir -p "$BASE"/{ollama_bin,ollama_models,scripts,logs,tmp}
# Keep large downloads and any pip/temp usage off $HOME and small /tmp (clusters).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASE/tmp/pip-cache}"
export TMPDIR="${TMPDIR:-$BASE/tmp}"
mkdir -p "$PIP_CACHE_DIR"
TARBALL="$BASE/tmp/ollama-linux-amd64-${OLLAMA_VERSION}.tar.zst"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo "Downloading Ollama ${OLLAMA_VERSION} ..."
  curl -fL "$TARBALL_URL" -o "$TARBALL"
  rm -rf "$BASE/ollama_bin"/*
  mkdir -p "$BASE/ollama_bin"
  tar --zstd -xf "$TARBALL" -C "$BASE/ollama_bin"
  rm -f "$TARBALL"

  if [[ -f "$BASE/ollama_bin/bin/ollama" ]]; then
    mv "$BASE/ollama_bin/bin/ollama" "$BASE/ollama_bin/ollama"
    rm -rf "$BASE/ollama_bin/bin"
  fi
  chmod +x "$BASE/ollama_bin/ollama"
else
  echo "SKIP_DOWNLOAD=1 — not unpacking tarball"
  chmod +x "$BASE/ollama_bin/ollama" 2>/dev/null || true
fi

echo "Installing scripts (OLLAMA_BASE=$BASE) ..."
mkdir -p "$BASE/scripts"
for f in ollama_infer.sh ollama_proxy.py; do
  src="$SCRIPT_DIR/scripts_templates/$f"
  if [[ ! -f "$src" ]]; then
    echo "Missing template: $src" >&2
    exit 1
  fi
  sed "s|__OLLAMA_BASE__|$BASE|g" "$src" >"$BASE/scripts/$f"
done
chmod +x "$BASE/scripts/ollama_infer.sh" "$BASE/scripts/ollama_proxy.py"

if [[ "${SKIP_PULL:-0}" != "1" ]]; then
  echo "Pulling model $OLLAMA_MODEL (large download) ..."
  export OLLAMA_MODELS="$BASE/ollama_models"
  export OLLAMA_HOST=127.0.0.1:11436
  "$BASE/ollama_bin/ollama" serve >"$BASE/tmp/ollama_setup_serve.log" 2>&1 &
  PID=$!
  sleep 3
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ollama serve failed to start. Last log:" >&2
    tail -20 "$BASE/tmp/ollama_setup_serve.log" >&2 || true
    exit 1
  fi
  OLLAMA_HOST=http://127.0.0.1:11436 "$BASE/ollama_bin/ollama" pull "$OLLAMA_MODEL"
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
else
  echo "SKIP_PULL=1 — not pulling model"
fi

echo ""
echo "Done."
echo ""
echo "1) Start the proxy on the login node (uses GPU only during each request):"
echo "     export OLLAMA_BASE=$BASE"
echo "     python3 $BASE/scripts/ollama_proxy.py"
echo ""
echo "2) In another terminal, run the UI and point it at the proxy (default port 11434):"
echo "     cd $SCRIPT_DIR/app && npm run dev -- --host 0.0.0.0"
echo "     # optional: echo 'VITE_OLLAMA_URL=http://127.0.0.1:11434/api/chat' >> app/.env.local"
echo ""
echo "3) Open the printed dev URL in your browser and use AI Generate Edit mode."
echo ""
echo "Forward mode (no Slurm — you run ollama yourself on a port):"
echo "     SQ_OLLAMA_FORWARD=http://127.0.0.1:11436 python3 $BASE/scripts/ollama_proxy.py"
echo ""
echo "Slurm overrides (if your account/partition differ):"
echo "     SQ_SLURM_PARTITION=interactive SQ_SLURM_ACCOUNT=3dv python3 $BASE/scripts/ollama_proxy.py"
echo ""
