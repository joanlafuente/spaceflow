#!/usr/bin/env bash
#
# One-time setup: clone SuperDec, create a venv, install deps, and download checkpoints
# under the shared Spaceflow checkout. After this, start the SuperDec service and point the UI at it.
#
# Usage:
#   cd /path/to/spaceflow
#   bash sq_ui/setup_superdec.sh
#
# Optional environment:
#   SQ_SUPERDEC_BASE      — install root (default <spaceflow>/superdec_ui)
#   SQ_SUPERDEC_SCRATCH   — legacy alias for SQ_SUPERDEC_BASE
#   SUPERDEC_REPO_URL     — override repo URL
#   SUPERDEC_REPO_REF     — branch/tag/commit to checkout
#   SKIP_CLONE            — if 1, reuse existing repo
#   SKIP_PIP              — if 1, skip pip install
#   SKIP_CHECKPOINTS      — if 1, skip checkpoint download
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACEFLOW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE="${SQ_SUPERDEC_BASE:-${SQ_SUPERDEC_SCRATCH:-$SPACEFLOW_ROOT/superdec_ui}}"
REPO_DIR="$BASE/repo"
VENV_DIR="$BASE/venv"
WEIGHTS_DIR="$BASE/weights"
SUPERDEC_REPO_URL="${SUPERDEC_REPO_URL:-https://github.com/elisabettafedele/superdec.git}"
SUPERDEC_REPO_REF="${SUPERDEC_REPO_REF:-main}"

echo "Install root: $BASE"
mkdir -p "$BASE"/{scripts,logs,runs,tmp,weights}
# Cluster: avoid filling $HOME (~/.cache/pip) or small /tmp during installs.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASE/tmp/pip-cache}"
export TMPDIR="${TMPDIR:-$BASE/tmp}"
mkdir -p "$PIP_CACHE_DIR"

if [[ "${SKIP_CLONE:-0}" != "1" ]]; then
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Cloning SuperDec ..."
    git clone "$SUPERDEC_REPO_URL" "$REPO_DIR"
  else
    echo "Reusing existing SuperDec repo at $REPO_DIR"
    if ! git -C "$REPO_DIR" fetch --all --tags; then
      echo "Warning: git fetch failed, continuing with existing local clone" >&2
    fi
  fi
  git -C "$REPO_DIR" checkout "$SUPERDEC_REPO_REF"
else
  echo "SKIP_CLONE=1 — reusing existing repo"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment ..."
  python3 -m venv "$VENV_DIR"
fi

if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  echo "Installing Python dependencies ..."
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"
  "$VENV_DIR/bin/pip" install -e "$REPO_DIR"
else
  echo "SKIP_PIP=1 — not installing Python packages"
fi

if [[ "${SKIP_CHECKPOINTS:-0}" != "1" ]]; then
  echo "Downloading checkpoints ..."
  mkdir -p "$WEIGHTS_DIR"/{shapenet,normalized}
  "$VENV_DIR/bin/pip" install gdown >/dev/null
  "$VENV_DIR/bin/python" -m gdown 1Nsgtm_nCyp6qbRgnenoJVqL88eS1GXmC -O "$WEIGHTS_DIR/shapenet/config.yaml"
  "$VENV_DIR/bin/python" -m gdown 1ypCViehSOzkCFL6dcCDfdPRzuj_MIayz -O "$WEIGHTS_DIR/shapenet/ckpt.pt"
  "$VENV_DIR/bin/python" -m gdown 1l0wpNssH7f3V61SUA4VcjrVy-ganmIp_ -O "$WEIGHTS_DIR/normalized/config.yaml"
  "$VENV_DIR/bin/python" -m gdown 1r1ydYXkMf7q6U99ze78-zkLiKnO3ICGk -O "$WEIGHTS_DIR/normalized/ckpt.pt"
else
  echo "SKIP_CHECKPOINTS=1 — reusing existing checkpoints"
fi

echo "Installing service scripts ..."
mkdir -p "$BASE/scripts"
cp "$SCRIPT_DIR/scripts/superdec_infer.py" "$BASE/scripts/superdec_infer.py"
sed "s|__SUPERDEC_BASE__|$BASE|g" "$SCRIPT_DIR/scripts/superdec_service.py" > "$BASE/scripts/superdec_service.py"
chmod +x "$BASE/scripts/superdec_infer.py" "$BASE/scripts/superdec_service.py"

echo ""
echo "Done."
echo ""
echo "1) Start the SuperDec service:"
echo "     export SUPERDEC_BASE=$BASE"
echo "     export SQ_SUPERDEC_CHECKPOINT_DIR=$WEIGHTS_DIR/normalized"
echo "     python3 $BASE/scripts/superdec_service.py"
echo ""
echo "2) Run the UI and point it at the service:"
echo "     cd $SCRIPT_DIR/app && npm run dev -- --host 0.0.0.0"
echo "     # optional: echo 'VITE_SUPERDEC_URL=http://127.0.0.1:11435' >> app/.env.local"
echo ""
echo "3) For text-to-pointcloud Create, also start the TRELLIS service in a compatible env:"
echo "     export SQ_TRELLIS_REPO_ROOT=$SPACEFLOW_ROOT"
echo "     export SQ_TRELLIS_SLURM_GPUS=1"
echo "     export SQ_TRELLIS_PYTHON=$SPACEFLOW_ROOT/envs/guideflow3d/bin/python"
echo "     export SQ_TRELLIS_SCRATCH=$SPACEFLOW_ROOT/spaceflow_runtime/trellis_ui"
echo "     export SQ_TRELLIS_FORCE_LOCAL=1   # only if already on a GPU node"
echo "     python3 $SCRIPT_DIR/scripts/trellis_service.py"
echo "     # optional: echo 'VITE_TRELLIS_URL=http://127.0.0.1:11437' >> app/.env.local"
echo ""
echo "Install layout:"
echo "     weights: $WEIGHTS_DIR"
echo "     runs:    $BASE/runs"
echo "     logs:    $BASE/logs"
echo ""
