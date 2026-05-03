#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=spaceflow-ab
#SBATCH --output=/work/scratch/%u/spaceflow/ab_runs/%x_%j.out
#SBATCH --error=/work/scratch/%u/spaceflow/ab_runs/%x_%j.err
#
# Slurm wrapper around scripts/ab_compare_superdec_vs_partfield.sh.
# Mirrors the cuda/Blender/HF-cache setup used by test_run_*.sh and submits
# the A/B comparison so it can run unattended on a GPU node.
#
# Usage (default chair-from-lamp test):
#   sbatch scripts/sbatch_ab_compare.sh
#
# Override the test fixture from the command line:
#   sbatch scripts/sbatch_ab_compare.sh \
#     --text_prompt "A propeller airplane" \
#     --appearance_mesh examples/lamp.glb \
#     --shape_superquadric examples/superquadrics/plane_sq.npz \
#     --shape_tau 6.0 \
#     --convert_yup_to_zup \
#     --out_root /work/scratch/$USER/spaceflow/ab_runs/lamp_to_plane
#
# After completion, look at:
#   $OUT_ROOT/superdec/out_app.glb           (SUPERDEC arm)
#   $OUT_ROOT/superdec/superdec/             (segments + correspondence PLYs + summary.json)
#   $OUT_ROOT/partfield/out_app.glb          (PartField arm)
#

set -euo pipefail

# --- Cluster env, identical to test_run_test.sh ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d
module load cuda/12.8
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/guideflow3d/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Per-user scratch redirection (matches test_run_s.sh) ---
SPACEFLOW_SCRATCH="${SPACEFLOW_SCRATCH:-/work/scratch/${SLURM_JOB_USER:-$USER}}"
export HF_HOME="${SPACEFLOW_SCRATCH}/spaceflow/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${SPACEFLOW_SCRATCH}/spaceflow/torch"
export XDG_CACHE_HOME="${SPACEFLOW_SCRATCH}/spaceflow/xdg_cache"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "${TORCH_HOME}/hub" "${XDG_CACHE_HOME}" \
         "${SPACEFLOW_SCRATCH}/spaceflow/ab_runs"

# --- Default fixture: lamp → wooden chair (cheap, both branches viable) ---
TEXT_PROMPT_DEFAULT="A wooden chair"
APP_PATH_DEFAULT="examples/lamp.glb"
SHAPE_SQ_DEFAULT="examples/superquadrics/chair_sq.npz"
SHAPE_TAU_DEFAULT="6.0"
OUT_ROOT_DEFAULT="${SPACEFLOW_SCRATCH}/spaceflow/ab_runs/lamp_to_chair"

if [[ $# -eq 0 ]]; then
  AB_ARGS=(
    --text_prompt "$TEXT_PROMPT_DEFAULT"
    --appearance_mesh "$APP_PATH_DEFAULT"
    --shape_superquadric "$SHAPE_SQ_DEFAULT"
    --shape_tau "$SHAPE_TAU_DEFAULT"
    --convert_yup_to_zup
    --out_root "$OUT_ROOT_DEFAULT"
  )
else
  AB_ARGS=("$@")
fi

cd /work/courses/3dv/team3/spaceflow

# --- Make sure we are on the SUPERDEC branch before launching ---
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "feature/superdec-correspondence" ]]; then
  echo "[sbatch_ab] not on feature/superdec-correspondence (got $CURRENT_BRANCH)." >&2
  echo "[sbatch_ab] git checkout feature/superdec-correspondence before submitting." >&2
  exit 4
fi

srun --ntasks=1 --export=ALL bash scripts/ab_compare_superdec_vs_partfield.sh "${AB_ARGS[@]}"
