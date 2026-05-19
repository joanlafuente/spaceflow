#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=spaceflow-partfield
#SBATCH --output=/work/scratch/%u/spaceflow/ab_runs/%x_%j.out
#SBATCH --error=/work/scratch/%u/spaceflow/ab_runs/%x_%j.err

set -euo pipefail

GUIDEFLOW3D_ENV="${GUIDEFLOW3D_ENV:-/work/courses/3dv/team3/guideflow3d/envs/guideflow3d}"
if [[ ! -x "$GUIDEFLOW3D_ENV/bin/python" ]]; then
  GUIDEFLOW3D_ENV="/work/courses/3dv/team3/spaceflow/envs/guideflow3d"
fi
export PATH="$GUIDEFLOW3D_ENV/bin:$PATH"
PYTHON_BIN="$GUIDEFLOW3D_ENV/bin/python"

module load cuda/12.8
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/guideflow3d/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPACEFLOW_SCRATCH="${SPACEFLOW_SCRATCH:-/work/scratch/${SLURM_JOB_USER:-$USER}}"
export HF_HOME="${SPACEFLOW_SCRATCH}/spaceflow/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${SPACEFLOW_SCRATCH}/spaceflow/torch"
export XDG_CACHE_HOME="${SPACEFLOW_SCRATCH}/spaceflow/xdg_cache"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "${SPACEFLOW_SCRATCH}/spaceflow/ab_runs"

cd /work/courses/3dv/team3/spaceflow

git checkout main || true
git show main:lib/opt/appearance.py > lib/opt/appearance.py
rm -rf lib/superdec

srun --ntasks=1 --export=ALL "$PYTHON_BIN" run.py \
  --output_dir "${SPACEFLOW_SCRATCH}/spaceflow/ab_runs/lamp_to_chair/partfield" \
  --text_prompt "A wooden chair" \
  --appearance_mesh examples/lamp.glb \
  --shape_superquadric_path examples/superquadrics/chair_sq.npz \
  --shape_tau 6.0 \
  --guidance_mode appearance \
  --convert_yup_to_zup