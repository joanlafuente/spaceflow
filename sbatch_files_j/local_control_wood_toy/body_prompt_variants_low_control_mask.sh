#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C 5060ti
#SBATCH --time=00:15:00
#SBATCH --job-name=wood-toy-body
#SBATCH --array=0-2
#SBATCH --output=/work/courses/3dv/team3/spaceflow/outputs/local_control_wood_toy_prompt_variants/logs/%x_%A_%a.out
#SBATCH --error=/work/courses/3dv/team3/spaceflow/outputs/local_control_wood_toy_prompt_variants/logs/%x_%A_%a.err

set -euo pipefail

PYTHON="/work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python"
module load cuda/12.8
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/guideflow3d/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/work/courses/3dv/team3/spaceflow_cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/work/courses/3dv/team3/spaceflow_cache/torch"
export XDG_CACHE_HOME="/work/courses/3dv/team3/spaceflow_cache/xdg"

RUN_NAME="local_control_wood_toy_prompt_variants"
CONTROL_SUPERQUADRIC_PATH="examples/superquadrics/wood_person_toy/wood_person.npz"
LOW_CONTROL_SUPERQUADRIC_MASK_PATH="examples/superquadrics/wood_person_toy/low_control_area.npz"
HIGH_CONTROL_SUPERQUADRIC_PATH="examples/superquadrics/wood_person_toy/wood_person_head.npz"
APPEARANCE_IMAGE="examples/table.jpg"

VARIANT_SLUGS=(
  "arms_wide"
  "legs_spread"
  "punch_forward"
)

TEXT_PROMPTS=(
  "A wooden toy person with both arms stretched wide"
  "A wooden toy person standing with legs spread apart and arms down"
  "A wooden toy person throwing a punch with one arm extended forward"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT_SLUGS[$IDX]}"
TEXT_PROMPT="${TEXT_PROMPTS[$IDX]}"
OUTPUT_DIR="outputs/${RUN_NAME}/${VARIANT}/high_control_10_low_control_3_low_control_mask_0_18_high_control_latents_pulling_dilation_3"

cd /work/courses/3dv/team3/spaceflow

echo "Running ${VARIANT}: ${TEXT_PROMPT}"
echo "Output directory: ${OUTPUT_DIR}"

srun --ntasks=1 --export=ALL \
  "$PYTHON" run_local_tau.py \
  --guidance_mode similarity \
  --appearance_image "$APPEARANCE_IMAGE" \
  --output_dir "$OUTPUT_DIR" \
  --shape_superquadric_path "$CONTROL_SUPERQUADRIC_PATH" \
  --shape_tau 3.0 \
  --convert_yup_to_zup \
  --low_control_superquadric_mask_path "$LOW_CONTROL_SUPERQUADRIC_MASK_PATH" \
  --shape_superquadric_high_control_path "$HIGH_CONTROL_SUPERQUADRIC_PATH" \
  --shape_tau_high_control 10.0 \
  --polyak_update_tau 0.18 \
  --text_prompt "$TEXT_PROMPT" \
  --local_tau_mode low_control_mask
