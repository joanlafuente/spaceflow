#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C 5060ti
#SBATCH --time=00:15:00
#SBATCH --job-name=spaceflow-test
#SBATCH --output=/work/courses/3dv/team3/spaceflow/outputs/test_%j.out
#SBATCH --error=/work/courses/3dv/team3/spaceflow/outputs/test_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d
module load cuda/12.8
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/guideflow3d/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


RUN_NAME="local_control_cheese_chair_test_0_375_polyak_binarized_mask"
CONTROL_SUPERQUADRIC_PATH="examples/superquadrics/cheese_chair/cheese_chair.npz"
LOW_CONTROL_SUPERQUADRIC_MASK_PATH="examples/superquadrics/cheese_chair/bb_cheesebackrest.npz"
HIGH_CONTROL_SUPERQUADRIC_PATH="examples/superquadrics/chair_w_small_backrest_high.npz"
APPEARANCE_IMAGE="examples/table.jpg"
TEXT_PROMPT="A chair"


cd /work/courses/3dv/team3/spaceflow

srun --ntasks=1 --export=ALL \
  /work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python run_local_tau.py \
  --guidance_mode similarity \
  --appearance_image "$APPEARANCE_IMAGE" \
  --output_dir "outputs/$RUN_NAME/high_control_10_low_control_3_low_control_mask" \
  --shape_superquadric_path "$CONTROL_SUPERQUADRIC_PATH" \
  --shape_tau 3.0 \
  --convert_yup_to_zup \
  --low_control_superquadric_mask_path "$LOW_CONTROL_SUPERQUADRIC_MASK_PATH" \
  --shape_superquadric_high_control_path "$HIGH_CONTROL_SUPERQUADRIC_PATH" \
  --shape_tau_high_control 10.0 \
  --text_prompt "$TEXT_PROMPT" \
  --polyak_update_tau 0.375 \
  --local_tau_mode low_control_mask