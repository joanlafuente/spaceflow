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


RUN_NAME="local_control_table_2_step_gen_new_sq"
CONTROL_SUPERQUADRIC_PATH="examples/superquadrics/table_local_control_2_step_gen/table_with_plant.npz"
LOW_CONTROL_SUPERQUADRIC_MASK_PATH="examples/superquadrics/table_local_control_2_step_gen/table_with_plant_table_bb.npz"
HIGH_CONTROL_SUPERQUADRIC_PATH="examples/superquadrics/table_local_control_2_step_gen/plant_high_control.npz"
APPEARANCE_IMAGE="examples/table.jpg"
TEXT_PROMPT="A table with a plant on top"


cd /work/courses/3dv/team3/spaceflow

srun --ntasks=1 --export=ALL \
  /work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python run_local_tau.py \
  --guidance_mode similarity \
  --appearance_image "$APPEARANCE_IMAGE" \
  --output_dir "outputs/$RUN_NAME/high_control_10_low_control_3_low_control_mask_0_25_high_control_latents_pulling_inv" \
  --shape_superquadric_path "$CONTROL_SUPERQUADRIC_PATH" \
  --shape_tau 3.0 \
  --convert_yup_to_zup \
  --low_control_superquadric_mask_path "$LOW_CONTROL_SUPERQUADRIC_MASK_PATH" \
  --shape_superquadric_high_control_path "$HIGH_CONTROL_SUPERQUADRIC_PATH" \
  --shape_tau_high_control 10.0 \
  --text_prompt "$TEXT_PROMPT" \
  --polyak_update_tau 0.25 \
  --local_tau_mode low_control_mask