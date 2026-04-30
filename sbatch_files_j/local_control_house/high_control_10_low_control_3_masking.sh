#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=spaceflow-test
#SBATCH --output=/work/courses/3dv/team3/spaceflow/outputs/test_%j.out
#SBATCH --error=/work/courses/3dv/team3/spaceflow/outputs/test_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d
module load cuda/12.8
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/guideflow3d/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


cd /work/courses/3dv/team3/spaceflow
srun --ntasks=1 --export=ALL \
  /work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python run_local_tau.py \
  --guidance_mode similarity \
  --appearance_image examples/house.jpg \
  --output_dir outputs/local_control_house/high_control_10_low_control_3_masking \
  --shape_superquadric_path examples/superquadrics/house_local_control/house_high_control.npz \
  --shape_tau 3.0 \
  --convert_yup_to_zup \
  --shape_superquadric_high_control_path examples/superquadrics/house_local_control/house_low_control.npz \
  --shape_tau_high_control 10.0 \
  --text_prompt "House" \
  --local_tau_mode masking