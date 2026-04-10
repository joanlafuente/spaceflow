#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --job-name=spaceflow-test
#SBATCH --output=/work/courses/3dv/team3/spaceflow/outputs_f/test_%j.out
#SBATCH --error=/work/courses/3dv/team3/spaceflow/outputs_f/test_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d
module load cuda/12.8
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/guideflow3d/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPACEFLOW_SCRATCH="${SPACEFLOW_SCRATCH:-/work/scratch/fscharitzer}"
export HF_HOME="${SPACEFLOW_SCRATCH}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${SPACEFLOW_SCRATCH}/torch"
export XDG_CACHE_HOME="${SPACEFLOW_SCRATCH}/xdg_cache"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "${TORCH_HOME}/hub" "${XDG_CACHE_HOME}"

cd /work/courses/3dv/team3/spaceflow
srun --ntasks=1 --export=ALL \
  /work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python run.py \
  --guidance_mode similarity \
  --appearance_image examples/ford_shelby.jpg \
  --output_dir outputs_f/test1 \
  --convert_yup_to_zup \
  --shape_superquadric_path examples/superquadrics/car_sq.npz \
  --shape_tau 6.0 \
  --text_prompt "A supercar" \
