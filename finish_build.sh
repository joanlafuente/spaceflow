#!/bin/bash
#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=guideflow-build
#SBATCH --output=/work/courses/3dv/team3/guideflow3d/outputs/build_%j.out
#SBATCH --error=/work/courses/3dv/team3/guideflow3d/outputs/build_%j.err

set -e  # exit on any error

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d
module load cuda/12.8

export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export CUMM_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export MAX_JOBS=2

echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

echo "=== Installing spconv (pre-built, same as SpaceControl) ==="
pip install cumm-cu120==0.4.11 spconv-cu120==2.3.6 --no-cache-dir

echo "=== Building vox2seq ==="
pip install third_party/TRELLIS/extensions/vox2seq \
  --no-build-isolation --no-cache-dir

echo "=== Verifying all imports ==="
python -c "
import torch; print('torch:', torch.__version__)
import flash_attn; print('flash_attn OK')
import nvdiffrast; print('nvdiffrast OK')
import diffoctreerast; print('diffoctreerast OK')
import kaolin; print('kaolin OK')
import spconv; print('spconv OK')
import vox2seq; print('vox2seq OK')
print('ALL OK')
"
