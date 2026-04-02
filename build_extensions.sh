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
echo "=== PyTorch: $(python -c 'import torch; print(torch.__version__)') ==="

cd /work/courses/3dv/team3/guideflow3d

echo "=== Building nvdiffrast ==="
git clone https://github.com/NVlabs/nvdiffrast.git ~/nvdiffrast
pip install ~/nvdiffrast --no-build-isolation --no-cache-dir

echo "=== Building diffoctreerast ==="
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ~/diffoctreerast
pip install ~/diffoctreerast --no-build-isolation --no-cache-dir

echo "=== Building mip-splatting ==="
git clone https://github.com/autonomousvision/mip-splatting.git ~/mip-splatting
pip install ~/mip-splatting/submodules/diff-gaussian-rasterization/ \
  --no-build-isolation --no-cache-dir

echo "=== Building flash-attn ==="
pip install flash-attn --no-build-isolation --no-cache-dir

echo "=== Building cumm from source (required before spconv) ==="
git clone https://github.com/FindDefinition/cumm.git ~/cumm --recursive
cd ~/cumm
pip install pccm --no-cache-dir
pip install -e . --no-build-isolation --no-cache-dir
cd /work/courses/3dv/team3/guideflow3d

echo "=== Building spconv from source ==="
git clone https://github.com/traveller59/spconv.git ~/spconv --recursive
cd ~/spconv
# Remove cumm from requirements since we installed it editably above
python -c "
import re
with open('pyproject.toml', 'r') as f:
    content = f.read()
# Remove cumm from dependencies list only
content = re.sub(r'\s*\"cumm[^\"]*\",?\n', '\n', content)
with open('pyproject.toml', 'w') as f:
    f.write(content)
print('patched pyproject.toml')
"

pip install -e . --no-build-isolation --no-cache-dir
cd /work/courses/3dv/team3/guideflow3d

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
