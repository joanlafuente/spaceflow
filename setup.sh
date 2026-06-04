#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SPACEFLOW_ENV_NAME:-spaceflow}"

if command -v conda >/dev/null 2>&1; then
  conda create -n "$ENV_NAME" python=3.10 -y
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "conda was not found. Create a Python 3.10 environment first, then rerun this script." >&2
  exit 1
fi

conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install \
  pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
  trimesh open3d xatlas pyvista pymeshfix igraph transformers omegaconf lightning==2.2 h5py yacs \
  scikit-image loguru boto3 mesh2sdf tetgen==0.6.4 pymeshlab plyfile einops libigl polyscope \
  potpourri3d simple_parsing arrgh vtk matplotlib numpy==1.26.4

pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install -U 'python-pycg[all]'
pip install xformers==0.0.28.post2 --no-deps --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install spconv-cu120
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

EXT_ROOT="${SPACEFLOW_EXTENSION_BUILD_ROOT:-/tmp/spaceflow_extensions}"
mkdir -p "$EXT_ROOT"

if [ "${SPACEFLOW_SKIP_SOURCE_EXTENSIONS:-0}" != "1" ]; then
  if [ ! -d "$EXT_ROOT/nvdiffrast" ]; then
    git clone https://github.com/NVlabs/nvdiffrast.git "$EXT_ROOT/nvdiffrast"
  fi
  pip install "$EXT_ROOT/nvdiffrast" --no-build-isolation --no-cache-dir

  if [ ! -d "$EXT_ROOT/diffoctreerast" ]; then
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git "$EXT_ROOT/diffoctreerast"
  fi
  pip install "$EXT_ROOT/diffoctreerast" --no-build-isolation --no-cache-dir

  if [ ! -d "$EXT_ROOT/mip-splatting" ]; then
    git clone https://github.com/autonomousvision/mip-splatting.git "$EXT_ROOT/mip-splatting"
  fi
  pip install "$EXT_ROOT/mip-splatting/submodules/diff-gaussian-rasterization" --no-build-isolation --no-cache-dir
fi

pip install third_party/TRELLIS/extensions/vox2seq --no-build-isolation --no-cache-dir

echo "Python environment ready: $ENV_NAME"
echo "Place PartField checkpoint at third_party/PartField/models/model_objaverse.ckpt before full SpaceFlow runs."
echo "Install UI dependencies with: cd sq_ui/app && npm install"
