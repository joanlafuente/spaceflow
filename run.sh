#!/usr/bin/bash

cd /work/courses/3dv/team3/spaceflow-minimal
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/work/courses/3dv/team3/spaceflow-minimal/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SQ_SPACEFLOW_PYTHON="/work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python"
export SQ_SPACEFLOW_STORAGE_ROOT=/work/courses/3dv/team3/spaceflow_runtime
export SQ_SPACEFLOW_ASSET_ROOT=$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_assets
export SQ_SPACEFLOW_RUN_ROOT=$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_runs

export SQ_SPACEFLOW_PORT=11480
export VITE_DEV_PROXY_SPACEFLOW=http://127.0.0.1:11480

/work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python sq_ui/scripts/spaceflow_service.py &
SPACEFLOW_PID=$!

cleanup() {
    kill "$SPACEFLOW_PID" 2>/dev/null || true
    wait "$SPACEFLOW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd /work/courses/3dv/team3/spaceflow-minimal/sq_ui/app
npm run dev -- --host 0.0.0.0
