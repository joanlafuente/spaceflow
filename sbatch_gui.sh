#!/usr/bin/bash

#SBATCH -A 3dv
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH -C 5060ti
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --job-name=sf-gui
#SBATCH --output=/home/msayfiddinov/spaceflow/outputs/%x-%j.out
#SBATCH --error=/home/msayfiddinov/spaceflow/outputs/%x-%j.err

cd /home/msayfiddinov/spaceflow
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/3dv/team3/guideflow3d/envs/guideflow3d

module purge
module load cuda/12.8

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;9.0;12.0"
export BLENDER_HOME="/home/msayfiddinov/spaceflow/blender-3.0.1-linux-x64/blender"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure passwordless SSH back to login node
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
fi
if ! grep -qF "$(cat ~/.ssh/id_ed25519.pub)" ~/.ssh/authorized_keys 2>/dev/null; then
    cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

TARGET_HOST="${SLURM_SUBMIT_HOST:-$(hostname -f)}"
TUNNEL_PORT=18080
echo "Starting reverse tunnel: ${TARGET_HOST}:${TUNNEL_PORT} -> localhost:8080 (compute)"
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R ${TUNNEL_PORT}:localhost:8080 "$USER@$TARGET_HOST" &
TUNNEL_PID=$!
sleep 3

if ! kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "ERROR: tunnel failed to start"
    exit 1
fi
echo "Tunnel running (pid $TUNNEL_PID)"
echo "Open http://localhost:${TUNNEL_PORT} on the login node (forward it via VS Code Ports panel)"

/work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python gui/gui_text_image.py

kill $TUNNEL_PID 2>/dev/null
