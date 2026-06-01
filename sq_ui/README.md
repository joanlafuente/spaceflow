# Superquadric UI

Web UI for editing superquadrics and running the fitting services used by this Spaceflow checkout.

Run each service in a separate terminal or `tmux` pane on the same host where you run Vite. In dev, Vite proxies browser calls to:

- SuperDec: `http://127.0.0.1:11435`
- SuperFlex: `http://127.0.0.1:11436`
- TRELLIS/Create: `http://127.0.0.1:11437`
- SpaceFlow runs/assets: `http://127.0.0.1:11438`
- Ollama/Edit: `http://127.0.0.1:11434`

## 1. SuperDec Service

```bash
cd /work/courses/3dv/team3/spaceflow

export SUPERDEC_BASE=/work/courses/3dv/team3/spaceflow/superdec_ui
export SQ_SUPERDEC_CHECKPOINT_DIR="$SUPERDEC_BASE/weights/normalized"

python3 "$SUPERDEC_BASE/scripts/superdec_service.py"
```

Health check:

```bash
curl -s http://127.0.0.1:11435/superdec/health | head
```

## 2. SuperFlex Service

This checkout uses the local SuperFlex weights from the shared course tree. Create the `ckpt.pt` name expected by the runner in that same weights directory:

```bash
cd /work/courses/3dv/team3/spaceflow

ln -sf epoch_1000.pt /work/courses/3dv/team3/spaceflow/superflex/weights/ckpt.pt
```

Then run:

```bash
cd /work/courses/3dv/team3/spaceflow

export SUPERFLEX_BASE=/work/courses/3dv/team3/spaceflow/superflex_ui
export SQ_SUPERFLEX_PYTHON="$(pwd)/envs/guideflow3d/bin/python"
export SQ_SUPERFLEX_CHECKPOINT_DIR=/work/courses/3dv/team3/spaceflow/superflex/weights
export SQ_SUPERFLEX_WORK_DIR=/work/courses/3dv/team3/spaceflow/superflex

python3 sq_ui/scripts/superflex_service.py
```

Health check:

```bash
curl -s http://127.0.0.1:11436/superflex/health | head
```

## 3. TRELLIS Service For Create

```bash
cd /work/courses/3dv/team3/spaceflow

export SQ_TRELLIS_REPO_ROOT="$(pwd)"
export SQ_TRELLIS_SLURM_GPUS=1
export SQ_TRELLIS_PYTHON="$(pwd)/envs/guideflow3d/bin/python"
export SQ_TRELLIS_SCRATCH=/work/courses/3dv/team3/spaceflow_runtime/trellis_ui

python3 sq_ui/scripts/trellis_service.py
```

Health check:

```bash
curl -s http://127.0.0.1:11437/trellis/health | head
```

## 4. SpaceFlow Service For Save/Run

The SpaceFlow button saves the editor scene as three cluster-side NPZ files and can launch
`run_local_tau.py` with the current high/low control split.

```bash
cd /work/courses/3dv/team3/spaceflow

export SQ_SPACEFLOW_PYTHON="$(pwd)/envs/guideflow3d/bin/python"
export SQ_SPACEFLOW_STORAGE_ROOT=/work/courses/3dv/team3/spaceflow_runtime
export SQ_SPACEFLOW_ASSET_ROOT=$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_assets
export SQ_SPACEFLOW_RUN_ROOT=$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_runs
export SQ_SPACEFLOW_CACHE_ROOT=$SQ_SPACEFLOW_STORAGE_ROOT/huggingface
export SQ_SPACEFLOW_XDG_CACHE_ROOT=$SQ_SPACEFLOW_STORAGE_ROOT/xdg_cache
export SQ_SPACEFLOW_SLURM_CONSTRAINT=5060ti
export SQ_SPACEFLOW_SLURM_EXCLUDE=studgpu-node09

python3 sq_ui/scripts/spaceflow_service.py
```

Health check:

```bash
curl -s http://127.0.0.1:11438/spaceflow/health | head
```

The service uses Slurm by default when launched on a login node and constrains runs to `5060ti` by default because FlashAttention fails on the older `2080ti` nodes. It also excludes `studgpu-node09` by default because that node currently reports an NVML driver/library mismatch and makes PyTorch fail CUDA initialization; set `SQ_SPACEFLOW_SLURM_EXCLUDE=` once the node is fixed. To run from an allocated GPU node without wrapping requests in `srun`, set:

```bash
export SQ_SPACEFLOW_FORCE_LOCAL=1
```

Runs are saved under `$SQ_SPACEFLOW_RUN_ROOT/<run_id>/output`. The UI launch path now passes `--full_pipeline`, so it first writes the structure-stage files (`sample.glb`, `struct_mesh_zup.glb`, control meshes, `struct_renders/000.png`, `voxels/struct_voxels.ply`) and then continues into PartField plus similarity/appearance refinement. The refined mesh is `out_sim.glb` for the current similarity-guided UI flow. Set `SQ_SPACEFLOW_FULL_PIPELINE=0` before starting the service if you need the old structure-only behavior.

## 5. Ollama Proxy For Edit

```bash
export OLLAMA_BASE=/work/courses/3dv/team3/spaceflow_runtime/superquadric_ui
python3 "$OLLAMA_BASE/scripts/ollama_proxy.py"
```

Smoke test:

```bash
curl -s http://127.0.0.1:11434/ | head
```

## 6. Run The Frontend

```bash
cd /work/courses/3dv/team3/spaceflow/sq_ui/app
npm run dev -- --host 0.0.0.0
```

Open the printed Vite URL, usually `http://<host>:5173`.

## 7. Open NPZ Files Directly

With the Vite dev server running, open an editor URL with an `npz` query parameter:

```text
http://<host>:5173/?npz=/work/courses/3dv/team3/spaceflow/datasets/abo/assets/B07B4Y45N1/superflex.npz
```

You can also paste a repo-relative NPZ URL on the Vite host:

```text
http://<host>:5173/datasets/abo/assets/B07B4W81YX/superflex.npz
```

Vite redirects that to the editor and serves the local `.npz` through `/_sq/npz`, so the browser does not download it first. By default URL-opened files load as stored, which matches the Z-up ABO/ShapeNet asset bundles. Add `&basis=zup` only for older files that need Z-up-to-Y-up conversion.

To build the URL for the currently running Vite port:

```bash
python3 sq_ui/scripts/open_npz_in_editor.py /work/courses/3dv/team3/spaceflow/datasets/abo/assets/B07B4W5V3C/superflex.npz
```

## Notes

- For only manual editing/import/export, the frontend alone is enough.
- For the SuperDec button, run SuperDec.
- For the SuperFlex button, run SuperFlex.
- For the SpaceFlow button, run SpaceFlow.
- For Create, run TRELLIS and SuperDec.
- For Edit, run Ollama proxy.
- Leave `sq_ui/app/.env.local` empty in dev unless you know you need an override; same-origin Vite proxying avoids browser `localhost` problems.
- The service wrappers use Slurm by default when launched on a login node. To run on a GPU node without wrapping each request in `srun`, set `SQ_SUPERDEC_FORCE_LOCAL=1`, `SQ_SUPERFLEX_FORCE_LOCAL=1`, `SQ_TRELLIS_FORCE_LOCAL=1`, or `SQ_SPACEFLOW_FORCE_LOCAL=1` as appropriate.
