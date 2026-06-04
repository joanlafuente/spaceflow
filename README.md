# SpaceFlow Minimal

This branch is the delivery-oriented version of the SpaceFlow repository. It keeps only the code needed for the Superquadric UI, the SpaceFlow service wrappers, the main SpaceFlow pipeline entrypoints, and the vendored runtime source used by that pipeline.

## What Is Included

- `sq_ui/app`: React/Vite superquadric editor.
- `sq_ui/scripts`: HTTP service wrappers for SpaceFlow, TRELLIS create, SuperDec, SuperFlex, experiments, and NPZ helpers.
- `run_local_tau.py`: current SpaceFlow pipeline entrypoint used by the UI service.
- `lib`: guidance, rendering, voxel, and PartField helper code.
- `third_party/TRELLIS`: source needed by the TRELLIS runtime, including the source-only `trellis/models` package.
- `third_party/PartField`: source needed for PartField feature extraction.
- `gui/pipeline.json`: local TRELLIS pipeline config used by `config/default.yaml`.

## What Is Not Included

Generated outputs, sbatch experiment collections, local datasets, logs, node builds, Python caches, downloaded model weights, and large local checkpoints are intentionally excluded.

You still need runtime weights/caches:

- TRELLIS checkpoints, either downloaded by Hugging Face or available in `HF_HOME`.
- PartField checkpoint at `third_party/PartField/models/model_objaverse.ckpt`.
- Optional SuperDec/SuperFlex checkpoints if you use those UI buttons.

## Setup

Use an existing compatible Python environment, or run:

```bash
bash setup.sh
```

Then install the frontend dependencies:

```bash
cd sq_ui/app
npm install
npm run build
```

## Run The UI

Start the services you need in separate terminals:

```bash
python sq_ui/scripts/spaceflow_service.py
python sq_ui/scripts/trellis_service.py
python sq_ui/scripts/superdec_service.py
python sq_ui/scripts/superflex_service.py
```

Then run Vite:

```bash
cd sq_ui/app
npm run dev -- --host 0.0.0.0
```

The SpaceFlow service defaults to repo-local runtime folders under `spaceflow_runtime/`. Override with `SQ_SPACEFLOW_STORAGE_ROOT`, `SQ_SPACEFLOW_ASSET_ROOT`, or `SQ_SPACEFLOW_RUN_ROOT` for cluster/shared storage.

The renderer downloads Blender under the repo by default if `SPACEFLOW_BLENDER_PATH` is not set. Set `SPACEFLOW_BLENDER_PATH=/path/to/blender` to use an existing Blender install.

## Pipeline CLI

```bash
python run_local_tau.py --help
```

`config/default.yaml` points to the repo-local `gui/pipeline.json` by default. Override with `SPACEFLOW_TRELLIS_PIPELINE_PATH` or `--trellis_pipeline_path` if your model config lives elsewhere.
