# SpaceFlow Minimal

This branch is the delivery-oriented version of the SpaceFlow repository. It keeps the superquadric UI, the SpaceFlow service, the SpaceFlow pipeline entrypoints, and the vendored runtime source needed by that pipeline.

## Included

- `sq_ui/app`: React/Vite SpaceFlow superquadric editor.
- `sq_ui/scripts/spaceflow_service.py`: HTTP service used by the UI to save inputs and launch runs.
- `sq_ui/scripts/run_spaceflow_experiment.py`: SpaceFlow variant runner used by the experiment button.
- `sq_ui/scripts/render_spaceflow_experiment_comparison.py`: helper for experiment comparison renders.
- `run_local_tau.py`: main SpaceFlow pipeline entrypoint used by the service.
- `lib`, `third_party`, `gui`, and `config`: runtime source/configuration required by the pipeline.

## Excluded

Generated outputs, sbatch experiment collections, local datasets, logs, node builds, Python caches, downloaded model weights, and large local checkpoints are intentionally excluded.

You still need the runtime checkpoints/caches expected by the pipeline, including the PartField checkpoint at `third_party/PartField/models/model_objaverse.ckpt`.

## Setup

Use an existing compatible Python environment, or run:

```bash
bash setup.sh
```

Install the frontend dependencies:

```bash
cd sq_ui/app
npm install
npm run build
```

## Run The UI

Start the SpaceFlow service:

```bash
python sq_ui/scripts/spaceflow_service.py
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
