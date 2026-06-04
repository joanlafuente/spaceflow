# SpaceFlow

SpaceFlow: Part-Wise Spatial and Semantic Guidance for Controllable 3D
Generation

SpaceFlow explores a practical question for text-to-3D systems: how can a user keep coarse geometric intent while still benefiting from a generative model's learned shape and appearance priors? The system uses a superquadric scene as an explicit control signal, applies local diffusion-time control over selected parts, and then refines the generated mesh with part-aware similarity guidance.

![Sailboat case study](docs/media/sailboat_case_study.png)

<video src="docs/media/sailboat_spin.mp4" controls muted loop playsinline poster="docs/media/sailboat_spin_poster.png" width="520"></video>

[View rotating sailboat preview](docs/media/sailboat_spin.mp4)

## Overview

Given a text prompt and an editable set of superquadrics, SpaceFlow produces a textured 3D asset through three stages:

1. **Spatial scaffold.** The UI exports all superquadrics, a high-control subset, and a low-control mask or bounding box.
2. **Structure generation.** TRELLIS generates sparse structure under SpaceControl constraints. Local-tau mode can apply different diffusion-time strengths to different spatial regions.
3. **Part-aware refinement.** PartField features and similarity guidance refine the generated structure toward global and local text or image appearance conditions.

The repository is a minimal delivery version of the project. It keeps the runtime pipeline, the superquadric editor, experiment launchers, local TRELLIS pipeline configuration, and the vendored code needed to reproduce the course experiments.

## Method

Superquadrics are used as a compact, editable proxy for user intent. They are expressive enough to block out object parts, but simple enough to manipulate interactively. Each primitive can be tagged as high-control or low-control. In local-tau experiments, high-control regions preserve the scaffold more strongly while low-control regions allow the generative prior to move more freely.

The local control variant passes the following signals into structure generation:

- `spatial_control_mesh.ply`: full superquadric control mesh.
- `high_control_spatial_control_mesh.ply`: subset of primitives that should be preserved more strongly.
- `low_control_superquadric_mask.ply`: low-control region used for local-tau masking.
- `shape_tau`, `shape_tau_high_control`, and `polyak_update_tau`: diffusion-time and model-averaging parameters for the local control schedule.

After structure generation, the mesh is normalized through Blender/TRELLIS tooling, voxelized, embedded with PartField, and optimized with similarity losses. The current pipeline supports global appearance prompts plus per-superquadric local text or image prompts.

## Qualitative Example

The sailboat case study was generated from:

- Prompt: `Sailboat`
- Global appearance: `white`
- Local appearance: `yellow` on four low-control primitives
- Local-tau settings: low tau `3`, high tau `10`, Polyak tau `0.18`

The figure above shows the intended control layout, the intermediate TRELLIS structure, and the final similarity-refined asset. The rotating preview is rendered from the resulting `out_sim.glb`.

Source run used for the media:

```text
/work/courses/3dv/team3/spaceflow_runtime/sq_ui_runs/20260603T204423Z_Sailboat_experiment
```

Regenerate the README media with:

```bash
/work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python docs/render_readme_media.py
```

## Repository Layout

- `run_local_tau.py`: main SpaceFlow pipeline entrypoint.
- `config/default.yaml`: runtime configuration and local TRELLIS pipeline path.
- `config/trellis_pipeline/pipeline.json`: mixed TRELLIS image/text model configuration used by spatial-control runs.
- `sq_ui/app`: React/Vite superquadric editor.
- `sq_ui/scripts/spaceflow_service.py`: HTTP service that saves SQ assets and launches local or Slurm runs.
- `sq_ui/scripts/run_spaceflow_experiment.py`: multi-variant experiment runner.
- `sq_ui/scripts/render_spaceflow_experiment_comparison.py`: CPU rasterizer for shared-view experiment figures.
- `docs/render_readme_media.py`: headless renderer for README figures and rotating previews.
- `lib`, `third_party`, `utils.py`: optimization, rendering, geometry, PartField, and TRELLIS runtime code.

## Running

Use the existing course environment when available. Otherwise run:

```bash
bash setup.sh
```

Build the editor:

```bash
cd sq_ui/app
npm install
npm run build
```

Start the backend service:

```bash
python sq_ui/scripts/spaceflow_service.py
```

Start the editor:

```bash
cd sq_ui/app
npm run dev -- --host 0.0.0.0
```

The service writes assets and runs under `spaceflow_runtime/` by default. Override paths with `SQ_SPACEFLOW_STORAGE_ROOT`, `SQ_SPACEFLOW_ASSET_ROOT`, or `SQ_SPACEFLOW_RUN_ROOT`.

## Runtime Notes

The pipeline expects TRELLIS and PartField checkpoints to be available through the configured Hugging Face cache or online download. The PartField checkpoint is expected at:

```text
third_party/PartField/models/model_objaverse.ckpt
```

For online Hugging Face access, keep `SQ_SPACEFLOW_OFFLINE_CACHE=0` or unset it. Offline cache mode can be enabled explicitly with:

```bash
SQ_SPACEFLOW_OFFLINE_CACHE=1 python sq_ui/scripts/spaceflow_service.py
```

The renderer uses Blender for pipeline mesh normalization. Set `SPACEFLOW_BLENDER_PATH=/path/to/blender` to use a specific Blender install.

## Status

This is a minimal research artifact, not a polished product package. Generated experiment outputs, Slurm logs, local datasets, downloaded model weights, and large checkpoints are intentionally excluded from version control. The tracked media in `docs/media/` is a small qualitative snapshot for the README.
