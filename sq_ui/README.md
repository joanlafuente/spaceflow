# Superquadric Editor UI

Web-based editor for composing 3D shapes from superquadric primitives and exporting `.npz` files for the Spaceflow pipeline.

This document is the **full setup guide** for running the UI with **SuperDec**, **TRELLIS (Create)**, and **Ollama / Gemma (Edit)** on a typical ETH Slurm cluster with scratch storage.

---

## What talks to what

| UI capability | What it does | Python service | Default port |
|---------------|--------------|----------------|--------------|
| **SuperDec** button | Fit superquadrics to an uploaded point cloud / mesh | `superdec_service.py` | `11435` |
| **Create** (AI Generate) | Text → point cloud (TRELLIS) → fit (SuperDec) | `trellis_service.py` **and** `superdec_service.py` | TRELLIS `11437`, SuperDec `11435` |
| **Edit** (AI Generate) | Change the current scene with natural language | `ollama_proxy.py` → Gemma on GPU | proxy `11434` |

The React app calls HTTP endpoints. **Development (`npm run dev`)**: Vite proxies `/superdec`, `/trellis`, and `/api` to `127.0.0.1` on the **login node**, so you can open the UI via the **Network** URL (e.g. `http://129.x:5173`) from your laptop without “failed to fetch” — as long as the Python services run on that same host. **Production build** (`npm run build`): set absolute `VITE_*_URL` values (or put the UI behind a reverse proxy). Optional: override proxy targets with `VITE_DEV_PROXY_SUPERDEC`, `VITE_DEV_PROXY_TRELLIS`, `VITE_DEV_PROXY_OLLAMA` when starting Vite.

---

## Prerequisites

- **Node.js** (for `sq_ui/app`): `npm install` / `npm run dev`
- **Slurm account** that can request GPUs on your cluster (defaults in the scripts use partition `interactive`, account `3dv`—change if yours differ)
- **Scratch space** under `/work/scratch/$USER/...` for weights, Ollama, caches (large downloads)
- **SuperDec**: run `setup_superdec.sh` once (clone, venv, checkpoints)
- **TRELLIS**: a Python env with TRELLIS dependencies. The repo expects **`envs/guideflow3d/bin/python`** at the **Spaceflow repo root** (set `SQ_TRELLIS_PYTHON` if yours differs)
- **Edit**: run `setup_ollama.sh` once (Ollama binary + Gemma weights in your scratch)

---

## First-time setup (recommended order)

Do this from the **Spaceflow repo root** (the directory that contains `sq_ui/` and `run.py`).

### 1) SuperDec (one-time install + weights)

```bash
cd /work/courses/3dv/team3/spaceflow

# Optional: own scratch root (default is /work/scratch/$USER/spaceflow/superdec_ui)
# export SQ_SUPERDEC_SCRATCH=/work/scratch/$USER/spaceflow/superdec_ui

bash sq_ui/setup_superdec.sh
```

Remember the printed **`Install root`** (below called `$SUPERDEC_BASE`). The script copies **`superdec_service.py`** and **`superdec_infer.py`** into `$SUPERDEC_BASE/scripts/` with paths filled in—**prefer starting that copy**, not the raw file under `sq_ui/scripts/`, unless you always set `SUPERDEC_BASE` yourself.

### 2) Ollama + Gemma for Edit (one-time)

```bash
cd /work/courses/3dv/team3/spaceflow

# Optional: export SQ_OLLAMA_SCRATCH=/work/scratch/$USER/spaceflow/superquadric_ui

bash sq_ui/setup_ollama.sh
```

Remember the printed **`OLLAMA_BASE`**.

### 3) Frontend dependencies

```bash
cd /work/courses/3dv/team3/spaceflow/sq_ui/app
npm install
```

### 4) Frontend URLs (`.env.local`) — optional in dev

With **`npm run dev`**, you usually **do not** need `.env.local`: the app uses **same-origin** paths and Vite forwards them to the services on the machine where Vite runs.

Add **`sq_ui/app/.env.local`** only if you want to override that (e.g. different ports or a remote service):

```bash
VITE_SUPERDEC_URL=http://127.0.0.1:11435
VITE_TRELLIS_URL=http://127.0.0.1:11437
VITE_OLLAMA_URL=http://127.0.0.1:11434/api/chat
```

**Note:** Typing `VITE_…=…` in a shell does **nothing** for Vite — variables must be in `.env.local` or the environment **when** you start `npm run dev`.

Restart `npm run dev` after changing env vars or `vite.config.ts`.

---

## Running the three services

Use **separate terminals** on the login node (or long-lived `tmux`/`screen` sessions). Order does not matter, but **Create** needs **both** TRELLIS and SuperDec up.

### A) SuperDec service

Always set **`SUPERDEC_BASE`** to the directory from `setup_superdec.sh` and **`SQ_SUPERDEC_CHECKPOINT_DIR`** to the normalized weights.

```bash
export SUPERDEC_BASE=/work/scratch/$USER/spaceflow/superdec_ui   # your actual path
export SQ_SUPERDEC_CHECKPOINT_DIR="$SUPERDEC_BASE/weights/normalized"

python3 "$SUPERDEC_BASE/scripts/superdec_service.py"
```

- Listens on **`http://127.0.0.1:11435`** by default (`SQ_SUPERDEC_PORT` to override).
- **Health check:** `curl -s http://127.0.0.1:11435/superdec/health | head`
- **Slurm:** inference is wrapped in `srun` with **`--gpus=1`** by default (`SQ_SUPERDEC_SLURM_GPUS` to change count or syntax your site expects).
- Other useful vars: `SQ_SUPERDEC_SLURM_PARTITION`, `SQ_SUPERDEC_SLURM_ACCOUNT`, `SQ_SUPERDEC_SLURM_TIME`, `SQ_SUPERDEC_SLURM_EXTRA_ARGS`, `SQ_SUPERDEC_TORCH_CUDA_ARCH_LIST`, `SQ_SUPERDEC_FORCE_LOCAL=1` (already on a GPU node).

### B) TRELLIS service (Create)

Run with **system `python3`**; the service spawns **`SQ_TRELLIS_PYTHON`** (defaults to `envs/guideflow3d/bin/python` under the repo) for the heavy job.

```bash
cd /work/courses/3dv/team3/spaceflow

export SQ_TRELLIS_REPO_ROOT="$(pwd)"
export SQ_TRELLIS_SLURM_GPUS=1
export SQ_TRELLIS_PYTHON="$(pwd)/envs/guideflow3d/bin/python"
export SQ_TRELLIS_SCRATCH=/work/scratch/$USER/spaceflow/trellis_ui

# If you are already on a GPU node and want one long-lived process (no srun per request):
# export SQ_TRELLIS_FORCE_LOCAL=1

python3 sq_ui/scripts/trellis_service.py
```

- Listens on **`http://127.0.0.1:11437`** (`SQ_TRELLIS_PORT`).
- **Health check:** `curl -s http://127.0.0.1:11437/trellis/health | head`
- Caches (Hugging Face, Torch, etc.) go under **`SQ_TRELLIS_SCRATCH`** / **`SQ_TRELLIS_CACHE_ROOT`** so they do not fill `~/.cache`.
- More options: `SQ_TRELLIS_SLURM_PARTITION`, `SQ_TRELLIS_SLURM_ACCOUNT`, `SQ_TRELLIS_SLURM_TIME`, `SQ_TRELLIS_SLURM_EXTRA_ARGS`.

### C) Ollama proxy (Edit)

```bash
export OLLAMA_BASE=/work/scratch/$USER/spaceflow/superquadric_ui   # your path from setup_ollama.sh

python3 "$OLLAMA_BASE/scripts/ollama_proxy.py"
```

- Listens on **`http://127.0.0.1:11434`** by default (`SQ_PROXY_PORT` to override).
- **Smoke test:** `curl -s http://127.0.0.1:11434/ | head` → should show *Ollama is running*.
- **Slurm GPU:** uses **`--gpus=1`** by default (`SQ_SLURM_GPUS` to override).
- **No Slurm / local Ollama:** `SQ_OLLAMA_FORWARD=http://127.0.0.1:11436 python3 "$OLLAMA_BASE/scripts/ollama_proxy.py"` (example port).

---

## Run the UI

```bash
cd /work/courses/3dv/team3/spaceflow/sq_ui/app
npm run dev -- --host 0.0.0.0
```

Open the printed URL (often `http://localhost:5173`). If you develop over SSH from your laptop, bind `0.0.0.0` and forward the port, or use the cluster’s published hostname.

---

## Service files (where they live)

| File | In repo | After setup |
|------|---------|-------------|
| SuperDec HTTP service | `sq_ui/scripts/superdec_service.py` (template; use with `SUPERDEC_BASE` or scratch copy) | `$SUPERDEC_BASE/scripts/superdec_service.py` |
| SuperDec worker | `sq_ui/scripts/superdec_infer.py` | copied to `$SUPERDEC_BASE/scripts/` |
| TRELLIS HTTP service | `sq_ui/scripts/trellis_service.py` | run from repo (no copy step) |
| TRELLIS worker | `sq_ui/scripts/trellis_infer.py` | used by service above |
| Ollama proxy + GPU wrapper | `sq_ui/scripts_templates/ollama_proxy.py`, `ollama_infer.sh` | `$OLLAMA_BASE/scripts/` after `setup_ollama.sh` |

---

## SuperDec: supported uploads

The UI uploads to the SuperDec service. Supported inputs include `.ply`, `.pcd`, `.xyz`, `.xyzn`, `.xyzrgb`, `.pts`, `.obj`, `.stl`.

- **Z-up** applies a basis fix to the editor’s Y-up frame.
- **Normalize point cloud** matches the generic-object inference path from the SuperDec demo.
- **LM optimization** is optional and slower; it needs a **GPU** (upstream LM code uses CUDA).
- **Editor scaling:** NPZ/JSON loads where **every** half-axis is tiny (max half-axis **&lt; 0.04**, typical SuperDec normalized fits) are **uniformly rescaled** so the **median** half-axis is about **2.5** (usable on the 0–5 sliders). **Translations** get the same factor so relative layout is unchanged. Templates and hand-sized presets are untouched. Programmatic import: pass **`skipEditorRescale: true`** to `importNpzToPrimitives` to keep raw numbers.

---

## Troubleshooting

1. **SuperDec / TRELLIS / Edit: “failed to fetch” in dev**
   - If you opened the UI as **`http://<cluster-ip>:5173`**, older setups sent API calls to **`localhost` on your laptop** (wrong). Current dev defaults use **Vite’s proxy**; restart **`npm run dev`** after pulling, and keep Python services on the **same host** as Vite.
   - **Do not** put **`http://<other-ip>:11435`** in `.env.local` unless that URL is the **same browser origin** as the page (same host **and** port as the dev server). A typo (e.g. `.130` vs `.131`) or a bare service port is a **different origin** — the browser will call a host your laptop often cannot reach. Prefer **no** `VITE_*` lines in dev; the app falls back to same-origin + proxy (and ignores cross-origin `VITE_*` in dev with a console warning).
   - If you set **`VITE_*`** to `http://127.0.0.1:...` but browse from another machine, that still means “localhost on the laptop” — remove those lines to use the proxy.

2. **“Cannot reach …” / CORS in the browser**
   - Confirm the service is listening: `curl` the health URLs above.
   - For a custom **`VITE_OLLAMA_URL`**, it must end with **`/api/chat`**.
   - Restart **`npm run dev`** after editing `.env.local`.

3. **Create does nothing or errors**
   - **Create needs both TRELLIS and SuperDec** running on the Vite host (or matching `VITE_*_URL` overrides).

4. **Slurm rejects GPU flags / mentions `--gres`**
   - All services use **`--gpus=...`** only (no `--gres`). Defaults: **`SQ_SUPERDEC_SLURM_GPUS=1`**, **`SQ_SLURM_GPUS=1`**, **`SQ_TRELLIS_SLURM_GPUS=1`**.
   - If you still see **`gres` in errors**, check **`SQ_SUPERDEC_SLURM_EXTRA_ARGS`**, **`SQ_TRELLIS_SLURM_EXTRA_ARGS`**, **`SQ_SLURM_EXTRA_ARGS`**, or your shell profile for **`--gres`** and remove it. The services strip `--gres` from those extras, but **refresh installed scripts** from the repo: `bash sq_ui/setup_superdec.sh` (with skips as needed) and re-copy the Ollama proxy via `bash sq_ui/setup_ollama.sh` so `superdec_service.py` and `ollama_proxy.py` match the repo.
   - Use **`SQ_*_SLURM_EXTRA_ARGS`** only for flags that are not GPU binding (or extra `--gpus`-compatible options your site documents).

5. **SuperDec points at the wrong install**
   - Always set **`SUPERDEC_BASE`** to **your** scratch tree from `setup_superdec.sh`, or run **`python3 "$SUPERDEC_BASE/scripts/superdec_service.py"`** from that install.

6. **TRELLIS `ModuleNotFoundError` or wrong Python**
   - Set **`SQ_TRELLIS_PYTHON`** to the interpreter that has TRELLIS deps (repo default: `envs/guideflow3d/bin/python`).

7. **Remote browser**
   Forward dev + service ports, e.g.
   `ssh -L 5173:localhost:5173 -L 11434:localhost:11434 -L 11435:localhost:11435 -L 11437:localhost:11437 user@login`

---

## Usage (editor)

1. **Add primitives** with “+ Add Primitive” (presets: Ball, Ellipsoid, Cylinder, Cube, Astroid).
2. **Select** in the left list or by clicking in the 3D view; **drag** a selected superquadric to change **translation**.
3. **Parameters** in the right panel: scales, shapes, translation, rotation (Euler ZYX, degrees).
4. **Export** → Download `.npz`.

**Templates** in the toolbar: single ellipsoid, table, chair.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Delete` / `Backspace` | Remove selected |
| `D` | Duplicate selected |

---

## Export format

The `.npz` contains arrays compatible with `run.py`’s `load_superquadric_from_file`:

| Key | Shape | Description |
|-----|-------|-------------|
| `scales` | `(N, 3)` | Half-axes A, B, C |
| `shapes` | `(N, 2)` | Exponents e₁, e₂ |
| `translations` | `(N, 3)` | Translations |
| `rotations` | `(N, 3, 3)` | Rotation matrices |

---

## Pipeline integration

```bash
python run.py --shape_superquadric_path path/to/exported.npz [other args...]
```

---

## Preview vs pipeline

The editor can show a **pipeline-normalized** preview (toggle in the Preview section). The exported `.npz` stores **raw** parameters (pre-normalization in the viewer).

---

## Validation

```bash
python sq_ui/scripts/test_npz_compat.py path/to/exported.npz
```

---

## Optional: `setup_superdec.sh` / `setup_ollama.sh` environment reference

**SuperDec setup**

- `SQ_SUPERDEC_SCRATCH` — install root (default `/work/scratch/$USER/spaceflow/superdec_ui`)
- `SUPERDEC_REPO_URL`, `SUPERDEC_REPO_REF`
- `SKIP_CLONE=1`, `SKIP_PIP=1`, `SKIP_CHECKPOINTS=1` to reuse trees

**Ollama setup**

- `SQ_OLLAMA_SCRATCH` — install root (default `/work/scratch/$USER/spaceflow/superquadric_ui`)
- `OLLAMA_VERSION`, `OLLAMA_MODEL` (default `gemma4:e2b`)
- `SKIP_DOWNLOAD=1`, `SKIP_PULL=1`

**Edit / multimodal**

With **Include viewport screenshot** (default on), the UI sends a PNG of the 3D view to the model together with the text. Use **Focus parts** or **+ selection** to steer which primitives to change; the model still returns a full `primitives` list.
