# Superquadric Editor UI

Interactive web-based editor for composing 3D shapes from superquadric primitives, producing `.npz` files compatible with the Spaceflow pipeline.

## Quick Start

```bash
cd sq_ui/app
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## AI Generate (Ollama on the ETH cluster)

Each person needs their **own** Ollama binary and model weights under **`/work/scratch/$USER`** (other users cannot read your scratch).

1. **One-time setup** from the repo root:

   ```bash
   bash sq_ui/setup_ollama.sh
   ```

   This downloads the Ollama Linux binary, pulls `gemma4:e2b` (configurable with `OLLAMA_MODEL`), and installs `ollama_proxy.py` + `ollama_infer.sh` into your scratch tree.

2. **Start the proxy** on the login node (it submits a short GPU job per request via Slurm):

   ```bash
   export OLLAMA_BASE=/work/scratch/$USER/spaceflow/superquadric_ui
   python3 "$OLLAMA_BASE/scripts/ollama_proxy.py"
   ```

3. **Run the UI** and ensure it talks to the proxy (default browser URL assumes `localhost:11434`):

   ```bash
   cd sq_ui/app
   # if needed:
   # echo 'VITE_OLLAMA_URL=http://127.0.0.1:11434/api/chat' >> .env.local
   npm run dev -- --host 0.0.0.0
   ```

**Without Slurm** (e.g. you run `ollama serve` yourself on port 11436):
`SQ_OLLAMA_FORWARD=http://127.0.0.1:11436 python3 "$OLLAMA_BASE/scripts/ollama_proxy.py"`

**Overrides:** `SQ_SLURM_PARTITION`, `SQ_SLURM_ACCOUNT`, `SQ_SLURM_GRES`, `SQ_SLURM_TIME`, `SQ_OLLAMA_SCRATCH` (install path for `setup_ollama.sh`).
**Slurm GPU override:** some clusters reject `--gres=gpu:1`. You can set `SQ_SLURM_GPUS=1` to use `--gpus=1` instead, or provide custom args via `SQ_SLURM_EXTRA_ARGS`.

**In the UI:** **AI Generate** → **Create** builds a new scene from a prompt; **Edit** sends the current preset (names, scales, shapes, translations, eulerDeg) plus your instruction (e.g. “make the wheels rounder”). Use **Focus parts** checkboxes or **+ selection** to steer which parts to change; the model still returns a full `primitives` list. With **Include viewport screenshot** (default on), a PNG of the 3D view is sent to multimodal models (e.g. Gemma 4) together with the text. Undo restores the previous scene.

## Usage

1. **Add primitives** using the "+ Add Primitive" button (presets: Ball, Ellipsoid, Cylinder, Cube, Astroid (star))
2. **Select a primitive** by clicking it in the left panel or the 3D viewport
3. **Adjust parameters** in the right panel:
   - **Scales (A, B, C)**: half-axes of the superellipsoid
   - **Shape (e₁, e₂)**: exponents — values <1 produce boxy shapes, 2 gives a sphere/ellipsoid, >2 produces pinched shapes
   - **Translation**: position in 3D space
   - **Rotation**: Euler angles (ZYX order, degrees)
4. **Export** via the Export button → "Download .npz"

## Templates

Use the toolbar buttons to load preset scenes:
- **⊙** Single Ellipsoid
- **⊞** Table (5 parts: top + 4 legs)
- **⊟** Chair (6 parts: seat + backrest + 4 legs)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Delete` / `Backspace` | Remove selected primitive |
| `D` | Duplicate selected |

## Export Format

The `.npz` file contains four arrays matching `run.py`'s `load_superquadric_from_file`:

| Key | Shape | Description |
|-----|-------|-------------|
| `scales` | `(N, 3)` | Half-axes A, B, C |
| `shapes` | `(N, 2)` | Exponents e₁, e₂ |
| `translations` | `(N, 3)` | Translation vectors |
| `rotations` | `(N, 3, 3)` | Rotation matrices |

## Pipeline Integration

```bash
python run.py --shape_superquadric_path path/to/exported.npz [other args...]
```

## Preview vs Pipeline

The editor shows a **pipeline-normalized view** by default (toggle in the Preview section). This replicates the normalization step in `load_superquadrics()`: centering the AABB and uniform-scaling to max extent = 1. The exported `.npz` stores **raw parameters** (pre-normalization).

## Validation

Run the compatibility test against exported files:

```bash
python sq_ui/scripts/test_npz_compat.py path/to/exported.npz
```
