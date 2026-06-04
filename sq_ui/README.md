# Superquadric UI

Web UI for editing superquadrics and launching the SpaceFlow-related services in this checkout.

Default service ports:

- SuperDec: `http://127.0.0.1:11435`
- SuperFlex: `http://127.0.0.1:11436`
- TRELLIS/Create: `http://127.0.0.1:11437`
- SpaceFlow runs/assets: `http://127.0.0.1:11438`
- Ollama/Edit: `http://127.0.0.1:11434`

## SpaceFlow Service

```bash
export SQ_SPACEFLOW_PYTHON="$(pwd)/envs/guideflow3d/bin/python"
export SQ_SPACEFLOW_STORAGE_ROOT="$(pwd)/spaceflow_runtime"
python sq_ui/scripts/spaceflow_service.py
```

Health check:

```bash
curl -s http://127.0.0.1:11438/spaceflow/health | head
```

Set `SQ_SPACEFLOW_FORCE_LOCAL=1` when already running on a GPU node and you do not want the service to wrap requests in `srun`.

## Optional Services

```bash
python sq_ui/scripts/trellis_service.py
python sq_ui/scripts/superdec_service.py
python sq_ui/scripts/superflex_service.py
```

Use environment variables to point those services at external checkpoints or installs:

- `SQ_TRELLIS_PYTHON`, `SQ_TRELLIS_REPO_ROOT`, `SQ_TRELLIS_SCRATCH`
- `SUPERDEC_BASE`, `SQ_SUPERDEC_PYTHON`, `SQ_SUPERDEC_CHECKPOINT_DIR`
- `SUPERFLEX_BASE`, `SQ_SUPERFLEX_PYTHON`, `SQ_SUPERFLEX_CHECKPOINT_DIR`, `SQ_SUPERFLEX_WORK_DIR`

## Frontend

```bash
cd sq_ui/app
npm install
npm run dev -- --host 0.0.0.0
```

Open the printed Vite URL, usually `http://<host>:5173`.

## Open NPZ Files Directly

With the Vite dev server running:

```text
http://<host>:5173/?npz=/absolute/path/to/file.npz
```

By default the dev server allows `.npz` files from the repo and `spaceflow_runtime/`. Override with `SQ_UI_NPZ_ROOTS=/path/a:/path/b`.
