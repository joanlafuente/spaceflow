# SpaceFlow UI

Web UI for editing superquadrics and launching SpaceFlow runs from this checkout.

Default service port:

- SpaceFlow runs/assets: `http://127.0.0.1:11438`

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

By default the dev server allows `.npz` files from the repo, `spaceflow_runtime/`,
the shared sibling `../spaceflow_runtime/`, and the course dataset folder at
`../spaceflow/datasets/` when those paths exist. Override with
`SQ_UI_NPZ_ROOTS=/path/a:/path/b`.
