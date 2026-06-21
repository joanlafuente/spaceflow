# Geometry Control Metrics

This folder contains small scripts for measuring control monotonicity:

```text
Spearman(assigned control tau, -part-wise Chamfer distance)
```

Higher values mean stronger local control corresponds to lower geometry
preservation error.

## Evaluate Existing Outputs

Run the metric on an existing experiment root:

```bash
../guideflow3d/envs/guideflow3d/bin/python metrics/geometry_control_metrics.py \
  --batch-root spaceflow_runtime/sq_ui_runs/20260605T184428Z_examples_structure_experiment \
  --output-dir metrics/outputs/control_monotonicity_existing
```

Outputs:

- `per_primitive_geometry_control.csv`
- `control_monotonicity_summary.json`
- `control_monotonicity.png`

## Run A Low-Tau Sweep

Prepare and run two examples with low-control tau values `1,3,6` and high
control fixed at `10`:

```bash
../guideflow3d/envs/guideflow3d/bin/python metrics/run_low_tau_sweep.py \
  --examples 1,2 \
  --low-taus 1,3,6 \
  --high-tau 10 \
  --use-srun
```

Then evaluate the sweep:

```bash
../guideflow3d/envs/guideflow3d/bin/python metrics/geometry_control_metrics.py \
  --batch-root spaceflow_runtime/metrics_low_tau_sweep \
  --output-dir metrics/outputs/control_monotonicity_low_tau_sweep
```
