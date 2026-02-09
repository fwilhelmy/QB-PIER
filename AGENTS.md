# Logs & Experiment Folder Structure

All experiment outputs are written under `logs/` (or `LOGS_DIR` / `expdir` Hydra overrides). Each run creates a timestamped experiment root containing per-model results.

## Contract
- The experiment root may vary across pipelines and experiments.
- Inside each model directory, the structure is fixed and guaranteed.
- Tooling should rely on the per-model invariant, not root-level naming variations.

## Experiment root

```text
logs/<expdir>/YYYY-MM-DD@HH-MM-SS/
```

Represents one atomic run with resolved Hydra config.

```text
<experiment_dir>/
├─ experiment_info.json
├─ <model_name_1>/
└─ <model_name_2>/
```

### `experiment_info.json`
Global run metadata (pipeline/version, timestamps, seeds, resolved overrides, optional environment info).

## Per-model layout (invariant)

```text
<experiment_dir>/<model_name>/
├─ .model
├─ training.log
├─ model.info
├─ .hydra/
│  ├─ config.yaml
│  ├─ hydra.yaml
│  └─ overrides.yaml
└─ <seed>/
   ├─ checkpoints/
   ├─ metrics/
   │  ├─ logger.pkl
   │  └─ metrics.csv
   ├─ tensorboard/
   │  ├─ events.out.tfevents.*
   │  └─ hparams.yaml
   └─ final.ckpt
```

This per-model layout is identical across models and experiments.

## File semantics (brief)
- `.model`: Marker for a valid model directory (optionally minimal ID).
- `training.log`: Plain-text execution log (training/eval/attack).
- `model.info`: Structured model metadata (architecture, data, training setup).
- `.hydra/`: Exact Hydra snapshot for reproducibility.
- `<seed>/checkpoints/`: Intermediate checkpoints.
- `<seed>/metrics/logger.pkl`: Full serialized metric history.
- `<seed>/metrics/metrics.csv`: Flat tabular metrics export.
- `<seed>/tensorboard/`: TensorBoard logs and hyperparameters.
- `<seed>/final.ckpt`: Final trained checkpoint.
