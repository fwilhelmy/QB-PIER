# Logs & Experiment Folder Structure

This project writes experiment outputs under the `logs/` directory (or the
`LOGS_DIR`/`expdir` Hydra overrides). Training and attack runs create a
timestamped experiment root, then save per-model outputs in a consistent
nested layout.

## Experiment root

Each run produces an experiment root folder, typically:

```
logs/<expdir>/<YYYY-MM-DD@HH-MM-SS>/
```

The root contains an `experiment_info.json` file with metadata about the run
(e.g., pipeline version, start time, and resolved configuration overrides).

## Per-model layout

Inside the experiment root, outputs are organized by model class, model
configuration, and seed:

```
experiment_dir/
└─ <model_class_name>/
   └─ <model_config_name>/
      └─ <seed>/
         ├─ checkpoints/
         ├─ metrics/
         ├─ tensorboard/
         └─ final.ckpt
```

Notes:
- A `.model` anchor file is created in each model configuration directory to
  mark it as loadable by utilities that crawl experiment folders.
- `final.ckpt` is the final checkpoint; the `checkpoints/` folder may include
  additional or best checkpoints depending on configuration.
- `metrics/` stores aggregated metrics (via `MetricsContainer` and `pandas`).
- `tensorboard/` holds TensorBoard event files for live monitoring.

## Example overrides

- `LOGS_DIR=./logs` (default)
- `expdir=training` or `expdir=attacks` (from Hydra configs)

These overrides control the experiment root under `logs/`.
