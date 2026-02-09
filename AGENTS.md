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
<experiment_dir>/
└─ <model_name>/
      ├─ .model
      ├─ training.log
      ├─ model.info
      ├─ .hydra/
         ├─ config.yaml
         ├─ hydra.yaml
         └─ overrides.yaml
      └─ <seed>/
         ├─ checkpoints/
         ├─ metrics/
            ├─ logger.pkl
            └─ metrics.csv
         ├─ tensorboard/
            ├─ <tensorboard output file>
            └─ hparams.yaml
         └─ final.ckpt
```