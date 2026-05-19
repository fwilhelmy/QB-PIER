# Artifacts

Training and attack outputs are runtime artifacts. They should usually live under `logs/` and should not be committed.

## Training artifacts

Training is launched with:

```bash
python src/train.py
```

Typical output root:

```text
logs/training/<date@time>/<hydra-overrides>/
```

Typical structure:

```text
logs/training/<run>/
├── .hydra/
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── .model
├── experiment.json
├── 0/
│   ├── final.ckpt
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
└── 42/
    ├── final.ckpt
    ├── checkpoints/
    ├── metrics/
    └── figures/
```

Important files:

- `.model`: model class name used by loading utilities.
- `experiment.json`: run-level metadata.
- `.hydra/`: resolved Hydra config and overrides.
- `<seed>/final.ckpt`: final checkpoint for one seed.
- `<seed>/checkpoints/`: intermediate checkpoints.
- `<seed>/metrics/`: saved training metrics.
- `<seed>/figures/`: saved training figures.

## Attack artifacts

Attacks are launched with:

```bash
python src/attack.py models_path=<path-to-trained-model-folder>
```

Typical output root:

```text
logs/attacks/attack_experiment/
```

Typical structure:

```text
logs/attacks/attack_experiment/
├── manifest.json
├── epsilons.json
├── tables/
└── seed_0/
    ├── labels.npy
    ├── audit_indices.npy
    ├── audit_clean_images_uint8.npy
    ├── baselines/
    │   └── <victim_model>/
    │       ├── logits_clean.npy
    │       └── preds_clean.npy
    ├── attacks/
    │   └── source=<source>__target=<target>.npy
    ├── attacker_audit_adv/
    │   └── source=<source>.npz
    └── tables/
        └── aggregates_<chunk>.parquet or .csv
```

Important files:

- `manifest.json`: metadata for the attack run.
- `epsilons.json`: epsilon grid.
- `labels.npy`: labels for the evaluated split.
- `baselines/<victim>/logits_clean.npy`: clean victim logits.
- `baselines/<victim>/preds_clean.npy`: clean victim predictions.
- `attacks/source=<source>__target=<target>.npy`: adversarial evaluation output for one source-target pair.
- `attacker_audit_adv/source=<source>.npz`: small adversarial image audit sample.
- `tables/`: aggregate result tables.

## Commit policy

Usually do not commit:

```text
logs/
data/processed/
__pycache__/
*.pyc
.ipynb_checkpoints/
```
