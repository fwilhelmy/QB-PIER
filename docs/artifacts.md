# Artifacts

This project writes runtime artifacts under `logs/`. Training and attack artifacts are different, but both should be treated as generated outputs, not source code.

## Training artifacts

Training is launched with:

```bash
python src/train.py
```

Default config:

```text
conf/train.yaml
```

Typical output root:

```text
logs/training/<date@time>/<hydra-overrides>/
```

Typical run structure:

```text
logs/training/2026-05-19@12-30-00/model=lenet,dataset=mnist/
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

- `.model`: stores the model class name used by loading utilities.
- `experiment.json`: stores run-level metadata.
- `.hydra/`: stores the resolved Hydra config and overrides.
- `<seed>/final.ckpt`: final checkpoint for one seed.
- `<seed>/checkpoints/`: intermediate checkpoints if checkpointing is enabled.
- `<seed>/metrics/`: training metrics when produced by the model/training pipeline.
- `<seed>/figures/`: training figures when produced by the model/training pipeline.

## Attack artifacts

Attacks are launched with:

```bash
python src/attack.py models_path=<path-to-trained-model-folder>
```

Default config:

```text
conf/attack.yaml
```

Typical output root:

```text
logs/attacks/attack_experiment/
```

Typical attack structure:

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

- `manifest.json`: metadata describing the attack run.
- `epsilons.json`: list of epsilon values used for evaluation.
- `labels.npy`: clean labels for the evaluated split.
- `audit_indices.npy`: sampled indices used for audit images.
- `audit_clean_images_uint8.npy`: clean audit images.
- `baselines/<victim_model>/logits_clean.npy`: cached clean logits for a victim model.
- `baselines/<victim_model>/preds_clean.npy`: cached clean predictions for a victim model.
- `attacks/source=<source>__target=<target>.npy`: adversarial evaluation result for one source-target pair.
- `attacker_audit_adv/source=<source>.npz`: small adversarial image audit sample.
- `tables/`: aggregate result tables.

## Commit policy

Do not commit generated artifacts unless there is a very specific reason.

Usually ignored/generated:

```text
logs/
data/processed/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Raw datasets, trained checkpoints, attack logits, and generated figures should usually stay out of Git.
