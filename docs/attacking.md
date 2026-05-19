# Attack pipeline

The attack pipeline is the second place where Hydra is allowed. Analytics should remain path-based and should not depend on Hydra.

Entrypoint:

```bash
python src/attack.py models_path=<path-to-trained-model-folder>
```

Default config:

```text
conf/attack.yaml
```

Default config groups:

```yaml
device: local
qdevice: sim
dataset: mnist
attack: fgsm
logging: verbose
model_load: final
models_path: ???
```

## Basic usage

Run FGSM on a trained model folder:

```bash
python src/attack.py \
  models_path=logs/training/2026-05-19@12-30-00 \
  model_load=final \
  dataset=mnist \
  attack=fgsm
```

Run with explicit quantum config:

```bash
python src/attack.py \
  models_path=logs/training/2026-05-19@12-30-00 \
  dataset=mnist \
  attack=fgsm \
  qdevice=sim
```

## Pipeline flow

The intended flow is:

```text
Hydra config
→ load trained models from models_path
→ load dataset
→ cache clean victim logits
→ generate adversarial examples by source model
→ evaluate every source-target pair
→ save logits, audit images, and aggregate tables
```

The current attack runner uses fixed epsilons from `0.00` to `1.00` in steps of `0.05`.

## Attack artifacts

Attack runs write artifacts under:

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

The important artifacts are:

- `manifest.json`: attack run metadata.
- `epsilons.json`: epsilon grid used during evaluation.
- `labels.npy`: labels for the evaluated dataset split.
- `audit_indices.npy`: selected indices for image audits.
- `audit_clean_images_uint8.npy`: clean audit images.
- `baselines/<victim_model>/logits_clean.npy`: cached clean logits.
- `baselines/<victim_model>/preds_clean.npy`: cached clean predictions.
- `attacks/source=<source>__target=<target>.npy`: adversarial evaluation output for one source-target pair.
- `attacker_audit_adv/source=<source>.npz`: adversarial audit images for one source model.
- `tables/`: aggregate result tables.

These notes are intentionally redundant with `docs/artifacts.md` because attack debugging usually requires knowing where logits and aggregate tables are saved.

## Current reliability

The only attack path that should be considered reliable right now is:

```text
MNIST dataset
+ FGSM attack
```

PGD exists in the codebase, but it is not part of the validated path yet.
