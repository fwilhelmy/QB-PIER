# Attack pipeline

Hydra is used for the attack pipeline.

Entrypoint:

```bash
python src/attack.py models_path=<path-to-trained-model-folder>
```

Default config:

```text
conf/attack.yaml
```

Default groups:

```yaml
device: local
qdevice: sim
dataset: mnist
attack: fgsm
logging: verbose
model_load: final
models_path: ???
```

## Common command

```bash
python src/attack.py \
  models_path=logs/training/<run-folder> \
  model_load=final \
  dataset=mnist \
  attack=fgsm
```

## Pipeline flow

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

## Artifacts produced by attacks

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
    ├── baselines/
    ├── attacks/
    ├── attacker_audit_adv/
    └── tables/
```

The key artifacts are:

- `manifest.json`: attack run metadata.
- `epsilons.json`: epsilon grid.
- `labels.npy`: labels for the evaluated split.
- `baselines/<victim>/logits_clean.npy`: clean logits.
- `attacks/source=<source>__target=<target>.npy`: adversarial evaluation output.
- `tables/`: aggregate result tables.

For the full artifact contract, see `docs/artifacts.md`.

## Current reliability

The reliable baseline is MNIST + FGSM. PGD exists but is not validated yet.
