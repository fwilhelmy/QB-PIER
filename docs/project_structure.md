# Project structure

This public source distribution is organized around three main areas:

```text
conf/          Hydra configs for training and attacks
src/           source code
docs/          plain markdown documentation
```

## Root structure

```text
project/
├── conf/
├── docs/
├── src/
├── data/
└── logs/
```

`data/` and `logs/` are runtime folders and should generally not be committed.

## Docs

```text
docs/
├── artifacts.md
├── training.md
├── attacking.md
├── implementation_state.md
└── project_structure.md
```

## Configs

```text
conf/
├── train.yaml
├── attack.yaml
├── attack/
├── dataset/
├── device/
├── logging/
├── model/
├── qdevice/
└── training/
```

Hydra should stay limited to training and attacks.

## Source code

```text
src/
├── analytics/
├── attacks/
├── datasets/
├── layers/
├── metrics/
├── models/
├── attack.py
├── train.py
├── loading.py
├── qdevice.py
└── utils.py
```

Main responsibilities:

- `src/train.py`: training entrypoint.
- `src/attack.py`: attack entrypoint.
- `src/qdevice.py`: explicit quantum device, shots, and noise configuration.
- `src/models/`: model definitions and model factory.
- `src/layers/`: classical, hybrid, and quantum layers.
- `src/datasets/`: dataset wrappers and dataset factory.
- `src/attacks/`: attack implementations and attack factory.
- `src/metrics/`: metric containers/profilers used during training.
- `src/analytics/`: generic training plots and metric helpers required by the training entrypoint. Research-specific analytics and notebooks are excluded.

Research notebooks, locally downloaded datasets, and experimental results are not part of the public source distribution. See `PUBLIC_RELEASE_CHECKLIST.md`.
