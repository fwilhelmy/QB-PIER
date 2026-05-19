# Project structure

The project is organized around four main areas:

```text
conf/          Hydra configs for training and attacks
experiments/   notebooks and prototypes
src/           source code
docs/          plain markdown documentation
```

## Root structure

```text
project/
├── conf/
├── docs/
├── experiments/
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

## Experiments

```text
experiments/
├── models/
├── attacks/
└── prototypes/
```

Notebooks are allowed to be messy. Production logic should move into `src/`.

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
- `src/analytics/`: figures, tables, and utility scripts for analysis.
