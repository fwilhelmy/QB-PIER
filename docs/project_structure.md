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

`data/` and `logs/` are generated/runtime folders and should generally not be committed.

## Docs

```text
docs/
├── artifacts.md
├── training.md
├── attacking.md
├── implementation_state.md
└── project_structure.md
```

Purpose:

- `artifacts.md`: where generated training and attack files are saved.
- `training.md`: how to run training and what training leaves behind.
- `attacking.md`: how to run attacks and what attacks leave behind.
- `implementation_state.md`: what currently works and what is only partially validated.
- `project_structure.md`: overview of the repository layout.

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

Hydra should stay limited to:

```text
training
attacks
```

Analytics should not require Hydra.

## Experiments

```text
experiments/
├── models/
├── attacks/
└── prototypes/
```

Purpose:

- `models/`: model exploration and training notebooks.
- `attacks/`: attack experiments and attack tutorials.
- `prototypes/`: UI tests, scratch notebooks, TensorBoard tests, and old exploratory work.

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

## Analytics

```text
src/analytics/
├── attacks/
├── model/
├── scripts/
├── tables/
├── training/
└── utils.py
```

Analytics should be practical and path-based. Prefer functions that accept:

```text
input_path
output_path
model filters
attack filters
save options
```

Avoid adding more global state or Hydra coupling here.

## Quantum device design

Quantum settings should be passed explicitly:

```text
Hydra qdevice config
→ train.py / attack.py
→ ModelFactory
→ model
→ quantum layer
→ ansatz
→ QNode
```

Do not reintroduce module-level global quantum state.
