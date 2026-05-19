# QB-PIER : Quantum Benchmarking Pipeline for Integrated Evaluation of Robustness

QB-PIER provides a set of tools to evaluate how robust classical and quantum neural networks are against adversarial attacks.

The current reliable baseline is intentionally narrow: MNIST training with FGSM attack evaluation. Other datasets, attacks, and some quantum paths exist, but they should be treated as partially validated unless stated otherwise.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Training Pipeline](#training-pipeline)
- [Attack Pipeline](#attack-pipeline)
- [Metrics & Logging](#metrics--logging)
- [Artifacts](#artifacts)
- [Implementation State](#implementation-state)
- [Credits](#credits)

## Overview

QB-PIER is built around two main workflows:

1. train classical, quantum, or hybrid neural networks;
2. evaluate their robustness under adversarial attacks.

The main validated path is:

```text
MNIST dataset
+ training pipeline
+ FGSM attack pipeline
```

For the exact current state of the implementation, see `docs/implementation_state.md`.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd QB-PIER
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell
```

Install the required Python packages for your environment. The project depends on the usual PyTorch / Lightning / Hydra / PennyLane stack used by the training and attack pipelines.

If you maintain a project-specific requirements file, install it here:

```bash
pip install -r requirements.txt
```

If no requirements file is present, install the environment manually according to your local CUDA/PyTorch and PennyLane setup.

## Project Structure

High-level layout:

```text
conf/          Hydra configs for training and attacks
experiments/   notebooks and prototypes
src/           source code
docs/          markdown documentation
```

For the deeper folder-by-folder breakdown, see `docs/project_structure.md`.

## Running Experiments

Training and attacks are launched through Hydra entrypoints:

```bash
python src/train.py ...
python src/attack.py ...
```

Hydra should stay limited to training and attacks. Analytics scripts should use normal paths and normal function arguments.

For concrete commands, see:

- `docs/training.md`
- `docs/attacking.md`

## Training Pipeline

The training pipeline handles dataset creation, model construction, training, checkpointing, metrics, and figures.

Example:

```bash
python src/train.py model=lenet dataset=mnist training=debug
```

Quantum models can use explicit qdevice configs:

```bash
python src/train.py model=vqm dataset=mnist training=dev qdevice=sim
python src/train.py model=vqm dataset=mnist training=dev qdevice=noisy
```

For details on config flow and training outputs, see `docs/training.md`.

## Attack Pipeline

The attack pipeline loads trained models, generates adversarial examples, evaluates source-target transfer pairs, and saves logits/tables for later analysis.

Example:

```bash
python src/attack.py \
  models_path=logs/training/<run-folder> \
  model_load=final \
  dataset=mnist \
  attack=fgsm
```

For details on attack flow and attack outputs, see `docs/attacking.md`.

## Metrics & Logging

Training metrics, checkpoints, and attack evaluations are saved under `logs/`.

Training runs produce checkpoints, metrics, optional figures, and Hydra snapshots. Attack runs produce clean baselines, adversarial logits, audit images, epsilon metadata, and aggregate tables.

For the full save layout, see `docs/artifacts.md`.

## Artifacts

Artifacts are the generated files left by training and attack runs. They are useful for plotting, recomputing metrics, and avoiding reruns.

Most artifacts should not be committed:

```text
logs/
data/processed/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

For the full artifact contract, see `docs/artifacts.md`.

## Implementation State

The currently validated path is MNIST + FGSM. PGD, FMNIST, and PlusMinus exist but should not be treated as stable yet.

For the full implementation status, see `docs/implementation_state.md`.

## Credits

QB-PIER is developed for experimentation with quantum machine learning robustness, hybrid architectures, and adversarial evaluation workflows.
