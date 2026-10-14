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

## Export Training Dynamics

Use the artifact-only exporter to create clean CSV files for downstream
training-dynamics analysis:

```bash
python scripts/export_training_dynamics.py --models-root <PATH_TO_MODELS> --output-dir <OUTPUT_DIR>
```

Example for the default model artifact tree:

```bash
python scripts/export_training_dynamics.py --models-root logs/ByModel --output-dir outputs/training_dynamics_export --overwrite true
```

The exporter crawls `.model` anchors and reads existing files such as
`metrics.csv`, `metrics.json`, `model.info`, Hydra YAML snapshots, and
TensorBoard `hparams.yaml`. It does not train models, run attacks, or load
checkpoint tensors. TensorBoard event scalar parsing is optional:

```bash
python scripts/export_training_dynamics.py --models-root logs/ByModel --output-dir outputs/training_dynamics_export --include-tensorboard-events true --overwrite true
```

Validate the outputs by opening `extraction_quality_report.csv` first, then
checking `extraction_summary.md` for missing metrics, malformed files, metric
naming variants, and accuracy-scale notes. Use `training_curves_long.csv` for
flexible plotting and `training_curves_wide.csv` for epoch-aligned summaries.

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

### Author and research

**QB-PIER (Quantum Benchmarking Pipeline for Integrated Evaluation of Robustness)** was designed and developed by **Félix Wilhelmy** as the experimental platform for his master's research at the **École de technologie supérieure (ÉTS), Université du Québec**. The research was supervised by **Patrick Cardinal** and co-supervised by **Olivier Landon-Cardinal**.

The project also benefited from the academic environment at ÉTS and the QuantumÉTS community, as well as discussions and guidance from members of the wider quantum computing and machine learning research communities. The full acknowledgements are included in the thesis.

**Related master's thesis:**

> Wilhelmy, F. (2026). *Robustesse adversariale en apprentissage automatique classique, quantique et hybride pour la classification d’images*. Master's thesis, École de technologie supérieure, Montréal, Québec.

For academic work based on the benchmark, its methodology, or its experimental results, please cite the thesis:

```bibtex
@mastersthesis{wilhelmy2026robustesse,
  author  = {Wilhelmy, Félix},
  title   = {Robustesse adversariale en apprentissage automatique classique, quantique et hybride pour la classification d'images},
  school  = {École de technologie supérieure},
  address = {Montréal, Québec},
  year    = {2026}
}
```

### AI-assisted development

**ChatGPT and Codex (OpenAI)** were used in developing and maintaining QB-PIER. As disclosed in the thesis, other AI-assisted tools used during the broader research process included **Cursor with Claude Opus (Anthropic)** and **Google NotebookLM**. These tools helped with learning, prototyping, repetitive programming tasks, code and results analysis, scripts, figures, and documentation.

AI-generated suggestions and outputs were reviewed and checked against the source code, experimental data, and relevant references, and were corrected or rejected where necessary. **All methodological choices, experiments, interpretations, and final project content remain the responsibility of Félix Wilhelmy.** AI tools are credited for assistance, not as authors or scientific sources of authority.

### Open-source libraries and development tools

QB-PIER builds on open-source software developed and maintained by their respective communities. Key libraries and tools used in the project include:

- **Neural networks and training:** [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/stable/), and [Lightning](https://lightning.ai/docs/pytorch/stable/) for models, data loading, training, metrics, and checkpoints.
- **Quantum computing:** [PennyLane](https://pennylane.ai/) for quantum circuits, simulation, quantum-classical interfaces, and differentiation.
- **Experiment configuration:** [Hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/), and [PyYAML](https://pyyaml.org/) for configuration-driven experiments and reproducibility.
- **Numerical and statistical analysis:** [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), and [scikit-learn](https://scikit-learn.org/) for processing data, evaluating results, and implementing analysis utilities.
- **Visualization and experiment inspection:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), and [TensorBoard](https://www.tensorflow.org/tensorboard) for figures and training diagnostics.
- **Additional workflow tools:** [Git](https://git-scm.com/) for version control and experiment traceability, [Jupyter](https://jupyter.org/) for exploratory analysis, and utility packages including `tqdm`, `Pillow`, `psutil`, and `discord-webhook`.

This is an acknowledgement of the principal tools rather than an exhaustive or version-pinned dependency list. Some packages support optional analysis or auxiliary features. All third-party packages remain subject to their respective licenses and attribution requirements.
