# QB-PIER : Quantum Benchmarking Pipeline for Integrated Evaluation of Robustness

QB-PIER provides a set of tools to evaluate how robust classical and quantum neural networks are against adversarial attacks.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Training Pipeline](#training-pipeline)
- [Metrics & Logging](#metrics-storage)
- [Running Tests](#tests)
- [Documentation](#documentation)
- [Credits](#credits)

## Overview
The codebase focuses on benchmarking the adversarial robustness of different classifiers. The main training and attack pipelines live under `src/`. Models are written with PyTorch and PyTorch Lightning. Quantum layers rely on PennyLane. The project also includes Jupyter notebooks demonstrating training, attacks, and data exploration.

Results (checkpoints, figures and TensorBoard logs) are saved under a timestamped folder created by Hydra and consumed by `LoggingService`.

## Installation
1. Create a Python 3 environment.
2. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, run `make requirements` to verify the interpreter and install everything.
   The pipeline requires `hydra-core` version 1.3 or newer.
3. For a quick virtual environment setup you can run `make create_environment`.
   Use `make help` to list all available Makefile commands.
4. Run `python test_environment.py` to confirm the Python interpreter meets the
   requirements.

## Project Structure
```
├── src/                  # Main source code
│   ├── attacks/          # Adversarial attack implementations
│   ├── data/             # Dataset management
│   ├── metrics/          # Metric containers and logging utilities
│   ├── models/           # Classical, quantum and hybrid models
│   ├── visualisation/    # Figure generation helpers
│   └── experiments/      # Experiment utilities
├── conf/                 # Hydra configuration files
├── logs/                 # Training outputs created at runtime
├── tests/                # Pytest unit tests
├── notebooks/            # Example notebooks and archived code
├── docs/                 # Sphinx documentation
└── HELPER                # Convenience scripts for remote runs
```

## Running Experiments
Experiments are orchestrated with [Hydra](https://github.com/facebookresearch/hydra).
The entry point [`main.py`](src/main.py) is decorated with `@hydra.main` and reads
defaults from [`conf/`](conf/). Any parameter can be overridden on the command line
to modify a run without touching the code.

Device profiles under `conf/device/` make it easy to switch hardware settings:

```bash
# Use the Google Colab GPU setup
python src/main.py device=colab_t4

# Run on a 64-CPU cluster
python src/main.py device=cpu_cluster

# Default local configuration
python src/main.py device=local
```

Quantum circuit layouts are configured through ansatz files under
`conf/model/ansatz/`. Each YAML specifies the ansatz class and optional
parameters. Models include a `defaults` entry like `- ansatz: strong` to select
the desired circuit.

To sweep over multiple values you can enable Hydra's multi-run mode:

```bash
python src/main.py -m model=ampvqc optimizer.lr=0.001,0.0001
```

## Training Pipeline
All models inherit from [`BaseModel`](src/models/BaseModel.py), a thin wrapper
around `LightningModule` that configures optimizers, callbacks and logging.
`configure_training()` builds a `Trainer` with the desired devices and
hyperparameters, and the `fit()` method simply forwards to `Trainer.fit()` while
saving a final checkpoint and the collected metrics. This hierarchy keeps the
training loop consistent across the classical, quantum and hybrid models found
under `src/models/`.


## Attack Pipeline
The script [`attack.py`](src/attack.py) performs pairwise transfer attacks using Hydra.
Example usage:
```bash
python src/attack.py experiment_dir=./logs/my_experiment
python src/attack.py attack=pgd dataset=fmnist stride=10 experiment_dir=./logs/my_experiment
```
Configuration files live under `conf/attack/` and `conf/dataset/` allowing easy overrides of attack type, datasets, and other options.

<a id="metrics-storage"></a>
## Metrics & Logging
Training and validation results are aggregated through
[`MetricsContainer`](src/metrics/MetricsContainer.py) and stored with
[`pandas`](https://pandas.pydata.org/). During training values are also written
to TensorBoard for live monitoring.

Logs follow the structure:

```
experiment_dir/
└─ model_class_name/
   └─ model_config_name/
      └─ seed/
         ├─ checkpoints/
         ├─ metrics/
         ├─ tensorboard/
         └─ final.ckpt
```

An empty `.model` file is created inside each configuration directory to mark it
as a loadable model. Utilities can scan the project and load any directory
containing this anchor file. In particular `load_utils.load_experiment()` now
recursively searches a folder for these anchors and loads every matching model.

The experiment root contains an `experiment_info.json` similar to:

```json
{
  "pipeline_version": "0.1.0",
  "start_time": "2025-07-04T00:46:06.599638",
  "cuda_available": false,
  "configuration": {"DATA_DIR": "./data/processed", ...}
}
```

Metrics can be accessed via `model.metrics`, loaded with
`load_utils.load_metrics()` or inspected in TensorBoard using
`tensorboard --logdir logs`.
Model checkpoints can be reloaded with `load_utils.load_model(path, best=True)`
where the `best` flag selects the best checkpoint or the final one.
Multiple directories can be loaded at once with
`load_utils.load_from_dict({"name": "path/to/model"})` which returns a mapping of
the provided names to the loaded models.

<a id="tests"></a>
## Running Tests
Unit tests live in `tests/`. After installing the development dependencies you
can run:

```bash
make lint  # style check with flake8
make test  # execute pytest suite
```
Both commands should complete without errors once all required packages are
available.

## Documentation
Sphinx documentation lives in `docs/`. After installing `Sphinx`, you can build the HTML pages with:
```bash
cd docs
make html
```
Open `_build/html/index.html` in a browser to read the generated documentation.

## Credits
- Initial code was adapted by the work of Samuel Richard and I from the course GTI755.
- I used some code from my assignements in the course IFT6135.
- Parts of the project were structured with the help of ChatGPT.
