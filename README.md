# QMLBenchmark

QMLBenchmark provides a set of tools to evaluate how robust classical and quantum neural networks are against adversarial attacks. The repository contains implementations of convolutional networks, hybrid quantum-classical models, and utilities to generate and test adversarial examples.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Documentation](#documentation)
- [Helper Commands & HPC Usage](#helper-commands--hpc-usage)
- [Credits](#credits)

## Overview
The codebase focuses on benchmarking the adversarial robustness of different classifiers. The main training and attack pipelines live under `src/` while an alternative implementation can be found in `newsrc/`. Models are written with PyTorch and PyTorch Lightning. Quantum layers rely on PennyLane. The project also includes Jupyter notebooks demonstrating training, attacks, and data exploration.

Supported components include:
- **Classical models** like LeNet.
- **Hybrid models** combining convolutional layers with variational quantum circuits.
- **Adversarial attacks** such as FGSM and PGD.
- **Dataset managers** for MNIST and Fashion-MNIST.
- **Visualization utilities** for training curves, circuit layouts, and attack results.

Results (checkpoints, figures and TensorBoard logs) are saved under a timestamped folder in `results/` via `LoggingService`.

## Installation
1. Create a Python 3 environment.
2. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, run `make requirements` to verify the interpreter and install everything.
3. For a quick virtual environment setup you can run `make create_environment`.
   Use `make help` to list all available Makefile commands.
4. Run `python test_environment.py` to confirm the Python interpreter meets the
   requirements.

## Project Structure
```
├── src/                  # Main source code
│   ├── attacks/          # FGSM, PGD and attack base classes
│   ├── data/             # Dataset download and loaders
│   ├── layers/           # Quantum layers and helpers
│   ├── models/           # Classical, quantum and hybrid models
│   ├── visualisation/    # Matplotlib utilities
│   ├── LoggingService.py # Handles results directory & TensorBoard
│   └── main.py           # Example training and evaluation script
├── newsrc/               # Alternative implementation with similar layout
├── notebooks/            # Jupyter notebooks demonstrating experiments
├── docs/                 # Sphinx documentation
├── results/              # Created at runtime to store logs and checkpoints
└── HELPER                # Convenience commands for remote runs
```

The `newsrc/` package mirrors the structure of `src/` and contains experimental versions of the models and training helpers.

## Running Experiments
Use `src/main.py` to train models and evaluate attacks from the command line. Basic usage:
```bash
python src/main.py -d 1 -w 4
```
This trains a set of models on MNIST using one device and four worker threads, then performs FGSM and PGD attacks. Figures and model checkpoints will be written inside `results/`.

For interactive experimentation, browse the notebooks under `notebooks/`. They include demonstrations of adversarial training, PGD experiments and visualization utilities.

## Documentation
Sphinx documentation lives in `docs/`. After installing `Sphinx`, you can build the HTML pages with:
```bash
cd docs
make html
```
Open `_build/html/index.html` in a browser to read the generated documentation.

## Helper Commands & HPC Usage
The `HELPER` file at the project root lists useful snippets for working on a
remote cluster. A typical workflow is:

```bash
ssh -l <user> replica.logti.etsmtl.ca
source miniconda3/bin/activate && conda activate QAML && cd QMLBenchmark
nohup python QMLBenchmark/src/main.py > funsies.out 2> oopsies.err < /dev/null &
tail -f funsies.out
```

The file also shows how to copy results back using `scp` once the job
completes.

## Credits
- Initial code for the quantum layers and FGSM implementation was adapted from work by Samuel Richard and [Nathan Inkawhich](https://github.com/inkawhich).
- Parts of the project were structured with the help of ChatGPT.
- The repository is based on the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) template.

Computations were performed on the **name-of-the-server** supercomputer, managed by Calcul Québec and the Digital Research Alliance of Canada.
