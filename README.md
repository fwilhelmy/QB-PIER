# QB-PIER : Quantum Benchmarking Pipeline for Integrated Evaluation of Robustness

QB-PIER provides a set of tools to evaluate how robust classical and quantum neural networks are against adversarial attacks.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Documentation](#documentation)
- [Credits](#credits)

## Overview
The codebase focuses on benchmarking the adversarial robustness of different classifiers. The main training and attack pipelines live under `src/`. Models are written with PyTorch and PyTorch Lightning. Quantum layers rely on PennyLane. The project also includes Jupyter notebooks demonstrating training, attacks, and data exploration.

Results (checkpoints, figures and TensorBoard logs) are saved under a timestamped folder in `logs/` via `LoggingService`.

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
│   ├── attacks/          # Adversarial attacks classes
│   ├── data/             # Dataset management
│   ├── logging           # Logging tools
│   ├── models/           # Classical, quantum and hybrid models
│   ├── visualisation/    # Figures generation functions
│   └── experiments/      # Configuration for running different experiments 
├── docs/                 # Sphinx documentation
├── logs/                 # Created at runtime to store logs and checkpoints
├── tests/                # Pytest unit tests
└── HELPER                # Convenience commands for remote runs
```

## Running Experiments
The script `src/main.py` shows how to configure an experiment with the
`Arguments` class. Setting `num_threads` limits the CPU threads used by PyTorch via `torch.set_num_threads`, which can be helpful on shared machines.

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
