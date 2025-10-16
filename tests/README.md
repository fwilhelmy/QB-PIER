# Test Suite Overview

The `tests/` directory is organized to mirror the major domains of QB-PIER.  Each
sub-directory groups together related unit and integration tests so new
contributions can easily locate the right place for additional coverage.

## Directory layout

- `attacks/`
  - Adversarial attack unit tests (e.g., FGSM, PGD) that validate perturbation
    logic in `src/attacks`.
- `datasets/`
  - Tests for data loading utilities and experiment discovery helpers.
- `layers/`
  - Quantum and classical layer coverage, including ansatz depth checks and
    convolutional primitives.
- `models/`
  - Reserved for model-specific tests.  Add new files here when covering model
    training or inference behaviour directly.

`conftest.py` remains at the root to configure shared fixtures and ensure the
project's `src/` package is importable from any test module.
