# Dataset Managers

The `data` module centralises dataset loading and preprocessing utilities for experiments.

- `MNISTManager.py` – downloads and prepares the MNIST handwritten digit dataset.
- `FMNISTManager.py` – handles the Fashion-MNIST clothing dataset.
- `PlusMinusManager.py` – generates a simple two-class synthetic dataset.
- `DatasetManager.py` – base class offering shared functionality like splits and transforms.
- `DatasetManagerFactory.py` – HydraFactory-based registry to create dataset managers by name.

These managers provide train/validation/test splits and normalisation used across the training and attack pipelines.
