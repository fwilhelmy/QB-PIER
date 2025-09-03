# Adversarial Attacks

This package implements common adversarial attack algorithms used to test model robustness.

- `FGSMAttack.py` – Fast Gradient Sign Method for single-step perturbations.
- `PGDAttack.py` – Projected Gradient Descent for iterative attacks.
- `TransferAttack.py` – generates adversarial examples on a surrogate model for transferability studies.
- `BaseAttack.py` – shared utilities and interface for attack implementations.

Attacks produce perturbed inputs that aim to fool trained models, enabling evaluation of defensive strategies.
