# Hybrid Models

Hybrid models combine classical neural network layers with quantum circuits to leverage the strengths of both paradigms.

- `AmpVQC.py` – amplitude encoding followed by a variational quantum circuit classifier.
- `HQE.py` – hybrid quantum encoder architecture.
- `HQNN.py` – hybrid quantum neural network.
- `HQuanv.py` – hybrid quantum convolutional model.

Each model subclasses `BaseModel` and is registered with `ModelFactory` for easy instantiation.
