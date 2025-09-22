# Hybrid Models

Hybrid models combine classical neural network layers with quantum circuits to leverage the strengths of both paradigms.

All reusable encoder and classifier building blocks now live inside
``models/hybrid/layers`` and expose concise names such as
``Convolution`` or ``VQC``.

- `AmpVQC.py` – amplitude encoding followed by a variational quantum circuit classifier.
- `HQE.py` – hybrid quantum encoder architecture with a final linear layer
  projecting the quantum outputs to class logits.
- `HQNN.py` – hybrid quantum neural network with pluggable encoders and
  classifiers. Supported encoders include convolutional, quanvolutional,
  linear and PCA-based variants, while classifiers can be a VQC,
  ParallelVQC or a classical MLP.
- `HQuanv.py` – hybrid quantum convolutional model.

Each model subclasses `BaseModel` and is registered with `ModelFactory` for easy instantiation.
