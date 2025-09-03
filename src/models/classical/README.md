# Classical Models

This directory contains implementations of classical neural networks used as baselines for robustness studies.

- `MLP.py` – fully connected multilayer perceptron.
- `MLPMixer.py` – MLP-Mixer architecture.
- `LeNet.py` – variant of the LeNet convolutional network.
- `ResNet18.py` – residual network with 18 layers.
- `ConvNet.py` – simple convolutional network for quick experiments.

All models inherit from `BaseModel` and can be instantiated through `ModelFactory`.
