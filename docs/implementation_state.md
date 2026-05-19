# Implementation state

This file documents what is actually reliable right now.

## Working baseline

The reliable path is:

```text
MNIST dataset
+ training pipeline
+ FGSM attack pipeline
+ basic transfer-attack storage/evaluation
```

Main supported dataset:

```text
conf/dataset/mnist.yaml
```

Main supported attack:

```text
conf/attack/fgsm.yaml
```

## Partially implemented or not validated

These exist but should not be treated as fully validated:

```text
conf/dataset/fmnist.yaml
conf/dataset/plusminus.yaml
conf/attack/pgd.yaml
src/attacks/PGD.py
```

## Quantum device state

Quantum device configuration should be explicit instead of global.

Current intended flow:

```text
conf/qdevice/*.yaml
→ QuantumDeviceConfig
→ model / quantum layer / ansatz
→ PennyLane QNode
```

`sim.yaml` is the no-noise simulator config. `noisy.yaml` is intended for finite-shot noisy simulation.

## Hydra state

Hydra should stay limited to:

```text
training
attacks
```

Analytics should accept normal paths and arguments.

## Known limitations

- FGSM is the only attack that should be considered working.
- MNIST is the only dataset that should be considered working.
- PGD exists but is not part of the validated path yet.
- FMNIST and PlusMinus exist but are not part of the validated path yet.
- Some notebooks are prototypes, not maintained production code.
- Some analytics scripts may still assume older result layouts.
