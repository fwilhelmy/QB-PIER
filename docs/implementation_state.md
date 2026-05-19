# Implementation state

This file documents what is actually reliable right now, not what the config tree suggests is finished.

## Working / primary path

The currently reliable path is:

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

These configs or modules exist, but should not be treated as fully validated:

```text
conf/dataset/fmnist.yaml
conf/dataset/plusminus.yaml
conf/attack/pgd.yaml
src/attacks/PGD.py
```

They may work in pieces, but the documented baseline is still MNIST + FGSM.

## Quantum device state

Quantum device configuration is explicit instead of global.

Current intended flow:

```text
conf/qdevice/*.yaml
→ QuantumDeviceConfig
→ model / quantum layer / ansatz
→ PennyLane QNode
```

Current qdevice configs:

```text
sim.yaml    # no noise, analytic / infinite-shot style simulation
noisy.yaml  # finite-shot noisy simulator using PennyLane NoiseModel
```

Noise is built in `src/qdevice.py` using PennyLane's native insertion-based noise tools.

## Hydra state

Hydra should stay limited to:

```text
training
attacks
```

Do not add Hydra dependencies to analytics unless there is a strong reason.

Analytics should accept normal paths and arguments.

## Known limitations

- FGSM is the only attack that should be considered working.
- MNIST is the only dataset that should be considered working.
- PGD exists but is not part of the validated path yet.
- FMNIST and PlusMinus exist but are not part of the validated path yet.
- Some old notebooks are prototypes and should not be treated as maintained code.
- Some analytics scripts may still assume older result layouts or old Hydra output metadata.
- The training and attack paths still need end-to-end smoke tests after structural changes.

## Next implementation priorities

Recommended order:

1. Run a clean MNIST + LeNet training smoke test.
2. Run a clean MNIST + VQM training smoke test with `qdevice=sim`.
3. Run a clean MNIST + VQM training smoke test with `qdevice=noisy`.
4. Run FGSM on a small trained model set.
5. Validate or remove PGD.
6. Validate or remove non-MNIST datasets.
7. Clean analytics scripts that still assume old paths or old Hydra metadata.
