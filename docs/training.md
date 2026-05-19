# Training pipeline

The training pipeline is the main place where Hydra should be used.

Entrypoint:

```bash
python src/train.py
```

Default config:

```text
conf/train.yaml
```

Default config groups:

```yaml
device: local
qdevice: sim
dataset: mnist
model: lenet
training: dev
logging: verbose
```

## Basic usage

Run the default training job:

```bash
python src/train.py
```

Run a small debug job:

```bash
python src/train.py model=lenet dataset=mnist training=debug
```

Run a quantum model with exact simulation:

```bash
python src/train.py model=vqm dataset=mnist training=dev qdevice=sim
```

Run a quantum model with finite-shot noisy simulation:

```bash
python src/train.py model=vqm dataset=mnist training=dev qdevice=noisy
```

Run a longer training config:

```bash
python src/train.py model=hvql dataset=mnist training=full device=local
```

## Pipeline flow

The intended flow is:

```text
Hydra config
→ seed setup
→ DatasetFactory
→ ModelFactory
→ model.configure_training(...)
→ model.fit(...)
→ checkpoints, metrics, figures, logs
```

Quantum configuration is passed explicitly:

```text
conf/qdevice/*.yaml
→ train.py
→ ModelFactory.create(..., quantum_config=...)
→ model
→ quantum layer
→ ansatz
→ PennyLane QNode
```

Do not reintroduce global quantum state.

## Quantum device configs

Current qdevice configs:

```text
conf/qdevice/sim.yaml
conf/qdevice/noisy.yaml
```

`sim.yaml` is the default exact simulator path:

```text
default.qubit
shots: null
noise: disabled
```

`noisy.yaml` is the finite-shot noisy path:

```text
default.mixed
shots: 1000
noise: PennyLane NoiseModel
```

Noise construction should stay centralized in `src/qdevice.py`.

## Training artifacts

Training writes artifacts under:

```text
logs/training/<date@time>/<hydra-overrides>/
```

Typical structure:

```text
logs/training/2026-05-19@12-30-00/model=lenet,dataset=mnist/
├── .hydra/
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── .model
├── experiment.json
├── 0/
│   ├── final.ckpt
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
└── 42/
    ├── final.ckpt
    ├── checkpoints/
    ├── metrics/
    └── figures/
```

The important artifacts are:

- `.model`: model class name for loading.
- `experiment.json`: run metadata.
- `.hydra/config.yaml`: resolved config snapshot.
- `.hydra/overrides.yaml`: command-line overrides.
- `<seed>/final.ckpt`: final model checkpoint for that seed.
- `<seed>/checkpoints/`: intermediate checkpoints.
- `<seed>/metrics/`: saved metrics.
- `<seed>/figures/`: saved training figures.

These notes are intentionally redundant with `docs/artifacts.md` because training users usually need the artifact layout directly beside the training commands.

## Current reliability

The only training path that should be considered reliable right now is:

```text
MNIST dataset
+ standard training pipeline
```

Other datasets and some quantum paths may exist, but they still need validation.
