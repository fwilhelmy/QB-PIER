# Training pipeline

Hydra is used for the training pipeline.

Entrypoint:

```bash
python src/train.py
```

Default config:

```text
conf/train.yaml
```

Default groups:

```yaml
device: local
qdevice: sim
dataset: mnist
model: lenet
training: dev
logging: verbose
```

## Common commands

```bash
python src/train.py model=lenet dataset=mnist training=debug
python src/train.py model=vqm dataset=mnist training=dev qdevice=sim
python src/train.py model=vqm dataset=mnist training=dev qdevice=noisy
python src/train.py model=hvql dataset=mnist training=full device=local
```

## Pipeline flow

```text
Hydra config
→ seed setup
→ DatasetFactory
→ ModelFactory
→ model.configure_training(...)
→ model.fit(...)
→ checkpoints, metrics, figures, logs
```

Quantum settings should be passed explicitly:

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

## Artifacts produced by training

Training writes artifacts under:

```text
logs/training/<date@time>/<hydra-overrides>/
```

Typical structure:

```text
logs/training/<run>/
├── .hydra/
├── .model
├── experiment.json
└── <seed>/
    ├── final.ckpt
    ├── checkpoints/
    ├── metrics/
    └── figures/
```

The key artifacts are:

- `.model`: model class name for loading.
- `.hydra/config.yaml`: resolved config snapshot.
- `.hydra/overrides.yaml`: command-line overrides.
- `<seed>/final.ckpt`: final checkpoint.
- `<seed>/metrics/`: saved metrics.
- `<seed>/figures/`: saved training figures.

For the full artifact contract, see `docs/artifacts.md`.

## Current reliability

The reliable baseline is MNIST training. Other datasets and some quantum paths exist, but they should be smoke-tested before being treated as stable.
