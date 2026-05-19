# AGENTS.md — Logs & Experiment Folder Structure (Training + Attacks)

This document defines the **logging contract** for training runs and attack evaluations.  
It is written for coding/plotting agents so they can **find artifacts reliably** and generate figures **without re-running experiments**.

---

## 1) Core principles (the contract)

1. **Experiment roots may vary**, but **per-model directories are invariant**.
2. Tooling should **never** assume a particular experiment naming scheme beyond:
   - there is an experiment root directory, and
   - it contains one folder per model configuration (plus optional global artifacts).
3. A directory is considered a valid model configuration folder **iff** it contains a `.model` file.
4. Attacks are stored **alongside** training outputs in the same experiment root (as `attack_experiment/`).

---

## 2) Experiment root layout

Inside an experiment root:

```text
<experiment_dir>/
├─ experiment_info.json              # optional global metadata
├─ attack_experiment/                # optional (present if attacks were run)
├─ <model_config_name_1>/            # per-model (invariant structure)
├─ <model_config_name_2>/
└─ ...
```

### `experiment_info.json` (optional)
Global metadata about the run (pipeline/version, timestamps, seeds, resolved overrides, environment info).
If missing, tooling should proceed using per-model `.hydra/` snapshots.

---

## 3) Training outputs (per-model layout — invariant)

Each model configuration has a fixed directory structure:

```text
<experiment_dir>/<model_config_name>/
├─ .model
├─ training.log
├─ model.info
├─ .hydra/
│  ├─ config.yaml
│  ├─ hydra.yaml
│  └─ overrides.yaml
└─ <seed>/
   ├─ checkpoints/
   ├─ metrics/
   │  ├─ logger.pkl
   │  ├─ metrics.csv
   │  └─ metrics.json                # optional
   ├─ tensorboard/
   │  ├─ events.out.tfevents.*
   │  └─ hparams.yaml
   ├─ profiler/                      # optional (if enabled)
   └─ final.ckpt
```

### 3.1 File semantics (training)

- `.model`  
  Anchor file identifying a valid model folder. It contains the model class name used by the loader.

- `training.log`  
  Plain-text training execution log.

- `model.info`  
  Architecture + training summary exported by the model at the end of training (includes training time/epoch, etc.).

- `.hydra/`  
  Full Hydra snapshot for reproducibility (`config.yaml`, `hydra.yaml`, `overrides.yaml`).

- `<seed>/checkpoints/`  
  Saved checkpoints during training (typically top-k based on `val/loss`).

- `<seed>/final.ckpt`  
  Final checkpoint at end of training.

- `<seed>/metrics/logger.pkl`  
  Raw serialized metric storage.

- `<seed>/metrics/metrics.csv`  
  Flattened tabular metrics export compiled from `logger.pkl`.

- `<seed>/tensorboard/`  
  TensorBoard events plus hyperparameters.

### 3.2 Training metrics (what exists)

Training logs, at minimum:
- `train/loss`, `train/acc`
- `val/loss`, `val/acc`
- `test/loss`, `test/acc`

Optional additional groups (if enabled):
- gradient norms (gradient profiler callback)
- device stats (device monitor callback)

Metrics are compiled into `metrics.csv` at the end of training.

---

## 4) Attack pipeline outputs

This folder is designed so downstream tooling (plotting, post-processing, paper figures) can be built **without re-running attacks**.

### 4.1 Global files

```text
attack_experiment/
├─ manifest.json
├─ epsilons.json
└─ tables/
   ├─ aggregates_seed=<seed>__source=<src>.parquet   (or .csv fallback)
   └─ ...
```

- `manifest.json`  
  Run metadata for the attack job (timestamp, Hydra run dir, resolved config snapshot, dtype notes).

- `epsilons.json`  
  Exact epsilon grid used.

- `tables/*.parquet` (preferred) or `tables/*.csv` (fallback)  
  Chunked aggregate tables, written per `(seed, attacker)` so the pipeline is robust to interruption and does not require parquet appends.

### 4.2 Per-seed files

```text
attack_experiment/
└─ seed_<seed>/
   ├─ labels.npy
   ├─ audit_indices.npy
   ├─ audit_clean_images_uint8.npy
   ├─ baselines/
   │  └─ <victim_name_safe>/
   │     ├─ logits_clean.npy
   │     └─ preds_clean.npy
   ├─ attacks/
   │  └─ source=<src_safe>__target=<tgt_safe>.npy
   ├─ attacker_audit_adv/
   │  └─ source=<src_safe>.npz
   └─ tables/
      └─ aggregates_seed.parquet (best-effort merge)
```

#### Dataset identity
- `labels.npy`  
  Ground-truth labels in the exact evaluated order. Shape `[N]`.

#### Audit subset (qualitative inspection, minimal storage)
- `audit_indices.npy`  
  Fixed indices used for audit images (default: **3 per class**, ~30 total).

- `audit_clean_images_uint8.npy`  
  Clean images for those indices (uint8 0–255). Shape `[K, H, W]`.

#### Clean baselines (per victim model)
Stored once per victim model to enable clean-vs-adv derived metrics:
- `baselines/<victim>/logits_clean.npy`  
  Victim logits on clean inputs (float16). Shape `[N, C]`.

- `baselines/<victim>/preds_clean.npy`  
  Clean argmax predictions (uint8). Shape `[N]`.

#### Attack logits (main artifact: “no rerun” guarantee)
For every attacker→victim pair and every epsilon:
- `attacks/source=<src>__target=<tgt>.npy`  
  Victim logits on adversarial inputs crafted by `src` at each epsilon (float16).  
  Shape **`[E, N, C]`**.

This is the critical artifact: from logits + labels (and clean baselines) you can recompute:
accuracy curves, confusion matrices, macro-F1, confidence/margin drops, transferability analyses, etc.

#### Audit adversarial images (per attacker)
Full images are stored only for the audit subset:
- `attacker_audit_adv/source=<src>.npz` containing:
  - `adv_images_uint8`: `[E, K, H, W]` (uint8 0–255)
  - `audit_indices`
  - `epsilons`
  - optional `audit_ok` (boolean mask over epsilons if an audit write failed for a given epsilon)

### 4.3 Aggregate metrics schema (attack tables)

Each row corresponds to one `(seed, source_model, target_model, epsilon)`:

Identifiers:
- `seed`, `source_model`, `target_model`, `epsilon`, `n_samples`

Core metrics:
- `robust_accuracy`  
  Fraction of samples with `pred_adv == y_true`.

- `flip_rate`  
  Fraction of samples with `pred_adv != pred_clean` (victim’s clean prediction).

- `ASR` (Attack Success Rate, untargeted, conditioned on victim clean-correct)  
  Among samples where `pred_clean == y_true`, fraction that become incorrect under attack.

Confidence / margin drops (computed from logits):
- `mean_confidence_drop_all`, `mean_margin_drop_all`  
  Mean drops computed over *all* samples.

- `mean_confidence_drop_cc`, `mean_margin_drop_cc`  
  Mean drops computed over the *clean-correct* subset (aligns with ASR conditioning).

Transfer metric:
- `transfer_gap`  
  `ASR_whitebox(source→source) - ASR_transfer(source→target)` at the same epsilon  
  (NaN if the whitebox row is missing).

---

## 5) Recommended tooling behavior (for agents)

### 5.1 Discovering trained models
- Crawl under a user-provided root path and treat directories containing `.model` as model configs.
- For each model config, iterate seed subfolders that are numeric.

### 5.2 Loading checkpoints
- `final.ckpt` is always present when training succeeded.
- `checkpoints/` may contain top-k checkpoints; “best” is usually lowest `val/loss` encoded in filename.

### 5.3 Using training metrics
- Prefer `metrics/logger.pkl` as the source of truth.
- Fall back to `metrics/metrics.csv`.

### 5.4 Using attack artifacts (preferred workflow)
- For aggregate plots: read parquet/csv chunks from `attack_experiment/tables/`.
- For metric recomputation or new metrics: use:
  - `seed_<seed>/baselines/<victim>/logits_clean.npy`
  - `seed_<seed>/attacks/source=<src>__target=<tgt>.npy`
  - `seed_<seed>/labels.npy`

Avoid relying exclusively on aggregate tables: logits cubes are the source of truth.

---

## 6) Coding rules for agents

These rules are part of the project contract. They are as important as the artifact layout above.

### 6.1 Hydra boundaries

- Hydra config is allowed only in the two real experiment pipelines:
  - `src/train.py`
  - `src/attack.py`
- No other path should launch through Hydra or depend on a live Hydra runtime.
- Analytics code may read old `.hydra/config.yaml` snapshots from saved runs when needed for metadata, but it must treat them as artifacts, not as runtime configuration.
- Analytics, table extraction, visualization, inspection, and utility code should use normal Python arguments, notebook config cells, or hard-coded constants.

### 6.2 No argparse unless explicitly requested

- Do not add `argparse` to new files unless the user explicitly asks for a CLI.
- For runnable utility scripts outside `train.py` and `attack.py`, use a simple `main()` with hard-coded arguments at the top of the file.
- Prefer obvious constants like:

```python
INPUT_DIR = "logs/ByModel"
OUTPUT_DIR = "outputs/<file_or_figure_name>"

def main() -> None:
    ...

if __name__ == "__main__":
    main()
```

### 6.3 Analytics figure workflow

Analytics figure modules should be library code only. Do not put a `main()` block in figure modules such as:

```text
src/analytics/model/hquanv.py
src/analytics/training/*.py
src/analytics/attacks/*.py
```

Each useful figure module should have a notebook beside it with the same base name, for example:

```text
src/analytics/model/hquanv.py
src/analytics/model/hquanv.ipynb
```

The notebook should drive the figure generation and should be split into exactly these practical blocks:

1. **Imports, utilities, and config**
   - default model root: `logs/ByModel`
   - default output path: `outputs/<figure_or_file_name>`
2. **Crawl and find models/artifacts**
   - discover model directories using `.model` anchors
   - discover attack folders/tables when working on attack figures
3. **Load models or prepare data**
   - inspect what was crawled
   - manually select a subset before expensive model loading or data preparation
4. **Generate the figure/table**
   - call the Python module functions
   - save under `outputs/<figure_or_file_name>`

The crawl and load/prepare blocks must stay separated. This lets the user inspect discovered models first, then choose which models to load or which attack rows to use.

### 6.4 Default paths

- Default model/artifact search root: `logs/ByModel`
- Default output root: `outputs/`
- Default output location for a figure/table should be `outputs/<file_or_figure_name>/` or `outputs/<file_or_figure_name>.<ext>` depending on what the code writes.

### 6.5 General coding style

- **Assuming root naming:** use `.model` as the anchor, not folder naming conventions.
- **Convoluted try/catch and verifications:** avoid bloating the code with convoluted verifications or try/catch during import or such. Make minimal implementation and verification will be added later as specified by the user.
- **TQDM usage:** when building looping behavior, make sure to add a tqdm bar to the loop.
- **Comparing means across heterogeneous variants:** attack aggregates are for computation convenience; conclusions should usually be drawn at the variant level when variants differ widely.
- **Audit images are sparse by design:** only the audit subset is stored as images; full-dataset visualization must be reconstructed from logits + original dataset if needed.

---

## 7) Quick glossary

- **Experiment root / experiment_dir**: one Hydra-resolved run folder containing multiple model configs.
- **Model config directory**: a folder with `.model` and seed subfolders; invariant structure.
- **Seed folder**: numeric folder under a model config, containing checkpoints + metrics.
- **Attack experiment folder**: `attack_experiment/` next to model folders; contains logits cubes + audit images + aggregate tables.