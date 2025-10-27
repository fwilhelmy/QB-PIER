import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, NullLocator

from metrics.LoggingService import LoggingService

log = logging.getLogger(__name__)

@rank_zero_only
def plot_models_together(trained_models):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')
    ax_train_loss, ax_val_loss = ax[0]
    ax_train_acc,  ax_val_acc  = ax[1]

    # Titles and Y-labels
    ax_train_loss.set_title('Training', fontweight='bold')
    ax_val_loss.set_title('Validation', fontweight='bold')
    ax_train_loss.set_yscale('log')
    ax_val_loss.set_yscale('log')

    ax_train_loss.set_ylabel('Loss')
    ax_train_acc.set_ylabel('Accuracy (%)')
    ax_val_loss.set_ylabel('')   # no ylabel on the two right-hand plots
    ax_val_acc.set_ylabel('')

    def _ensure_array(data, dtype=None):
        if isinstance(data, dict) or data is None:
            return np.array([], dtype=dtype if dtype is not None else float)
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    # Plot each model’s mean±std curves
    for name, seed_dict in trained_models.items():
        losses_tr, losses_val, accs_tr, accs_val = [], [], [], []
        train_steps_ref, val_steps_ref = None, None

        for s in seed_dict.values():
            metrics = s.metrics

            train_steps = _ensure_array(metrics['train/step'])
            val_steps   = _ensure_array(metrics['val/step'])
            train_loss  = _ensure_array(metrics['train/loss'])
            val_loss    = _ensure_array(metrics['val/loss'])
            train_acc   = _ensure_array(metrics['train/acc'])
            val_acc     = _ensure_array(metrics['val/acc'])

            if train_loss.shape != train_steps.shape or train_acc.shape != train_steps.shape:
                raise ValueError("Training metrics and steps lengths do not match")
            if val_loss.shape != val_steps.shape or val_acc.shape != val_steps.shape:
                raise ValueError("Validation metrics and steps lengths do not match")

            if train_steps_ref is None:
                train_steps_ref = train_steps
            elif train_steps.shape != train_steps_ref.shape or not np.array_equal(train_steps, train_steps_ref):
                raise ValueError("Inconsistent training step counts across seeds")

            if val_steps_ref is None:
                val_steps_ref = val_steps
            elif val_steps.shape != val_steps_ref.shape or not np.array_equal(val_steps, val_steps_ref):
                raise ValueError("Inconsistent validation step counts across seeds")

            losses_tr.append(train_loss)
            losses_val.append(val_loss)
            accs_tr.append(train_acc)
            accs_val.append(val_acc)

        if (
            train_steps_ref is None or val_steps_ref is None
            or train_steps_ref.size == 0 or val_steps_ref.size == 0
        ):
            continue

        Ltr = np.vstack(losses_tr);   Lval = np.vstack(losses_val)
        Atr = np.vstack(accs_tr);     AVal = np.vstack(accs_val)

        line_with_std(ax_train_loss, train_steps_ref, Ltr.mean(0),   Ltr.std(0),   label=name)
        line_with_std(ax_val_loss,   val_steps_ref,   Lval.mean(0),  Lval.std(0),  label=name)
        line_with_std(ax_train_acc,  train_steps_ref, Atr.mean(0),   Atr.std(0),   label=name)
        line_with_std(ax_val_acc,    val_steps_ref,   AVal.mean(0),  AVal.std(0),  label=name)

    # Decide which epochs to show (max 8 ticks)
    first_model   = list(next(iter(trained_models.values())).values())[0]
    epoch_steps   = first_model.metrics['epoch']
    n_epochs      = len(epoch_steps)
    epoch_nums    = list(range(1, n_epochs + 1))
    max_ticks     = 8
    if n_epochs > max_ticks:
        stride = math.ceil(n_epochs / max_ticks)
        idxs   = list(range(0, n_epochs, stride))
        if idxs[-1] != n_epochs - 1:
            idxs.append(n_epochs - 1)
        bottom_steps  = [epoch_steps[i] for i in idxs]
        bottom_labels = [epoch_nums[i]  for i in idxs]
    else:
        bottom_steps  = epoch_steps
        bottom_labels = epoch_nums

    # Now configure each subplot according to your rules:
    for subax, (row, col) in zip(
        (ax_train_loss, ax_val_loss, ax_train_acc, ax_val_acc),
        [(0, 0),       (0, 1),       (1, 0),        (1, 1)]
    ):
        # — Epochs on the bottom
        subax.set_xticks(bottom_steps)
        subax.set_xticklabels(bottom_labels)
        # only bottom row shows the "Epoch" label
        subax.set_xlabel('Epoch' if row == 1 else '')

        # — Steps on the top
        ax_top = subax.twiny()
        ax_top.set_xlim(subax.get_xlim())
        ax_top.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        # bottom row: no step-label or tick-labels
        if row == 1:
            ax_top.set_xlabel('')
            ax_top.set_xticklabels([])
        # top row: no step-label, but keep tick-labels
        else:
            ax_top.set_xlabel('')
        ax_top.xaxis.set_label_position('top')
        ax_top.xaxis.tick_top()

        # — no y-labels on right column
        if col == 1:
            subax.set_ylabel('')

        subax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    LoggingService().save_figure(fig, "models_together")

def line_with_std(ax, x, mean, std, label=None, **plot_kwargs):
    """Plot a line with a shaded standard deviation band."""
    ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

@rank_zero_only
def plot_training_curves(model_name, seed_dict, fig_dir=None):
    """Plot averaged loss, accuracy and weights norm over multiple seeds,
       with dashed step-grid and solid epoch-grid, ensuring epoch ticks
       start at 1 and end at the final epoch."""
    seeds = sorted(seed_dict.keys())
    first_model = seed_dict[seeds[0]]

    # Collect per-seed metrics
    train_loss_seeds, val_loss_seeds = [], []
    train_acc_seeds, val_acc_seeds = [], []
    train_w2_seeds = []
    step_lists = []
    for seed in seeds:
        df = seed_dict[seed].metrics.format_metrics()
        step_lists.append(df["step"].to_numpy())
        train_loss_seeds.append(df["train_loss"].to_numpy())
        val_loss_seeds.append(  df["val_loss"].to_numpy())
        train_acc_seeds.append(df["train_acc"].to_numpy())
        val_acc_seeds.append(  df["val_acc"].to_numpy())
        train_w2_seeds.append( df["train_weights_l2"].to_numpy())

    # Ensure consistent steps
    steps = step_lists[0]
    if any(len(sl) != len(steps) for sl in step_lists):
        raise ValueError("Inconsistent step counts across seeds")

    # Stack + mean/std
    tl = np.vstack(train_loss_seeds);   tlm, tls = tl.mean(0), tl.std(0)
    vl = np.vstack(val_loss_seeds);     vlm, vls = vl.mean(0), vl.std(0)
    ta = np.vstack(train_acc_seeds);    tam, tas = ta.mean(0), ta.std(0)
    va = np.vstack(val_acc_seeds);      vam, vas = va.mean(0), va.std(0)
    tw = np.vstack(train_w2_seeds);     twm, tws = tw.mean(0), tw.std(0)

    # Make subplots
    fig, (ax_loss, ax_acc, ax_w2) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # LOSS
    line_with_std(ax_loss, steps, tlm, tls, label="Train Loss")
    line_with_std(ax_loss, steps, vlm, vls, label="Val Loss")
    ax_loss.set_ylabel("Loss (log)")
    ax_loss.set_yscale('log')
    ax_loss.legend()
    ax_loss.grid(axis='x', linestyle='--', linewidth=0.5)

    # ACCURACY
    line_with_std(ax_acc, steps, tam, tas, label="Train Acc")
    line_with_std(ax_acc, steps, vam, vas, label="Val Acc")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.legend()
    ax_acc.grid(axis='x', linestyle='--', linewidth=0.5)

    # WEIGHTS L2
    line_with_std(ax_w2, steps, twm, tws, label="Weights L2")
    ax_w2.set_ylabel("Weights L2 Norm")
    ax_w2.set_xlabel("Step")
    ax_w2.legend()
    ax_w2.grid(axis='x', linestyle='--', linewidth=0.5)

    # Prepare epoch lookup
    epoch_steps = np.array(first_model.metrics["epoch"])
    epoch_numbers = np.arange(1, len(epoch_steps) + 1)

    # Add epoch axis on top:
    for idx, ax in enumerate((ax_loss, ax_acc, ax_w2)):
        ax_epoch = ax.twiny()
        ax_epoch.set_xlim(ax.get_xlim())

        # pick up to 10 “nice” step-positions
        ax_epoch.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

        # map each chosen step → its epoch number
        def epoch_fmt(x, pos):
            idx = np.searchsorted(epoch_steps, x, side='right') - 1
            if 0 <= idx < len(epoch_numbers):
                return str(epoch_numbers[idx])
            return ''

        ax_epoch.xaxis.set_major_formatter(FuncFormatter(epoch_fmt))
        ax_epoch.xaxis.tick_top()

        # ensure endpoints (epoch 1 and final) are included in ticks
        ticks = ax_epoch.get_xticks()
        # keep only those within the first and last epoch step
        ticks = ticks[(ticks >= epoch_steps[0]) & (ticks <= epoch_steps[-1])]
        ticks = np.unique(np.concatenate([ticks, [epoch_steps[0], epoch_steps[-1]]]))
        ax_epoch.set_xticks(ticks)

        if idx == 0:
            ax_epoch.set_xlabel("Epoch")
            ax_epoch.xaxis.set_label_position('top')
        else:
            ax_epoch.set_xticklabels([])

        # solid grid on epochs
        ax_epoch.grid(axis='x', linestyle='-', linewidth=0.8)

    # Final layout & save
    fig.suptitle(f"Training Curves — {model_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    LoggingService().save_figure(fig, f"training_{model_name}", fig_dir)

def plot_gradient_flow(grad_dict, title):
    layers = [k.replace('.weight', '').replace('.bias', '') for k in grad_dict.keys()]
    num_ticks = len(next(iter(grad_dict.values())))
    data = np.array([grad_dict[k] for k in grad_dict.keys()])

    means = [np.mean(grad_dict[k]) for k in grad_dict.keys()]
    stds = [np.std(grad_dict[k]) for k in grad_dict.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(16, len(layers) * 0.4 + 3), gridspec_kw={'width_ratios': [70, 30]})

    ax0 = axes[0]
    sns.heatmap(data, ax=ax0, xticklabels=range(1, num_ticks + 1), yticklabels=layers, cmap='viridis', cbar=False)
    ax0.set_title('Gradient Flow Heatmap')
    ax0.yaxis.tick_left()
    ax0.yaxis.set_label_position("left")

    ax1 = axes[1]
    y_positions = np.arange(len(layers))
    ax1.barh(y_positions, means, xerr=stds, capsize=5, color='skyblue')
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlabel('Gradient Norm')
    ax1.set_title('Layer-wise Gradient Flow')

    bar_height = 0.8
    ax1.set_ylim((len(layers) - 1) + bar_height / 2, 0 - bar_height / 2)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes

@rank_zero_only
def generate_gradient_flow_plots(model, fig_dir=None):
    if not hasattr(model.metrics._raw, 'grad_l2'):
        log.info("No gradient data found in model metrics. Skipping gradient flow plots.")
        return
    grad_norms = model.metrics._raw['grad_l2']

    weight_grads = {k: v for k, v in grad_norms.items() if 'weight' in k and 'bias' not in k and 'bn' not in k and 'norm' not in k}
    if weight_grads == {}: raise ValueError(f"No weight gradients found.")
    weight_title = f"Gradient Flow of Weights"
    weight_fig, weight_axes = plot_gradient_flow(weight_grads, weight_title)
    LoggingService().save_figure(weight_fig, "gradient_flow_weight.png", fig_dir)

    bias_grads = {k: v for k, v in grad_norms.items() if 'bias' in k and 'bn' not in k and 'norm' not in k}
    if bias_grads == {}:
        log.info("No bias gradients found.")
        return
    bias_title = f"Gradient Flow of Biases"
    bias_fig, bias_axes = plot_gradient_flow(bias_grads, bias_title)
    LoggingService().save_figure(bias_fig, "gradient_flow_bias", fig_dir)