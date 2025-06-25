import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.logging.LoggingService import LoggingService

def plot_training_result(model, train_len, val_len):
    results = model.all_metrics

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    fig.suptitle(f"Training Loss and Accuracy with {model.model_name}")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.2)
    ax1.plot(results["train_acc"], label="Training")
    ax1.plot(results["val_acc"], label="Validation")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper right")
    ax2.plot(results["train_loss"], label="Training")
    ax2.plot(results["val_loss"], label="Validation")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch #")

    # Add hyperparameters to the plot
    fig.text(
        0.01,
        0.01,
        "Model: {}, Devices: {}, max_steps: {}, Learning Rate: {}\nSamples (T|V): ({}|{}), Accuracy (T|V): ({}|{})".format(
            type(model).__name__,
            model.hparams["devices"],
            model.hparams["max_steps"],
            model.hparams["learning_rate"],
            train_len,
            val_len,
            results["train_acc"][-1],
            results["val_acc"][-1],
        ),
        fontsize=10,
    )

    # TODO Extract the time taken to train the model
    # TODO Extract number of training and validation samples

    # Save the results
    LoggingService().save_figure(fig, "training_result", model)

def plot_models_together(models):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # Plotting training loss
    ax[0].set_title('Training Loss per Epoch')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    # Plotting validation accuracy
    ax[1].set_title('Validation Accuracy per Epoch')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (%)')

    for model in models:
        model_name = model.model_name
        history = model.history
        epochs = list(range(1, len(history['train_loss']) + 1))
        ax[0].plot(epochs, history['train_loss'], label=f'{model_name}')
        ax[1].plot(epochs, history['val_acc'], label=f'{model_name}', linestyle='--')

        ax[0].legend()
        ax[1].legend()

    # Saving the results
    LoggingService().save_figure(fig, "models_together")


def line_with_std(ax, x, mean, std, label=None, **plot_kwargs):
    """Plot a line with a shaded standard deviation band."""
    ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def plot_training_curves(trained_models, title_prefix="training"):
    """Plot averaged loss, accuracy and weights norm over multiple seeds."""
    for model_name, seed_dict in trained_models.items():
        seeds = sorted(seed_dict.keys())

        train_loss_seeds = []
        val_loss_seeds = []
        train_acc_seeds = []
        val_acc_seeds = []
        train_weights_l2_seeds = []
        step_lists = []
        for seed in seeds:
            df = seed_dict[seed].metrics.get_metrics()
            step_lists.append(df["step"].to_numpy())
            train_loss_seeds.append(df["train_loss"].to_numpy())
            val_loss_seeds.append(df["val_loss"].to_numpy())
            train_acc_seeds.append(df["train_acc"].to_numpy())
            val_acc_seeds.append(df["val_acc"].to_numpy())
            train_weights_l2_seeds.append(df["train_weights_l2"].to_numpy())

        steps = step_lists[0]
        if not all(len(sl) == len(steps) for sl in step_lists):
            raise ValueError("Inconsistent step counts across seeds")

        tl = np.vstack(train_loss_seeds)
        vl = np.vstack(val_loss_seeds)
        ta = np.vstack(train_acc_seeds)
        va = np.vstack(val_acc_seeds)
        tw = np.vstack(train_weights_l2_seeds)

        tl_mean, tl_std = tl.mean(0), tl.std(0)
        vl_mean, vl_std = vl.mean(0), vl.std(0)
        ta_mean, ta_std = ta.mean(0), ta.std(0)
        va_mean, va_std = va.mean(0), va.std(0)
        tw_mean, tw_std = tw.mean(0), tw.std(0)

        fig, (ax_loss, ax_acc, ax_weights) = plt.subplots(1, 3, figsize=(24, 6))

        line_with_std(ax_loss, steps, tl_mean, tl_std, label="Train Loss")
        line_with_std(ax_loss, steps, vl_mean, vl_std, label="Val Loss")
        ax_loss.set_title(f"{model_name} Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", linewidth=0.5)
        ax_loss.legend()

        line_with_std(ax_acc, steps, ta_mean, ta_std, label="Train Acc")
        line_with_std(ax_acc, steps, va_mean, va_std, label="Val Acc")
        ax_acc.set_title(f"{model_name} Accuracy")
        ax_acc.set_xlabel("Step")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.0)
        ax_acc.grid(True, linestyle="--", linewidth=0.5)
        ax_acc.legend()

        line_with_std(ax_weights, steps, tw_mean, tw_std, label="Train Weights L2")
        ax_weights.set_title(f"{model_name} Weights L2")
        ax_weights.set_xlabel("Step")
        ax_weights.set_ylabel("Weights L2 Norm")
        ax_weights.grid(True, linestyle="--", linewidth=0.5)
        ax_weights.legend()

        fig.suptitle(f"{title_prefix.capitalize()} Curves — {model_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        LoggingService().save_figure(fig, f"{title_prefix}_{model_name}")


def plot_gradient_flow(grad_dict, title, save_path):
    layers = [k.replace('.weight', '').replace('.bias', '') for k in grad_dict.keys()]
    num_epochs = len(next(iter(grad_dict.values())))
    data = np.array([grad_dict[k] for k in grad_dict.keys()])

    means = [np.mean(grad_dict[k]) for k in grad_dict.keys()]
    stds = [np.std(grad_dict[k]) for k in grad_dict.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(16, len(layers) * 0.4 + 3), gridspec_kw={'width_ratios': [70, 30]})

    ax0 = axes[0]
    sns.heatmap(data, ax=ax0, xticklabels=range(1, num_epochs + 1), yticklabels=layers, cmap='viridis', cbar=False)
    ax0.set_xlabel('Epoch')
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
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def generate_gradient_flow_plots(models):
    for md in models.values():
        model = md[0]
        grad_norms = model.metrics._raw['grad_l2']

        weight_grads = {k: v for k, v in grad_norms.items() if 'weight' in k and 'bias' not in k and 'bn' not in k and 'norm' not in k}
        bias_grads = {k: v for k, v in grad_norms.items() if 'bias' in k and 'bn' not in k and 'norm' not in k}

        weight_title = f"Gradient Flow of Weights for {model.model_name}"
        weight_save = os.path.join(model.log_dir, "gradient_flow_weight.png")
        plot_gradient_flow(weight_grads, weight_title, weight_save)

        bias_title = f"Gradient Flow of Bias for {model.model_name}"
        bias_save = os.path.join(model.log_dir, "gradient_flow_bias.png")
        plot_gradient_flow(bias_grads, bias_title, bias_save)


def plot_gradient_flow_simple(grad_norms: dict, title: str, save_path: str):
    layers, norms = zip(*sorted(grad_norms.items(), key=lambda kv: kv[0]))
    plt.figure(figsize=(8, 4))
    for name, vals in zip(layers, norms):
        plt.plot(vals, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(title)
    plt.legend(fontsize=6, ncol=2, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_gradient_flow_plots_from_dirs(list_of_dirs: list, legend_names: list):
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        if not os.path.exists(json_path):
            raise AssertionError(f"No results.json in {logdir}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        norms = data['train_grad_norms']

        weights = {k: v for k, v in norms.items() if 'weight' in k and 'bias' not in k}
        biases = {k: v for k, v in norms.items() if 'bias' in k}

        w_path = os.path.join(logdir, f"gradient_flow_{name}_weights.png")
        b_path = os.path.join(logdir, f"gradient_flow_{name}_biases.png")
        plot_gradient_flow_simple(weights, f"Gradient Flow—Weights ({name})", w_path)
        plot_gradient_flow_simple(biases, f"Gradient Flow—Biases ({name})", b_path)

