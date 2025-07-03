import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metrics.LoggingService import LoggingService

def plot_models_together(trained_models):
    trained_first_model = list(next(iter(trained_models.values())).values())[0]
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')
    
    # Define subplots
    ax_train_loss = ax[0, 0]
    ax_val_loss = ax[0, 1]
    ax_train_acc = ax[1, 0]
    ax_val_acc = ax[1, 1]
    
    # Set titles and labels
    ax_train_loss.set_title('Training')
    ax_val_loss.set_title('Validation')
    ax_train_loss.set_ylabel('Loss')
    ax_train_acc.set_ylabel('Accuracy (%)')
    ax_train_acc.set_xlabel('Step')
    ax_val_acc.set_xlabel('Step')
    
    # Iterate over each model. Each model contains a list of seeds in model.seeds.
    for model_name, seed_dict in trained_models.items():
        first_model = list(seed_dict.values())[0]

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        
        # Aggregate metrics for each seed in the model
        for model_seed in seed_dict.values():
            metrics = model_seed.metrics.format_metrics()
            train_loss_list.append(np.array(metrics['train_loss']))
            val_loss_list.append(np.array(metrics['val_loss']))
            train_acc_list.append(np.array(metrics['train_acc']))
            val_acc_list.append(np.array(metrics['val_acc']))
        
        # Convert lists to arrays for vectorised operations
        train_loss_arr = np.vstack(train_loss_list)
        val_loss_arr = np.vstack(val_loss_list)
        train_acc_arr = np.vstack(train_acc_list)
        val_acc_arr = np.vstack(val_acc_list)
        
        # Calculate mean and standard deviation across seeds
        train_loss_mean = train_loss_arr.mean(axis=0)
        train_loss_std = train_loss_arr.std(axis=0)
        val_loss_mean = val_loss_arr.mean(axis=0)
        val_loss_std = val_loss_arr.std(axis=0)
        train_acc_mean = train_acc_arr.mean(axis=0)
        train_acc_std = train_acc_arr.std(axis=0)
        val_acc_mean = val_acc_arr.mean(axis=0)
        val_acc_std = val_acc_arr.std(axis=0)
        
        # Create step indices
        steps = np.array(first_model.metrics.format_metrics()['step'])
        
        # Plot curves with shaded std regions using the helper function line_with_std
        line_with_std(ax_train_loss, steps, train_loss_mean, train_loss_std, label=model_name)
        line_with_std(ax_val_loss, steps, val_loss_mean, val_loss_std, label=model_name)
        line_with_std(ax_train_acc, steps, train_acc_mean, train_acc_std, label=model_name)
        line_with_std(ax_val_acc, steps, val_acc_mean, val_acc_std, label=model_name)

    epoch_end_steps = trained_first_model.metrics["epoch"]
    for epoch_step in epoch_end_steps:
        ax_train_loss.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)
        ax_val_loss.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)
        ax_train_acc.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)
        ax_val_acc.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)
    
    # Add legends to each subplot
    ax_train_loss.legend()
    ax_val_loss.legend()
    ax_train_acc.legend()
    ax_val_acc.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Training and Validation Curves of for models", fontsize=16)
    
    # Saving the results
    LoggingService().save_figure(fig, "models_together")


def line_with_std(ax, x, mean, std, label=None, **plot_kwargs):
    """Plot a line with a shaded standard deviation band."""
    ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def plot_training_curves(model_name, seed_dict):
    """Plot averaged loss, accuracy and weights norm over multiple seeds."""
    seeds = sorted(seed_dict.keys())

    first_model = seed_dict[seeds[0]]
    model_name = first_model.model_name
    fig_dir = f"{first_model.__class__.__name__}/{model_name}"

    train_loss_seeds = []
    val_loss_seeds = []
    train_acc_seeds = []
    val_acc_seeds = []
    train_weights_l2_seeds = []
    step_lists = []
    for seed in seeds:
        df = seed_dict[seed].metrics.format_metrics()
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

    fig, (ax_loss, ax_acc, ax_weights) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    line_with_std(ax_loss, steps, tl_mean, tl_std, label="Train Loss")
    line_with_std(ax_loss, steps, vl_mean, vl_std, label="Val Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", linewidth=0.5)
    ax_loss.legend()

    line_with_std(ax_acc, steps, ta_mean, ta_std, label="Train Acc")
    line_with_std(ax_acc, steps, va_mean, va_std, label="Val Acc")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(True, linestyle="--", linewidth=0.5)
    ax_acc.legend()

    line_with_std(ax_weights, steps, tw_mean, tw_std, label="Train Weights L2")
    ax_weights.set_xlabel("Step")
    ax_weights.set_ylabel("Weights L2 Norm")
    ax_weights.grid(True, linestyle="--", linewidth=0.5)
    ax_weights.legend()

    # Add vertical lines marking the end of each epoch
    # The epoch endpoints are retrieved from the model metrics
    epoch_end_steps = first_model.metrics["epoch"]
    for epoch_step in epoch_end_steps:
        ax_loss.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)
        ax_acc.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)
        ax_weights.axvline(x=epoch_step, color="gray", linestyle="--", alpha=0.8)

    fig.suptitle(f"Training Curves — {model_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    LoggingService().save_figure(fig, f"training_{model_name}", fig_dir)

def plot_gradient_flow(grad_dict, title):
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
    return fig, axes

def generate_gradient_flow_plots(model):
    grad_norms = model.metrics._raw['grad_l2']
    fig_dir = f"{model.__class__.__name__}/{model.model_name}"

    weight_grads = {k: v for k, v in grad_norms.items() if 'weight' in k and 'bias' not in k and 'bn' not in k and 'norm' not in k}
    if weight_grads == {}: raise ValueError(f"No weight gradients found in {model.model_name}.")
    weight_title = f"Gradient Flow of Weights for {model.model_name}"
    weight_fig, weight_axes = plot_gradient_flow(weight_grads, weight_title)
    LoggingService().save_figure(weight_fig, "gradient_flow_weight.png", fig_dir)

    bias_grads = {k: v for k, v in grad_norms.items() if 'bias' in k and 'bn' not in k and 'norm' not in k}
    if bias_grads == {}: 
        print(f"No bias gradients found in {model.model_name}.") 
        return
    bias_title = f"Gradient Flow of Bias for {model.model_name}"
    bias_fig, bias_axes = plot_gradient_flow(bias_grads, bias_title)
    LoggingService().save_figure(bias_fig, "gradient_flow_bias", fig_dir)