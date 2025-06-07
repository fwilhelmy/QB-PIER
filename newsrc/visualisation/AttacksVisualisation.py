import os
import sys
# ─── Make sure Python can find `LoggingService.py` in the parent directory ───
# (i.e. <repo_root>/newsrc/LoggingService.py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ─────────────────────────────────────────────────────────────────────────────


import matplotlib.pyplot as plt
import numpy as np
from LoggingService import LoggingService

# Function to plot adversarial examples
# TODO ADD TITLE WITH MODEL NAME
def plot_adversarial_examples(model, epsilons, examples):
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()

    # Saving the results
    LoggingService().save_figure(plt, "adversarial_examples", model)

# Function to plot accuracy vs epsilon
def plot_accuracy_vs_epsilon(model, attack_name, epsilons, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title(f"{attack_name} attack on {model.model_name} model (Accuracy vs Epsilon)" )
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")

    # Saving the results
    LoggingService().save_figure(plt, "accuracy_vs_epsilon", model.model_name)

def visualize_attacks_results(model, adv_exemples, num_images=5):
    # Sort the epsilon levels for consistent ordering
    epsilons = sorted(adv_exemples.keys())
    num_epsilons = len(epsilons)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_epsilons, num_images, figsize=(num_images * 2, num_epsilons * 2))
    fig.suptitle('Adversarial Attacks Visualization', fontsize=16)

    # If there's only one epsilon level or one image, axes may not be a 2D array
    if num_epsilons == 1:
        axes = [axes]
    else:
        axes = axes

    for row, eps in enumerate(epsilons):
        perturbed_images = adv_exemples[eps]
        for col in range(num_images):
            ax = axes[row][col] if num_epsilons > 1 else axes[col]
            
            # Ensure the index does not exceed the available images
            if col < len(perturbed_images):
                img = perturbed_images[col]
                # If the image has more than 2 dimensions, assume it's multi-channel and take the first channel
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    img = img.squeeze(0)
                elif img.ndim == 2:
                    img = img  # Grayscale image

                ax.imshow(img, cmap='gray')
            else:
                ax.axis('off')  # Hide axes if no image is available

            ax.axis('off')  # Hide axis ticks

            # Set titles for the first row
            if row == 0:
                ax.set_title(f"Image {col+1}", fontsize=10)

        # Set y-axis labels for each row
        axes[row][0].set_ylabel(f"Epsilon: {eps}", rotation=0, labelpad=50, fontsize=12, va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    
    LoggingService().save_figure(plt, f"attacks_visualization{suffix}", model)

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # you can replace with pure matplotlib if you prefer

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, model_name):
    """
    y_true: 1D array of ground-truth labels
    y_pred: 1D array of predicted labels (e.g. after adversarial attack)
    class_names: list of class names for ticks
    save_path: directory where to save
    model_name: used in filename
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix under Attack")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_confusion_matrix.png")
    plt.close()

import matplotlib.pyplot as plt
import torch

def plot_perturbation_and_saliency(model, x_orig, epsilons, loss_fn, save_path, model_name):
    """
    x_orig: a single input tensor (1×C×H×W)
    epsilons: list of eps values to compute FGSM perturbations
    loss_fn: the same loss used in training (e.g. nn.CrossEntropyLoss())
    """
    model.eval()
    x_orig = x_orig.clone().detach().requires_grad_(True)
    outputs = model(x_orig)
    # pick the true label as the one with max logit
    y_label = outputs.argmax(dim=1)
    losses = loss_fn(outputs, y_label)
    losses.backward()
    grad_sign = x_orig.grad.data.sign().squeeze().cpu().numpy()  # (C×H×W)

    fig, axes = plt.subplots(2, len(epsilons), figsize=(3*len(epsilons), 6))
    for idx, eps in enumerate(epsilons):
        perturbation = eps * grad_sign
        x_adv = (x_orig.detach().cpu().squeeze().numpy() + perturbation).clip(0,1)

        # show adversarial example
        axes[0, idx].imshow(x_adv.squeeze(), cmap='gray')
        axes[0, idx].set_title(f"ε={eps:.2f}")
        axes[0, idx].axis('off')

        # show saliency map (absolute gradients)
        saliency = abs(grad_sign).max(axis=0)
        axes[1, idx].imshow(saliency, cmap='hot')
        axes[1, idx].set_title("Saliency (|∂L/∂x|)")
        axes[1, idx].axis('off')

    plt.suptitle(f"{model_name} Perturbations & Saliency")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(f"{save_path}/{model_name}_perturb_saliency.png")
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

def plot_loss_landscape(model, x0, y0, dir1, dir2, alphas, betas, loss_fn, save_path, model_name):
    """
    Visualize loss surface around (x0, y0) along two directions dir1, dir2.
    - x0: base input tensor flattened (D,)
    - y0: true label
    - dir1, dir2: two random normalized direction vectors (D,)
    - alphas, betas: 1D arrays of scaling factors
    """
    Z = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            x_pert = x0 + a*dir1 + b*dir2
            x_tensor = torch.from_numpy(x_pert).unsqueeze(0).to(next(model.parameters()).device)
            out = model(x_tensor)
            Z[i,j] = loss_fn(out, y0.unsqueeze(0)).item()

    A, B = np.meshgrid(betas, alphas)
    fig = plt.figure(figsize=(6,5))
    cp = plt.contourf(A, B, Z, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel("β"); plt.ylabel("α")
    plt.title(f"{model_name} Loss Landscape")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_loss_landscape.png")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_hybrid_attention(attentions, tokens, save_path, model_name):
    """
    attentions: list or array of shape (num_layers, num_heads, S, S)
    tokens: list of input tokens or token positions (length S)
    """
    n_layers, n_heads, S, _ = attentions.shape
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2*n_heads, 2*n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            att = attentions[l, h]
            im = ax.imshow(att, cmap='bone')
            if l == 0: ax.set_title(f"H{h+1}")
            if h == 0: ax.set_ylabel(f"L{l+1}")
            ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks([])
    plt.suptitle(f"{model_name} Attention Heads")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(f"{save_path}/{model_name}_attention_maps.png")
    plt.close()

import matplotlib.pyplot as plt

def plot_gradient_flow(grads_dict, title, save_file):
    """
    grads_dict: {layer_name: [list of gradient‐norms per batch or epoch]}
    """
    plt.figure(figsize=(8,6))
    for layer, norms in grads_dict.items():
        plt.plot(norms, label=layer)
    plt.hlines(0, 0, len(next(iter(grads_dict.values()))), colors='black', linewidth=0.5)
    plt.xlabel("Step"); plt.ylabel("Gradient Norm")
    plt.title(title)
    plt.legend(loc='upper right', fontsize=6)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

def plot_accuracy_with_uncertainty(model_name, epsilons, accuracies_matrix):
    """
    Plot mean accuracy vs. epsilon with a shaded standard-deviation band.
    - epsilons: list or 1D array of epsilon values.
    - accuracies_matrix: 2D array of shape (num_seeds, num_epsilons).
    """
    mean_acc = np.mean(accuracies_matrix, axis=0)
    std_acc = np.std(accuracies_matrix, axis=0)
    plt.figure(figsize=(6,4))
    plt.plot(epsilons, mean_acc, marker='o', label='Mean Accuracy')
    plt.fill_between(epsilons, mean_acc-std_acc, mean_acc+std_acc, color='blue', alpha=0.2, 
                     label='±1 Std Dev')
    plt.xlabel("Epsilon"); plt.ylabel("Accuracy")
    plt.title(f"{model_name}: Adversarial Accuracy vs. Epsilon (mean±std over seeds)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_attack_effect_on_accuracy(figstr, models, epsilons, accuracies):
    plt.figure(figsize=(10, 6))

    for model in models:
        plt.plot(accuracies[model.model_name].keys(), accuracies[model.model_name].values(), label=model.model_name, marker='o')

    plt.xlabel('Attack Strength (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of Adversarial Attacks on Model Accuracy', fontsize=14)
    plt.xticks(epsilons, [f"{e:.2f}" for e in epsilons], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models', fontsize=10)
    plt.tight_layout()

    LoggingService().save_figure(plt, figstr)