import matplotlib.pyplot as plt
import numpy as np
from LoggingService import LoggingService

# Function to plot adversarial examples
# TODO ADD TITLE WITH MODEL NAME
def plot_adversarial_examples(model_name, epsilons, examples):
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
    LoggingService().save_figure(plt)

# Function to plot accuracy vs epsilon
def plot_accuracy_vs_epsilon(model_name, attack_name, epsilons, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title(f"{attack_name} attack on {model_name} model (Accuracy vs Epsilon)" )
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")

    # Saving the results
    LoggingService().save_figure(plt)

def visualize_attacks_results(model_name, adv_exemples, num_images=5):
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
    
    LoggingService().save_figure(plt)

def plot_attack_effect_on_accuracy(models, epsilons, accuracies):
    plt.figure(figsize=(10, 6))

    for model in models:
        plt.plot(epsilons, accuracies[model.model_name], label=model.model_name, marker='o')

    plt.xlabel('Attack Strength (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of Adversarial Attacks on Model Accuracy', fontsize=14)
    plt.xticks(epsilons, [f"{e:.2f}" for e in epsilons], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models', fontsize=10)
    plt.tight_layout()

    LoggingService().save_figure(plt)