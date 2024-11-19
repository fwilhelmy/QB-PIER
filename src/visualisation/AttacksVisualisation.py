import matplotlib.pyplot as plt
import numpy as np
from VisualisationUtils import save_figure

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
    save_figure(plt, model_name)
    plt.close()

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
    save_figure(plt, model_name)
    plt.close()