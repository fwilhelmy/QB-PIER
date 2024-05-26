import matplotlib.pyplot as plt
import numpy as np
import os

# Function to plot adversarial examples
def plot_adversarial_examples(rootdir, model_name, epsilons, examples):
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

    # Create a directory to store the results
    path = f"results/{rootdir}/{model_name}"
    os.mkdir(path)

    # Saving the results to the ./results directory
    plt.savefig(path + "/adversarial_examples.png")

# Function to plot accuracy vs epsilon
def plot_accuracy_vs_epsilon(rootdir, model_name, epsilons, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")

    # Create a directory to store the results if it doesn't exist
    path = f"results/{rootdir}/{model_name}"
    if not os.path.exists(path):
        os.mkdir(path)

    # Saving the results to the ./results directory
    plt.savefig(path + "/accuracy_vs_epsilon.png")
