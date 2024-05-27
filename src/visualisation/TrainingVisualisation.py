import matplotlib.pyplot as plt
import os
import pennylane as qml

def plot_training_result(path, model_name, model_trainer, training_results):
    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
    fig.suptitle(f"Training Loss and Accuracy with {model_name}")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.2)
    ax1.plot(training_results["train_acc"], label="Training")
    ax1.plot(training_results["val_acc"], label="Validation")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper right")
    ax2.plot(training_results["train_loss"], label="Training")
    ax2.plot(training_results["val_loss"], label="Validation")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch #")

    # Add hyperparameters to the plot
    fig.text(
        0.01,
        0.01,
        "Epochs: {}, Learning Rate: {}, Batch Size: {}, Train Size: {}, Val Size: {}, Time: {:.2f}s".format(
            model_trainer.epochs,
            model_trainer.init_lr,
            model_trainer.batch_size,
            training_results["train_samples"],
            training_results["val_samples"],
            training_results["time_elapsed"],
        ),
        fontsize=10,
    )

    # Saving the results to the ./results directory
    plt.savefig(f"{path}/train_plot.png")

def plot_models_together(path, training_results):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # Plotting training loss
    ax[0].set_title('Training Loss per Epoch')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    # Plotting validation accuracy
    ax[1].set_title('Validation Accuracy per Epoch')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (%)')

    for model_name, history in training_results.items():
        epochs = list(range(1, len(history['train_loss']) + 1))
        ax[0].plot(epochs, history['train_loss'], label=f'{model_name}')
        ax[1].plot(epochs, history['val_acc'], label=f'{model_name}')

    ax[0].legend()
    ax[1].legend()

    # Saving the results to the ./results directory
    plt.savefig(f"{path}/full_train.png")