import matplotlib.pyplot as plt
from LoggingService import LoggingService

def plot_training_result(model, train_len, val_len):
    results = model.history

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
        "Model: {}, Devices: {}, Epochs: {}, Learning Rate: {}, Samples (T|V): ({}|{}), Accuracy (T|V): ({}|{})".format(
            type(model).__name__,
            model.devices,
            model.max_epochs,
            model.learning_rate,
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
    LoggingService().save_figure(plt, "training_result", model)

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
        LoggingService().save_figure(plt, "models_together")
