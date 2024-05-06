import logging
from models.HQNN_Quanv import HQNN_Quanv
from ModelTrainer import ModelTrainer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def register_model(model_name, model_trainer):
    """Register a model with its configuration."""
    if model_name not in models:
        models[model_name] = {'trainer': model_trainer}
        logging.info(f"Model '{model_name}' registered.")
    else:
        logging.warning(f"Model '{model_name}' already registered.")
    return models

def train_all(models, train_dataset, val_dataset):
    """Train all registered models."""
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Training model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.train(dataset, train_split, val_split)

    return results

def evaluate_all(models, test_dataset):
    """Evaluate all registered models."""
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Evaluating model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.evaluate()
    return results

def plot_training_result(rootdir, model_name, model_trainer, training_results):
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
            training_results["elapsed_time"],
        ),
        fontsize=10,
    )

    # Create a directory to store the results
    path = f"results/{rootdir}/{model_name}"
    os.mkdir(path)

    # Saving the results to the ./results directory
    plt.savefig(path + "/train_plot.png")

# Create a directory to store the results
rootdir = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
os.mkdir(f"results/{rootdir}")

# Load the dataset for the experiments
split = 0.7

train_samples = 128
val_samples = 128
test_samples = 128

data = MNIST(root="../data", train=True, download=True, transform=ToTensor())

if debug:
    
dataset = MNIST(root="../data", train=True, download=True, transform=ToTensor())
testDataset = MNIST(root="../data", train=False, download=True, transform=ToTensor())

model = HQNN_Quanv(in_channels=1, classes=len(dataset.classes))
models = {}
models = register_model('Model1_Test_64', ModelTrainer(
    model))
models = register_model('Model2_Test_70', ModelTrainer(
    model,
    batch_size=70))

training_results = train_all(models, dataset, train_split=0.0083333334, val_split=0.008333334)
evaluation_results = evaluate_all(models, testDataset)

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
plt.savefig(f"results/{rootdir}/full_train.png")
