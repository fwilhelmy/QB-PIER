import logging
from ModelTrainer import ModelTrainer
from datetime import datetime
import os

from src.models.Classical.LeNet import LeNet
from models.Hybrid.HQNN_Quanv import HQNN_Quanv
from models.Hybrid.HQNN_Parallel import HQNN_Parallel

from Visualisation import plot_training_result, plot_models_together
from DataLoader import load_data

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

def batch_train(models, train_dataset, val_dataset):
    """Train all registered models."""
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Training model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.train(train_dataset, val_dataset)
        plot_training_result(rootdir, model_name, trainer, results[model_name])

    return results

def batch_evaluate(models, test_dataset):
    """Evaluate all registered models."""
    results = {}
    for model_name, model_info in models.items():
        logging.info(f"Evaluating model '{model_name}'...")
        trainer = model_info['trainer']
        results[model_name] = trainer.evaluate(test_dataset)
    return results

# Create a directory to store the results
rootdir = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
os.mkdir(f"results/{rootdir}")

trainDataset, valDataset, testDataset = load_data('FMNIST', num_samples=128)

# model = HQNN_Quanv(in_channels=1, classes=len(data.classes))
model = LeNet(numChannels=1, classes=len(data.classes))
models = {}
models = register_model('Model1_Test_64', ModelTrainer(
    model))
models = register_model('Model2_Test_70', ModelTrainer(
    model,
    batch_size=70))

training_results = batch_train(models, trainDataset, valDataset)
evaluation_results = batch_evaluate(models, testDataset)

plot_models_together(rootdir, training_results)