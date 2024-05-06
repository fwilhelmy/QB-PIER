# Refactoring the PyTorch training code into a more modular and efficient form
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import os
import pickle
import logging
from torch.nn import Module
from models.HQNN_Quanv import HQNN_Quanv

# Importing a hypothetical models package
# from models import LeNet, HQNN_Quanv, HQNN_Parallel
# HQNN_Quanv(in_channels=1, classes=len(trainData.dataset.classes))

class ModelTrainer():
    def __init__(self, model, optimizer=None, lossFn=None, dataset=None, device=None, init_lr=1e-3, batch_size=64, epochs=10, train_split=0.7, val_split=0.3):
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_split = train_split
        self.val_split = val_split
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset if dataset else MNIST(root="../data", train=True, download=True, transform=ToTensor())
        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=self.init_lr)
        self.lossFn = lossFn if lossFn else nn.CrossEntropyLoss()

        # Calculate the train/validation split
        trainSamples = int((len(dataset)) * self.train_split)
        valSamples = int((len(dataset)) * self.val_split)
        unusedSamples = len(dataset) - trainSamples - valSamples

        # Create Subset objects for train, validation, and test
        trainData, valData, _ = random_split(
            dataset,
            [trainSamples, valSamples, unusedSamples],
        )

        # Initialize the train, validation, and test data loaders
        self.trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=self.batch_size)
        self.valDataLoader = DataLoader(valData, batch_size=self.batch_size)

        # Calculate steps per epoch for training and validation set
        self.trainSteps = len(self.trainDataLoader.dataset) // self.batch_size
        self.valSteps = len(self.valDataLoader.dataset) // self.batch_size

    def train(self):
        # Initialize a dictionary to store training history
        H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print("[TRAIN] Training the model...")

        startTime = time.time()  # To measure how long training is going to take

        # loop over our epochs
        for e in range(0, self.epochs):
            # Set the model in training mode
            self.model.train()
            # Initialize the total training/validation loss/correct predictions
            totalTrainLoss, totalValLoss, trainCorrect, valCorrect = 0, 0, 0, 0

            # Training
            for x, y in self.trainDataLoader:
                # Send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # Perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = self.lossFn(pred, y)
                # Zero out the gradients, perform the backpropagation step, and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Sum the loss to the total training loss so far
                totalTrainLoss += loss
                # Calculate the number of correct predictions
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Validation
            with torch.no_grad():
                # Set the model in evaluation mode
                self.model.eval()
                # Loop over the validation set
                for x, y in self.valDataLoader:
                    # Send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # Make the predictions and calculate the validation loss
                    pred = self.model(x)
                    totalValLoss += self.lossFn(pred, y)
                    # Calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / self.trainSteps
            avgValLoss = totalValLoss / self.valSteps
            # Calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(self.trainDataLoader.dataset)
            valCorrect = valCorrect / len(self.valDataLoader.dataset)

            # Update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_acc"].append(valCorrect)

            # Print the model training and validation information
            print(
                "[TRAIN] EPOCH: {}/{} | Train loss: {:0.6f} | Train acc: {:0.4f} | Val loss: {:0.6f} | Val acc: {:0.4f} "
                "| Time elapsed: {:0.2f} ".format(
                    e + 1,
                    self.epochs,
                    avgTrainLoss,
                    trainCorrect,
                    avgValLoss,
                    valCorrect,
                    time.time() - startTime,
                )
            )
        # Finish measuring how long training took
        endTime = time.time()
        print("[TRAIN] Finished training the model...")
        print("[TRAIN] Total time taken to train the model: {:.2f}s".format(endTime - startTime))

        return H
    
    def evaluate(self, data):
        testDataset = MNIST(root="../data", train=False, download=True, transform=ToTensor())
        testDataLoader = DataLoader(testDataset, batch_size=self.batch_size)

        correct = 0
        total = 0

        print("[END] Evaluating the model...")
        # We can now evaluate the network on the test set
        with torch.no_grad():  # Turn off autograd for testing evaluation
            self.model.eval()  # Set the model in evaluation mode
            preds = []  # Initialize a list to store our predictions
            # Evaluation with test dataset
            for x, y in testDataLoader:
                # Send the input to the device
                x = x.to(self.device)
                # Make the predictions and add them to the list
                pred = self.model(x)
                preds.extend(pred.argmax(axis=1).cpu().numpy())

        accuracy = 100 * correct / total

        # Generate a classification report
        # print(classification_report([y for _, y in testData], np.array(preds), target_names=testData.dataset.classes))

        logging.info(f'Accuracy of the model on the test images: {accuracy:.2f}%')
        return accuracy

# This refactored code structure improves modularity and maintainability, making it easier to modify components like the model, dataset, or training parameters.

# Usage example (commented out):
# trainer = TrainModel()
# model = LeNet(numChannels=1, classes=len(trainer.train_loader.dataset.classes))
# history = trainer.train(model)
