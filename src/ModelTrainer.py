import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import logging
import time
from torch import nn
from Utils import get_device

class ModelTrainer():
    def __init__(self, model, device=None, init_lr=1e-3, batch_size=64, epochs=10):
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = get_device() if device is None else device
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.init_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, dataset, train_split=0.7, val_split=0.3):
        H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Calculate the train/validation split
        H["train_samples"] = int((len(dataset)) * train_split)
        H["val_samples"] = int((len(dataset)) * val_split)
        unusedSamples = len(dataset) - H["train_samples"] - H["val_samples"]

        # Create Subset objects for train, validation, and test
        trainData, valData, _ = random_split(
            dataset,
            [H["train_samples"], H["val_samples"], unusedSamples],
        )

        # Initialize the train, validation, and test data loaders
        trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=self.batch_size)
        valDataLoader = DataLoader(valData, batch_size=self.batch_size)

        # Calculate steps per epoch for training and validation set
        trainSteps = len(trainDataLoader.dataset) // self.batch_size
        valSteps = len(valDataLoader.dataset) // self.batch_size

        start_time = time.time()
        for epoch in range(self.epochs):
            # Set the model in training mode
            self.model.train()
            # Initialize the total training/validation loss/correct predictions
            totalTrainLoss, totalValLoss, trainCorrect, valCorrect = 0, 0, 0, 0

            # Training
            for x, y in trainDataLoader:
                # Send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # Perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
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
                for x, y in valDataLoader:
                    # Send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # Make the predictions and calculate the validation loss
                    pred = self.model(x)
                    totalValLoss += self.loss_fn(pred, y)
                    # Calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / self.trainSteps
            avgValLoss = totalValLoss / self.valSteps
            # Calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)

            # Update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_acc"].append(valCorrect)
            H["time_elapsed"] = time.time() - start_time
            
            logging.info(('Epoch [{}/{}]: Train Acc: {:.2f}%, Train Loss: {:.4f} | Val Acc: {:.2f}%, Val Loss: {:.4f}').format(
                epoch + 1,
                self.epochs,
                trainCorrect,
                avgTrainLoss,
                valCorrect,
                avgValLoss,
            ))

        logging.info(f'Training complete in {H["time_elapsed"]/60:.2f}m')
        return H

    def validate(self, dataset):
        logging.info(f'Validating model...')

        testDataLoader = DataLoader(dataset, batch_size=self.batch_size)

        start_time = time.time()
        with torch.no_grad():  # Turn off autograd for testing evaluation
            self.model.eval()  # Set the model in evaluation mode
            preds = []  # Initialize a list to store our predictions
            accuracy = 0
            # Evaluation with test dataset
            for x, y in testDataLoader:
                # Send the input to the device
                x = x.to(self.device)
                # Make the predictions and add them to the list
                pred = self.model(x)
                preds.extend(pred.argmax(axis=1).cpu().numpy())
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # log acc and time in same line
        logging.info('Validation Completed. Test Acc: {:.2f}% | Time Elapsed: {:.2f}s'.format(
            accuracy / len(testDataLoader.dataset) * 100,
            time.time() - start_time
        ))

        return accuracy / len(testDataLoader.dataset)