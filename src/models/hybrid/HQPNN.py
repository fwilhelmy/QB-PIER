from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import BatchNorm2d
from torch import flatten
from models.quantum.ParallelVQC import ParallelVQC
from models.quantum.ansatzs.BasicEntanglingAnsatz import BasicEntanglingAnsatz
from models.BaseModel import BaseModel
import torch

from torch.optim import Adam
from torch import nn
import time


class HQPNN(BaseModel):
    """Hybrid Quantum Parrallel Neural Network (HQPNN) model."""
    def __init__(self, in_channels, classes, in_qlayer = 20, n_qubits = 5, n_qdepth = (3), qcircuit = "strongly_entangling", **kwargs):
        # Call the parent constructor
        super().__init__(**kwargs)

        # Initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=(5, 5), padding=2
        )
        self.batchNorm1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2
        )
        self.batchNorm2 = BatchNorm2d(32)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=1568, out_features=in_qlayer)
        self.relu3 = ReLU()

        # Initialize the quantum layer
        self.qlayer1 = ParallelVQC(
            size_in=in_qlayer,
            n_layers=n_qdepth,
            n_qubits=n_qubits,
            ansatz_cls=BasicEntanglingAnsatz,
        )

        # Initialize our softmax classifier
        self.fc2 = Linear(in_features=in_qlayer, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # ==== Convolutional layer
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)  # flatten the output from the previous layer and pass it

        # ==== FC layer
        x = self.fc1(x)
        x = self.relu3(x)

        # ==== Quantum layer
        x = self.qlayer1(x)

        # ==== Softmax classifier
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output  # return the output predictions
    

    # old fit method
    def fit2(self, trainDataLoader, valDataLoader):
        # Calculate steps per epoch for training and validation set
        trainSteps = len(trainDataLoader.dataset) // trainDataLoader.batch_size
        valSteps = len(valDataLoader.dataset) // valDataLoader.batch_size

        # Configure the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Use the optimizer and loss function from the BaseModel
        opt = Adam(self.parameters(), lr=0.001)
        lossFn = nn.CrossEntropyLoss()

        # Initialize a dictionary to store training history
        H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        print("[TRAIN] Training the model...")

        startTime = time.time()  # To measure how long training is going to take

        # loop over our epochs
        for e in range(0, self.max_epochs):
            # Set the model in training mode
            self.train()
            # Initialize the total training/validation loss/correct predictions
            totalTrainLoss, totalValLoss, trainCorrect, valCorrect = 0, 0, 0, 0

            # Training
            for x, y in trainDataLoader:
                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # Perform a forward pass and calculate the training loss
                pred = self(x)
                loss = lossFn(pred, y)
                # Zero out the gradients, perform the backpropagation step, and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # Sum the loss to the total training loss so far
                totalTrainLoss += loss
                # Calculate the number of correct predictions
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Validation
            with torch.no_grad():
                # Set the model in evaluation mode
                self.eval()
                # Loop over the validation set
                for x, y in valDataLoader:
                    # Send the input to the device
                    (x, y) = (x.to(device), y.to(device))
                    # Make the predictions and calculate the validation loss
                    pred = self(x)
                    totalValLoss += lossFn(pred, y)
                    # Calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            # Calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)

            # Update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["val_acc"].append(valCorrect)
            intervalTime = time.time() - startTime

            # Print the model training and validation information
            print(
                "[TRAIN] EPOCH: {}/{} | Train loss: {:0.6f} | Train acc: {:0.4f} | Val loss: {:0.6f} | Val acc: {:0.4f} "
                "| Time elapsed: {:0.2f}".format(
                    e + 1,
                    self.max_epochs,
                    avgTrainLoss,
                    trainCorrect,
                    avgValLoss,
                    valCorrect,
                    intervalTime,
                )
            )

        # Finish measuring how long training took
        endTime = time.time()
        print("[TRAIN] Finished training the model...")
        print(
            "[TRAIN] Total time taken to train the model: {:.2f}s".format(endTime - startTime)
        )
        
        return H