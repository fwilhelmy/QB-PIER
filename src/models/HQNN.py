import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

class HQNN(nn.Module):
    def __init__(self, numChannels, classes):
        super(HQNN, self).__init__()
        self.num_qubits = numChannels # Input size
        self.initialize_weights()

    # Generating random weights
    # TODO Should be changed because of the Barren Platteaus problem
    def initialize_weights(self):
        self.theta = nn.Parameter(torch.randn(self.num_qubits))

    def forward(self, x):
        return QuantumLayer.apply(x, self.theta)

class HybridQuantumModel(nn.Module):
    def __init__(self, num_qubits, num_classes):
        super(HybridQuantumModel, self).__init__()
        self.conv_layer = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.quantum_layer = QuantumLayerModule(num_qubits)
        self.fc_layer = nn.Linear(num_qubits, 64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.quantum_layer(x)
        x = self.fc_layer(x)
        x = self.softmax(x)
        return x

# Create an instance of the model
model = HybridQuantumModel(num_qubits=4, num_classes=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Display model summary
print(model)