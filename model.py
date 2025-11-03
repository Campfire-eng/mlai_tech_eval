"""
Neural network model definition
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """Simple feedforward neural network for classification"""

    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax here - will use CrossEntropyLoss
        return x


def build_model(input_dim, num_classes):
    """Build a simple feedforward neural network for classification"""
    model = SimpleNN(input_dim, num_classes)
    return model
