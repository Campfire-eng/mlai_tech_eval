"""
Training script for the baseline model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataset import get_data
from model import build_model


def train():
    """Train the model"""
    # Load data
    X, y, num_classes = get_data()

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    print(model)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters())}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train model
    print("\nTraining model...")
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Evaluate
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).numpy()
        accuracy = np.mean(predicted_classes == y)

    print(f"\nFinal Training Accuracy: {accuracy:.4f}")
    print(f"\nSample predictions vs actual:")
    with torch.no_grad():
        sample_predictions = model(X_tensor[:5])
        sample_probs = torch.softmax(sample_predictions, dim=1)
        for i in range(5):
            pred_class = predicted_classes[i]
            confidence = sample_probs[i, pred_class].item()
            print(f"  Predicted class: {pred_class}, Actual class: {y[i]}, Confidence: {confidence:.4f}")

    return model


if __name__ == "__main__":
    train()
