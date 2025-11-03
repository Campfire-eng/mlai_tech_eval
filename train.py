"""
Training script for the baseline Forest Covertype classification model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import get_data
from model import build_model


def evaluate_model(model, X, y, device=None):
    """Evaluate model and return accuracy and detailed metrics"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    
    accuracy = accuracy_score(y, predicted_classes)
    
    return accuracy, predicted_classes


def train():
    """Train the model"""
    # Detect and set device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    print()
    
    # Load data with splits
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, scaler = get_data()
    
    # Convert to PyTorch tensors and move to device
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Build model and move to device
    print("\nBuilding model...")
    model = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
    model = model.to(device)
    print(model)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Training for 10 epochs...")
    print()
    
    # Track metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    
    # Calculate validation loss
    def compute_val_loss(model, X_val, y_val):
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            outputs = model(X_val_tensor)
            loss = criterion(outputs, y_val_tensor)
        return loss.item()
    
    for epoch in range(10):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_dataloader:
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
        
        train_accuracy = correct / total
        avg_loss = epoch_loss / len(train_dataloader)
        
        # Validation phase
        val_accuracy, _ = evaluate_model(model, X_val, y_val, device)
        val_loss = compute_val_loss(model, X_val, y_val)
        
        # Track metrics
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        
        print(f"Epoch {epoch+1:2d}/10 - Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
    
    print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f}")
    
    # Create and save plots
    print("\n" + "="*60)
    print("SAVING TRAINING PLOTS")
    print("="*60)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = plots_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved loss plot: {loss_path}")
    plt.close()
    
    # Plot 2: Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = plots_dir / "training_accuracy.png"
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved accuracy plot: {acc_path}")
    plt.close()
    
    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss subplot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_path = plots_dir / "training_curves.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved combined plot: {combined_path}")
    plt.close()
    
    print("="*60)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_accuracy, test_predictions = evaluate_model(
        model, X_test, y_test, device
    )
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    print("\n" + "="*60)
    print("BASELINE MODEL COMPLETE")
    print("="*60)
    print("\nThis baseline model has room for improvement.")
    print("Explore the results and identify areas to enhance.")
    print()
    
    return model


if __name__ == "__main__":
    train()
