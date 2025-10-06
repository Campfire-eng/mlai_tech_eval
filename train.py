"""
Training script for the baseline model
"""

import numpy as np
from dataset import get_data
from model import build_model


def train():
    """Train the model"""
    # Load data
    X, y, num_classes = get_data()

    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    print(model.summary())

    # Train model
    print("\nTraining model...")
    history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    # Evaluate
    print("\nEvaluating model...")
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y)

    print(f"\nFinal Training Accuracy: {accuracy:.4f}")
    print(f"\nSample predictions vs actual:")
    for i in range(5):
        print(f"  Predicted class: {predicted_classes[i]}, Actual class: {y[i]}, Confidence: {predictions[i][predicted_classes[i]]:.4f}")

    return model, history


if __name__ == "__main__":
    train()
