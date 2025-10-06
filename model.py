"""
Neural network model definition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim, num_classes):
    """Build a simple feedforward neural network for classification"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output layer for classification
    ])

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
