"""
model.py — CNN for MNIST digit + math symbol recognition
Trains on MNIST; extended symbol set loaded from file if available.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "smartboard_model.h5"

# Class labels: 0-9 digits + basic math symbols
DIGIT_LABELS = [str(i) for i in range(10)]
SYMBOL_LABELS = ['+', '-', '×', '÷', '=', '(', ')']
ALL_LABELS = DIGIT_LABELS + SYMBOL_LABELS


def build_cnn(num_classes: int = 10) -> keras.Model:
    """
    Build a compact but strong CNN for 28×28 grayscale classification.
    Architecture: Conv→Pool→Conv→Pool→Dense→Dropout→Output
    """
    model = keras.Sequential([
        # Input
        layers.Input(shape=(28, 28, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name="SmartBoard_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_mnist_model(epochs: int = 10, save_path: str = MODEL_PATH) -> keras.Model:
    """Train CNN on MNIST and save weights."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize + reshape
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]   # (60000, 28, 28, 1)
    x_test  = x_test[..., np.newaxis]

    model = build_cnn(num_classes=10)
    logger.info(model.summary())

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    logger.info(f"Training for up to {epochs} epochs...")
    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Test accuracy: {acc:.4f} | Loss: {loss:.4f}")

    model.save(save_path)
    logger.info(f"Model saved → {save_path}")
    return model


def load_or_train_model(save_path: str = MODEL_PATH) -> keras.Model:
    """Load saved model; train from scratch if not found."""
    if os.path.exists(save_path):
        logger.info(f"Loading model from {save_path}")
        try:
            model = keras.models.load_model(save_path)
            return model
        except Exception as e:
            logger.warning(f"Failed to load model ({e}). Retraining...")

    logger.info("No saved model found. Training now (this may take a few minutes)...")
    return train_mnist_model(save_path=save_path)


def predict_digit(model: keras.Model, img_28x28: np.ndarray) -> tuple[int, float, np.ndarray]:
    """
    Run inference on a single 28×28 float32 array (values in [0,1]).
    Returns: (predicted_class, confidence, all_probabilities)
    """
    if img_28x28.ndim == 2:
        img_28x28 = img_28x28[..., np.newaxis]
    inp = img_28x28[np.newaxis, ...]       # (1, 28, 28, 1)
    probs = model.predict(inp, verbose=0)[0]  # shape (10,)
    cls   = int(np.argmax(probs))
    conf  = float(probs[cls])
    return cls, conf, probs


def get_top_k_predictions(probs: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """Return top-k (label, probability) pairs sorted descending."""
    indices = np.argsort(probs)[::-1][:k]
    return [(DIGIT_LABELS[i] if i < len(DIGIT_LABELS) else '?', float(probs[i]))
            for i in indices]
