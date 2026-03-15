# ================================================================
#  BRAIN TUMOR DETECTION V2.0 — train_model.py
#  Advanced CNN with better accuracy
#  Run: py train_model.py
# ================================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json

# ── Settings ──────────────────────────────────────────────────
IMG_SIZE     = 150        # Larger image = better accuracy
DATASET_PATH = "dataset"
MODEL_FILE   = "model.h5"
HISTORY_FILE = "training_history.json"
EPOCHS       = 25
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
# ──────────────────────────────────────────────────────────────


def load_images():
    """Load and preprocess all images from dataset folders"""
    images, labels = [], []
    categories = {"tumor": 1, "no_tumor": 0}

    print("\n📂 Loading images from dataset...")
    print("-" * 40)

    for folder, label in categories.items():
        path = os.path.join(DATASET_PATH, folder)

        if not os.path.exists(path):
            print(f"   ❌ Missing: {path}")
            continue

        files = [f for f in os.listdir(path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        print(f"   📁 {folder:12s} → {len(files):4d} images")

        for fname in files:
            img = cv2.imread(os.path.join(path, fname))
            if img is None:
                continue

            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Apply CLAHE for better contrast
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Normalize
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


def build_advanced_model():
    """
    Advanced CNN with:
    - More conv layers
    - Global Average Pooling
    - Better regularization
    """
    model = Sequential([

        # ── Block 1 ──
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        # ── Block 2 ──
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        # ── Block 3 ──
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # ── Block 4 ──
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dropout(0.4),

        # ── Classifier ──
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training(history):
    """Save training accuracy/loss graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['accuracy'],     label='Train Accuracy', color='#2196F3', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Val Accuracy',   color='#4CAF50', linewidth=2)
    ax1.set_title('Model Accuracy',  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['loss'],     label='Train Loss', color='#F44336', linewidth=2)
    ax2.plot(history['val_loss'], label='Val Loss',   color='#FF9800', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('static/training_graph.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   📊 Training graph saved: static/training_graph.png")


def train():
    print("\n" + "=" * 56)
    print("   🧠  Brain Tumor Detection V2.0 — Training")
    print("=" * 56)

    # Load data
    X, y = load_images()

    if len(X) == 0:
        print("\n❌ No images found!")
        print("   Add images to dataset/tumor/ and dataset/no_tumor/")
        return

    tumor_count    = int(np.sum(y == 1))
    no_tumor_count = int(np.sum(y == 0))

    print(f"\n✅ Dataset Summary:")
    print(f"   Total    : {len(X)} images")
    print(f"   Tumor    : {tumor_count} images")
    print(f"   No Tumor : {no_tumor_count} images")

    # One-hot encode
    Y = to_categorical(y, num_classes=2)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n📊 Data Split:")
    print(f"   Training : {len(X_train)} images")
    print(f"   Testing  : {len(X_test)} images")

    # Build model
    print("\n🔧 Building advanced CNN model...")
    model = build_advanced_model()
    model.summary()

    # Advanced data augmentation
    aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_FILE, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=0.00001, verbose=1)
    ]

    # Train
    print(f"\n🚀 Training started... (up to {EPOCHS} epochs)\n")
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # Predictions for report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n" + "=" * 56)
    print("   ✅  TRAINING COMPLETE!")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   Loss     : {loss:.4f}")
    print(f"   Model    : {MODEL_FILE}")
    print("=" * 56)

    print("\n📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes,
                                 target_names=['No Tumor', 'Tumor']))

    # Save history
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open(HISTORY_FILE, 'w') as f:
        json.dump(hist_dict, f)

    # Save graphs
    os.makedirs('static', exist_ok=True)
    plot_training(history.history)

    print("\n▶  Now run: py app.py")
    print("   Open:     http://127.0.0.1:5000\n")


if __name__ == "__main__":
    train()
