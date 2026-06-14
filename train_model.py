"""
train_model.py — Skin Lens CNN Training Script
==============================================
Trains a CNN on the ISIC skin lesion dataset (or any compatible dataset)
and saves the model to model/skin_lens_cnn.h5

Dataset structure expected:
    data/
      train/
        benign_nevus/        ← folder name = class label
        seborrheic_keratosis/
        dysplastic_nevus/
        bcc/
        melanoma/
        scc/
      val/
        (same structure)

Quick start:
    pip install tensorflow pillow numpy scikit-learn matplotlib
    python train_model.py

The script:
  1. Loads images with augmentation
  2. Fine-tunes MobileNetV2 (fast, accurate, mobile-friendly)
  3. Saves the best checkpoint + final model
  4. Prints a classification report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR      = 'data'
MODEL_DIR     = 'model'
MODEL_PATH    = os.path.join(MODEL_DIR, 'skin_lens_cnn.h5')
INPUT_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 10    # Train only the new head first
EPOCHS_FINE   = 20    # Then unfreeze and fine-tune deeper layers
LEARNING_RATE = 1e-4
NUM_CLASSES   = 6

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Class names (must match folder names in data/train/) ──────────────────────
CLASS_NAMES = [
    'benign_nevus',
    'seborrheic_keratosis',
    'dysplastic_nevus',
    'bcc',
    'melanoma',
    'scc',
]


# ── Data generators ────────────────────────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    """
    MobileNetV2 base (pretrained on ImageNet) + custom classification head.
    MobileNetV2 is chosen for:
      - High accuracy on medical imaging tasks
      - Fast inference (runs on CPU in ~0.3s per image)
      - Small model size (~14MB)
    """
    base = MobileNetV2(
        input_shape=(*INPUT_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False  # Freeze base initially

    inputs = keras.Input(shape=(*INPUT_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Skin Lens CNN Training")
    print("=" * 60)

    # ── Check if data exists ──
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"\n❌ Training data not found in '{DATA_DIR}/train'.")
        print("Please organize your dataset as follows:")
        print(f"  {DATA_DIR}/")
        print("    ├── train/")
        for name in CLASS_NAMES:
            print(f"    │   ├── {name}/  (put images here)")
        print("    └── val/")
        print("        ├── ...          (same structure for validation)")
        return

    train_gen, val_gen = build_generators()
    model, base = build_model()

    # ── Calculate class weights to handle imbalance ──
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\n⚖️  Class weights: {class_weights_dict}")

    # ── Phase 1: Train head only ──
    print("\n📦 Phase 1: Training classification head (base frozen)...")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    model.summary()

    callbacks_p1 = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        ),
    ]

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks_p1,
        class_weight=class_weights_dict,
    )

    # ── Phase 2: Fine-tune top layers of base ──
    print("\n🔓 Phase 2: Fine-tuning top layers of MobileNetV2...")
    base.trainable = True

    # Only unfreeze the top 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE / 10),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    callbacks_p2 = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        ),
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=callbacks_p2,
        class_weight=class_weights_dict,
    )

    # ── Evaluation ──
    print("\n📊 Evaluating on validation set...")
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Plot training history ──
    plot_history(history1, history2)

    print(f"\n✅ Model saved to: {MODEL_PATH}")


def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc, label='Train Accuracy')
    ax1.plot(val_acc, label='Val Accuracy')
    ax1.axvline(x=len(h1.history['accuracy']), color='grey', linestyle='--', label='Fine-tune start')
    ax1.set_title('Accuracy'); ax1.legend()

    ax2.plot(loss, label='Train Loss')
    ax2.plot(val_loss, label='Val Loss')
    ax2.axvline(x=len(h1.history['loss']), color='grey', linestyle='--', label='Fine-tune start')
    ax2.set_title('Loss'); ax2.legend()

    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150)
    print("📈 Training plot saved to model/training_history.png")


if __name__ == '__main__':
    train()
