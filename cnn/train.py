from preprocess import load_metadata, add_image_paths

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import numpy as np

# ==========================
# DATASET
# ==========================

df = load_metadata()
df = add_image_paths(df)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# ==========================
# CLASS WEIGHTS
# ==========================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"]
)

class_weights = {
    0: class_weights[0],
    1: class_weights[1]
}

print("Class Weights:", class_weights)

# ==========================
# DATA LOADERS
# ==========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens",
    labels=None
)

# Placeholder for now
print("Dataset Preparation Complete")

# ==========================
# MODEL
# ==========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()