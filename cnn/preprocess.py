import pandas as pd
import os
import numpy as np
from PIL import Image

# ==========================
# PATHS
# ==========================

METADATA_PATH = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_metadata.csv"

IMAGE_DIR_1 = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_images_part_1"
IMAGE_DIR_2 = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_images_part_2"

IMG_SIZE = (224, 224)

# ==========================
# LOAD DATASET
# ==========================

def load_metadata():
    df = pd.read_csv(METADATA_PATH)

    # Binary classification
    df["label"] = (df["dx"] == "mel").astype(int)

    return df

# ==========================
# IMAGE PATH RESOLUTION
# ==========================

def get_image_path(image_id):

    path1 = os.path.join(IMAGE_DIR_1, image_id + ".jpg")
    path2 = os.path.join(IMAGE_DIR_2, image_id + ".jpg")

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None

# ==========================
# ATTACH IMAGE PATHS
# ==========================

def add_image_paths(df):

    df["image_path"] = df["image_id"].apply(get_image_path)

    missing = df["image_path"].isnull().sum()

    print(f"Missing Images: {missing}")

    return df

# ==========================
# LOAD + PREPROCESS IMAGE
# ==========================

def preprocess_image(image_path):

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)

    image = image / 255.0

    return image

# ==========================
# VERIFY PIPELINE
# ==========================

if __name__ == "__main__":

    df = load_metadata()

    print(f"Total Images: {len(df)}")

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    df = add_image_paths(df)

    sample_image = preprocess_image(df.iloc[0]["image_path"])

    print("\nSample Image Loaded")

    print("Shape:", sample_image.shape)

    print("Min Pixel:", sample_image.min())

    print("Max Pixel:", sample_image.max())