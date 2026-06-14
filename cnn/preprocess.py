import pandas as pd
import os

metadata_path = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_metadata.csv"

image_dir_1 = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_images_part_1"
image_dir_2 = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_images_part_2"

df = pd.read_csv(metadata_path)

# Create binary labels
df["label"] = (df["dx"] == "mel").astype(int)

# Create image path column
def get_image_path(image_id):
    path1 = os.path.join(image_dir_1, image_id + ".jpg")
    path2 = os.path.join(image_dir_2, image_id + ".jpg")

    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

df["image_path"] = df["image_id"].apply(get_image_path)

print("Total Records:", len(df))
print("Missing Images:", df["image_path"].isnull().sum())

print("\nSample Paths:")
print(df[["image_id", "image_path"]].head())