import pandas as pd

metadata_path = r"C:\Users\kalin\OneDrive\Desktop\Projects\Personal\SkinLens\HAM10000_metadata.csv"

df = pd.read_csv(metadata_path)

# Binary labels
df["label"] = (df["dx"] == "mel").astype(int)

print("Total Images:", len(df))

print("\nBinary Distribution:")
print(df["label"].value_counts())

print("\nLabel Meaning:")
print("0 = Benign")
print("1 = Melanoma")