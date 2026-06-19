import os
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

df["label"] = df["label"].map({0: 1, 1: 0})



df = df.dropna(subset=["text"])


df = df[df["text"].str.strip() != ""]

df = df.drop_duplicates(subset=["text"])

df = df[["text", "label"]]

df = df.reset_index(drop=True)

print("\nDataset Shape:", df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Text:", df.duplicated(subset=["text"]).sum())


os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/dataset3_cleaned.csv"

df.to_csv(output_path, index=False)

print(f"\n Cleaned dataset saved to: {output_path}")