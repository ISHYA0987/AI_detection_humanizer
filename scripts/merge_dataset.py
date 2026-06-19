import os
import pandas as pd


datasets = [
    ("dataset1", "data/processed/dataset1_cleaned.csv"),
    ("dataset2", "data/processed/dataset2_cleaned.csv"),
    ("dataset3", "data/processed/dataset3_cleaned.csv"),
    ("dataset4", "data/processed/human_dataset4_cleaned.csv"),
]

dfs = []

for name, path in datasets:
    df = pd.read_csv(path)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(df["label"].value_counts())

    
    df["source"] = name

    dfs.append(df)


merged_df = pd.concat(dfs, ignore_index=True)

print("\nShape before removing duplicates:", merged_df.shape)

before = len(merged_df)

merged_df.drop_duplicates(subset=["text"], inplace=True)

after = len(merged_df)

print(f"Removed {before - after} duplicate texts")


merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal Dataset Information")
print("-" * 40)

print(merged_df.info())

print("\nMissing Values")
print(merged_df.isnull().sum())

print("\nLabel Distribution")
print(merged_df["label"].value_counts())

print("\nLabel Percentage")
print(
    (merged_df["label"].value_counts(normalize=True) * 100).round(2)
)

print("\nSource Distribution")
print(merged_df["source"].value_counts())

os.makedirs("data/processed", exist_ok=True)

merged_df.to_csv(
    "data/processed/final_dataset.csv",
    index=False
)

print("\nFinal dataset saved successfully!")

print(f"\nTotal Samples : {len(merged_df)}")
print(f"Human Samples : {(merged_df['label']==0).sum()}")
print(f"AI Samples    : {(merged_df['label']==1).sum()}")

print("\nRandom Samples")
print(merged_df.sample(5))