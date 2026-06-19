import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("data/processed/final_dataset_trimmed.csv")

print("=" * 60)
print("Original Dataset")
print("=" * 60)

print("Shape:", df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nLabel Percentage:")
print((df["label"].value_counts(normalize=True) * 100).round(2))


train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

def print_stats(name, dataset):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Shape:", dataset.shape)

    print("\nLabel Distribution:")
    print(dataset["label"].value_counts())

    print("\nLabel Percentage:")
    print((dataset["label"].value_counts(normalize=True) * 100).round(2))


print_stats("Training Set", train_df)
print_stats("Validation Set", val_df)
print_stats("Test Set", test_df)

print("\n" + "=" * 60)
print("Dataset Split Completed Successfully!")
print("=" * 60)

print("\nFiles Saved:")
print("✓ data/processed/train.csv")
print("✓ data/processed/validation.csv")
print("✓ data/processed/test.csv")