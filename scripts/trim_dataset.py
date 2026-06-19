import pandas as pd

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("data/processed/final_dataset.csv")

print("Original Shape:", df.shape)

# ==========================================
# Create Features
# ==========================================
df["word_count"] = df["text"].str.split().str.len()
df["char_count"] = df["text"].str.len()

# ==========================================
# Remove Short Texts
# ==========================================
df = df[df["word_count"] >= 30]

print("After removing texts <30 words:", df.shape)

# ==========================================
# Remove Very Long Texts
# ==========================================
df = df[df["word_count"] <= 2500]

print("After removing texts >2500 words:", df.shape)

# ==========================================
# Remove Very Long Character Texts
# ==========================================
df = df[df["char_count"] <= 15000]

print("After removing texts >15000 characters:", df.shape)

# ==========================================
# Remove Duplicate Texts
# ==========================================
df = df.drop_duplicates(subset=["text"])

# ==========================================
# Remove Temporary Columns
# ==========================================
df = df.drop(columns=["word_count", "char_count"])

# ==========================================
# Reset Index
# ==========================================
df.reset_index(drop=True, inplace=True)

# ==========================================
# Final Statistics
# ==========================================
print("\nFinal Shape:", df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nLabel Percentage:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

# ==========================================
# Save
# ==========================================
df.to_csv(
    "data/processed/final_dataset_trimmed.csv",
    index=False
)

print("\n✅ Trimmed dataset saved successfully!")