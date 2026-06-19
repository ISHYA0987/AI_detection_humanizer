import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/final_dataset.csv")


print("First Five Samples")

print(df.head())


print("Dataset Information")


df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Text Samples:")
print(df.duplicated(subset=["text"]).sum())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nSource Distribution:")
print(df["source"].value_counts())

if "char_count" not in df.columns:
    df["char_count"] = df["text"].astype(str).str.len()

if "word_count" not in df.columns:
    df["word_count"] = df["text"].astype(str).str.split().str.len()


sample_size = min(50000, len(df))
plot_df = df.sample(n=sample_size, random_state=42)

print(f"\nUsing {sample_size:,} samples for visualization.")

plot_df = plot_df.copy()
plot_df["Label"] = plot_df["label"].map({
    0: "Human",
    1: "AI"
})


sns.set_theme(style="whitegrid")


plt.figure(figsize=(6, 5))

sns.countplot(
    data=plot_df,
    x="Label",
    order=["Human", "AI"]
)

plt.title("Class Distribution")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

sns.countplot(
    data=plot_df,
    x="source",
    order=plot_df["source"].value_counts().index
)

plt.title("Dataset Source Distribution")
plt.xlabel("")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.hist(
    plot_df["char_count"],
    bins=50
)

plt.title("Character Count Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.hist(
    plot_df["word_count"],
    bins=50
)

plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=plot_df,
    x="Label",
    y="word_count",
    order=["Human", "AI"],
    showfliers=False
)

plt.title("Word Count by Class")
plt.xlabel("")
plt.ylabel("Word Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))

sns.boxplot(
    data=plot_df,
    x="Label",
    y="char_count",
    order=["Human", "AI"],
    showfliers=False
)

plt.title("Character Count by Class")
plt.xlabel("")
plt.ylabel("Character Count")
plt.tight_layout()
plt.show()