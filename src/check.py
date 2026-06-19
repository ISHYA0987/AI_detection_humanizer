import pandas as pd

df = pd.read_csv("data/processed/final_dataset_trimmed.csv")

duplicates = (
    df.groupby("text")["label"]
      .nunique()
      .reset_index()
)

conflicting = duplicates[duplicates["label"] > 1]

print("Conflicting labels:", len(conflicting))