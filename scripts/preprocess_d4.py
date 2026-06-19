import pandas as pd
import os

df = pd.read_csv("data/raw/shuffled_Human.csv")

print(df.info())
print(df.head())

df = df[["Text"]]

df = df.rename(columns={
    "Text": "text"
})

df = df.dropna(subset=["text"])

df = df[df["text"].str.strip() != ""]

df = df.drop_duplicates(subset=["text"])

df["label"] = 0

df.reset_index(drop=True, inplace=True)

print(df.shape)
print(df.head())

os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/human_dataset4_cleaned.csv", index=False)

print("Saved successfully")