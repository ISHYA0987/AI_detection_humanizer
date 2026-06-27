from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Read dataset from Hugging Face
df = pd.read_csv("hf://datasets/gsingh1-py/train/train.csv")

# Save locally
output_path = RAW_DIR / "train.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head())