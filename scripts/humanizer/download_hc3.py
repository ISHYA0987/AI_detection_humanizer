from pathlib import Path
import pandas as pd

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent


RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "all.jsonl"
OUTPUT_FILE = RAW_DIR / "hc3.csv"

print("Loading HC3 dataset...")

df = pd.read_json(INPUT_FILE, lines=True)

print(f"Dataset loaded successfully!")
print(f"Rows    : {len(df):,}")
print(f"Columns : {list(df.columns)}")

# =====================================================
# Save CSV
# =====================================================
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDataset saved to:")
print(OUTPUT_FILE)