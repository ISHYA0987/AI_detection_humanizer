import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "google/flan-t5-large"

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "data" / "processed" / "paired_dataset.csv"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading dataset...")
df = pd.read_csv(DATASET)

print(f"Total samples: {len(df):,}")

input_lengths = []
target_lengths = []

for text in df["input_text"]:

    ids = tokenizer.encode(
        str(text),
        add_special_tokens=True
    )

    input_lengths.append(len(ids))

for text in df["target_text"]:

    ids = tokenizer.encode(
        str(text),
        add_special_tokens=True
    )

    target_lengths.append(len(ids))

input_lengths = np.array(input_lengths)
target_lengths = np.array(target_lengths)


def print_stats(name, arr):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Mean        : {arr.mean():.2f}")
    print(f"Median      : {np.median(arr):.2f}")
    print(f"Min         : {arr.min()}")
    print(f"Max         : {arr.max()}")
    print(f"90 Percentile : {np.percentile(arr,90):.0f}")
    print(f"95 Percentile : {np.percentile(arr,95):.0f}")
    print(f"99 Percentile : {np.percentile(arr,99):.0f}")


print_stats("INPUT TOKENS", input_lengths)
print_stats("TARGET TOKENS", target_lengths)

print("\nSamples exceeding limits")

for limit in [256, 512, 768, 1024]:

    count = (input_lengths > limit).sum()

    print(f"Input > {limit:4d} : {count:6d} ({count/len(df)*100:.2f}%)")

print()

for limit in [128, 256, 512]:

    count = (target_lengths > limit).sum()

    print(f"Target > {limit:3d} : {count:6d} ({count/len(df)*100:.2f}%)")