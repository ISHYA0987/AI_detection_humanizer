import ast
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from pathlib import Path
from sklearn.model_selection import train_test_split

import html
import re

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

HC3_FILE = RAW_DIR / "all.jsonl"
STORY_FILE = RAW_DIR / "train.csv"

OUTPUT_FILE = PROCESSED_DIR / "paired_dataset.csv"

TRAIN_FILE = SPLITS_DIR / "train.csv"
VALID_FILE = SPLITS_DIR / "valid.csv"
TEST_FILE = SPLITS_DIR / "test.csv"




def clean_text(text):

    if text is None:
        return None

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize newlines and tabs
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Normalize quotation marks
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

    # Normalize dashes
    text = (
        text.replace("–", "-")
            .replace("—", "-")
    )

    # Normalize ellipsis
    text = text.replace("…", "...")

    # Replace HC3 placeholders
    text = re.sub(r"URL_\d+", "<URL>", text)
    text = re.sub(r"EMAIL_\d+", "<EMAIL>", text)
    text = re.sub(r"PHONE_\d+", "<PHONE>", text)
    text = re.sub(r"DATE_\d+", "<DATE>", text)
    text = re.sub(r"TIME_\d+", "<TIME>", text)
    text = re.sub(r"PERSON_\d+", "<PERSON>", text)
    text = re.sub(r"LOCATION_\d+", "<LOCATION>", text)
    text = re.sub(r"NUMBER_\d+", "<NUMBER>", text)

    # Remove zero-width characters
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Collapse repeated punctuation
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    # Fix tokenized contractions
    text = re.sub(r"\b([A-Za-z]+)\s+'\s+([A-Za-z]+)\b", r"\1'\2", text)

# Fix common contractions
    replacements = {
        " n't": "n't",
        " 's": "'s",
        " 're": "'re",
        " 've": "'ve",
        " 'll": "'ll",
        " 'd": "'d",
        " 'm": "'m",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.strip()

    if len(text) < 20:
        return None

    return text


def parse_answers(value):

    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [str(x).strip() for x in value if str(x).strip()]
    except:
        pass

    if isinstance(value, str):

        value = value.strip()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]

        except:
            pass

        return [value]

    return [str(value).strip()]
def load_hc3():

    print("Loading HC3 Dataset...")

    rows = []

    with open(HC3_FILE, "r", encoding="utf-8") as f:

        for line in f:

            sample = json.loads(line)

            question = clean_text(sample.get("question"))

            if question is None:
                continue

            human_answers = parse_answers(sample.get("human_answers"))
            ai_answers = parse_answers(sample.get("chatgpt_answers"))

            if len(human_answers) == 0 or len(ai_answers) == 0:
                continue

            human_answers = [clean_text(x) for x in human_answers]
            ai_answers = [clean_text(x) for x in ai_answers]

            human_answers = [x for x in human_answers if x is not None]
            ai_answers = [x for x in ai_answers if x is not None]

            if len(human_answers) == 0 or len(ai_answers) == 0:
                continue

            for ai in ai_answers:

                input_text = (
                    "Humanize the following AI-generated answer.\n\n"
                    f"Question:\n{question}\n\n"
                    f"AI Answer:\n{ai}"
                )

                for human in human_answers:

                    rows.append(
                        {
                            "input_text": input_text,
                            "target_text": human,
                            "source": "HC3",
                            "model": "ChatGPT"
                        }
                    )

    df = pd.DataFrame(rows)

    print(f"HC3 training pairs: {len(df):,}")

    return df
def load_story():

    print("Loading Story Dataset...")

    df = pd.read_csv(STORY_FILE)

    model_columns = [
        "gemma-2-9b",
        "mistral-7B",
        "qwen-2-72B",
        "llama-8B",
        "accounts/yi-01-ai/models/yi-large",
        "GPT_4-o"
    ]

    MAX_INPUT_WORDS = 1200
    MAX_TARGET_WORDS = 1200

    rows = []
    removed = 0

    for _, row in df.iterrows():

        prompt = clean_text(row["prompt"])
        human_story = clean_text(row["Human_story"])

        if prompt is None or human_story is None:
            continue

        target_words = len(human_story.split())

        if target_words > MAX_TARGET_WORDS:
            removed += len(model_columns)
            continue

        for model in model_columns:

            ai_story = clean_text(row[model])

            if ai_story is None:
                continue

            if "error" in ai_story.lower():
                continue

            input_text = (
                "Humanize the following AI-generated story.\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"AI Story:\n{ai_story}"
            )

            input_words = len(input_text.split())

            if input_words > MAX_INPUT_WORDS:
                removed += 1
                continue

            rows.append(
                {
                    "input_text": input_text,
                    "target_text": human_story,
                    "source": "Story",
                    "model": model
                }
            )

    story_df = pd.DataFrame(rows)

    story_df.drop_duplicates(
        subset=["input_text", "target_text"],
        inplace=True
    )

    story_df.reset_index(drop=True, inplace=True)

    print(f"Removed corrupted/oversized samples : {removed:,}")
    print(f"Story training pairs              : {len(story_df):,}")

    return story_df
def main():

    hc3_df = load_hc3()
    story_df = load_story()

    print("\nMerging datasets...")

    dataset = pd.concat(
        [hc3_df, story_df],
        ignore_index=True
    )

    print(f"Total pairs before cleaning : {len(dataset):,}")

    dataset.dropna(
        subset=["input_text", "target_text"],
        inplace=True
    )

    dataset.drop_duplicates(
        subset=["input_text", "target_text"],
        inplace=True
    )

    dataset["input_length"] = dataset["input_text"].str.split().str.len()
    dataset["target_length"] = dataset["target_text"].str.split().str.len()

    dataset = dataset[
        (dataset["input_length"] >= 10)
        & (dataset["target_length"] >= 10)
    ]

    dataset.drop(
        columns=["input_length", "target_length"],
        inplace=True
    )

    dataset = shuffle(
        dataset,
        random_state=42
    ).reset_index(drop=True)

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )
    print("\nRemoving extremely long samples...")

    before = len(dataset)

    dataset["input_words"] = dataset["input_text"].str.split().str.len()
    dataset["target_words"] = dataset["target_text"].str.split().str.len()

    dataset = dataset[
        (dataset["input_words"] <= 1500) &
        (dataset["target_words"] <= 1500)
    ]

    removed = before - len(dataset)

    dataset.drop(
        columns=["input_words", "target_words"],
        inplace=True
    )

    print(f"Removed {removed} oversized samples.")

    print("\nCreating Train / Validation / Test Splits...")

    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8"
    )

    valid_df.to_csv(
        VALID_FILE,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n======================================")
    print("Dataset Created Successfully")
    print("======================================")

    print(f"Total Samples : {len(dataset):,}")
    print(f"Train Samples : {len(train_df):,}")
    print(f"Valid Samples : {len(valid_df):,}")
    print(f"Test Samples  : {len(test_df):,}")

    print(f"\nProcessed Dataset : {OUTPUT_FILE}")
    print(f"Train Split       : {TRAIN_FILE}")
    print(f"Validation Split  : {VALID_FILE}")
    print(f"Test Split        : {TEST_FILE}")

    print("\nSource Distribution")
    print(dataset["source"].value_counts())

    print("\nModel Distribution")
    print(dataset["model"].value_counts())

    print("\nSample")
    print(dataset.head())


if __name__ == "__main__":
    main()