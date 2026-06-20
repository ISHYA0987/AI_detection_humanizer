import pandas as pd

from datasets import Dataset

from transformers import AutoTokenizer

from config import *


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def load_datasets():

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Keep only required columns
    train_df = train_df[["text", "label"]]
    val_df = val_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
    )

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
    )

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
    )

    return train_dataset, val_dataset, test_dataset, tokenizer