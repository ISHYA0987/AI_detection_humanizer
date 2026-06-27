import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from config import *


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_dataframe(path):

    df = pd.read_csv(path)

    df = df[["input_text", "target_text"]]

    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def tokenize_function(examples):

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def load_datasets():

    train_df = load_dataframe(TRAIN_FILE)
    valid_df = load_dataframe(VALID_FILE)
    test_df = load_dataframe(TEST_FILE)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    valid_dataset = valid_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=valid_dataset.column_names,
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
    )


def get_data_collator(model):

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )


def get_tokenizer():

    return tokenizer