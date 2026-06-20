import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler



def extract_features(text):
    """
    Extract handcrafted linguistic features useful for
    AI-generated text detection.
    """

    text = str(text)

    words = text.split()

    word_count = len(words)

    char_count = len(text)

    # Sentence count
    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)

    # Average word length
    avg_word_length = (
        np.mean([len(word) for word in words])
        if word_count > 0 else 0
    )

    # Average sentence length
    avg_sentence_length = word_count / sentence_count

    # Character-level counts
    punctuation_count = len(re.findall(r"[.,!?;:]", text))
    comma_count = text.count(",")
    exclamation_count = text.count("!")
    question_count = text.count("?")
    digit_count = len(re.findall(r"\d", text))

    # Uppercase ratio
    uppercase_ratio = (
        sum(char.isupper() for char in text)
        / max(char_count, 1)
    )

    # Lexical diversity
    unique_word_ratio = (
        len(set(words)) / max(word_count, 1)
    )

    # Average characters per sentence
    avg_chars_per_sentence = char_count / sentence_count

    return [

        word_count,

        char_count,

        sentence_count,

        avg_word_length,

        avg_sentence_length,

        punctuation_count,

        comma_count,

        exclamation_count,

        question_count,

        digit_count,

        uppercase_ratio,

        unique_word_ratio,

        avg_chars_per_sentence

    ]



def build_feature_dataframe(text_series):
    """
    Convert a pandas Series of text into a DataFrame
    containing handcrafted linguistic features.
    """

    features = text_series.apply(extract_features)

    feature_df = pd.DataFrame(
        features.tolist(),
        columns=[

            "word_count",

            "char_count",

            "sentence_count",

            "avg_word_length",

            "avg_sentence_length",

            "punctuation_count",

            "comma_count",

            "exclamation_count",

            "question_count",

            "digit_count",

            "uppercase_ratio",

            "unique_word_ratio",

            "avg_chars_per_sentence"

        ]
    )

    return feature_df



def scale_features(train_df, val_df, test_df):
    """
    Standardize numerical features using only the
    training statistics.
    """

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)

    val_scaled = scaler.transform(val_df)

    test_scaled = scaler.transform(test_df)

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler
    )