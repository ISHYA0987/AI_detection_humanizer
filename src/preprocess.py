import re


def clean_text(text):
    """
    Basic text preprocessing for AI text detection.

    We intentionally preserve punctuation, stopwords,
    capitalization patterns (except lowercase for TF-IDF),
    and sentence structure because these contain stylistic
    information useful for detecting AI-generated text.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()