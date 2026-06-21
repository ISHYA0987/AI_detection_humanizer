import re


def clean_text(text):
    """
    Basic preprocessing.
    """

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text