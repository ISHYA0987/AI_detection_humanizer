import re


def get_text_statistics(text):
    """
    Compute useful text statistics.
    """

    words = text.split()

    word_count = len(words)

    character_count = len(text)

    sentence_count = len(
        re.findall(r"[.!?]+", text)
    )

    paragraph_count = len(
        [p for p in text.split("\n") if p.strip()]
    )

    reading_time = round(word_count / 200, 2)

    avg_sentence_length = round(
        word_count / max(sentence_count, 1),
        2
    )

    return {

        "words": word_count,

        "characters": character_count,

        "sentences": sentence_count,

        "paragraphs": paragraph_count,

        "reading_time": reading_time,

        "avg_sentence": avg_sentence_length

    }