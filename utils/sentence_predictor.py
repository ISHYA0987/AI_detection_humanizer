from detector.predictor import predict_text
import re


def split_sentences(text):

    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if s.strip()
    ]


def analyze_sentences(text):

    sentences = split_sentences(text)

    results = []

    for sentence in sentences:

        prediction, confidence, human_prob, ai_prob = predict_text(sentence)

        if confidence >= 80:

            color = "red"

        elif confidence >= 50:

            color = "orange"

        else:

            color = "green"

        results.append({

            "sentence": sentence,

            "prediction": prediction,

            "confidence": confidence,

            "human_probability": human_prob,

            "ai_probability": ai_prob,

            "color": color

        })

    return results