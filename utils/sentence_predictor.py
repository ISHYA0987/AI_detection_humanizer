from detector.predictor import predict
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

        result = predict(sentence)

        # Skip very short sentences if validation fails
        if not result["success"]:
            continue

        prediction = result["prediction"]
        confidence = result["confidence"]
        human_prob = result["human_probability"]
        ai_prob = result["ai_probability"]

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