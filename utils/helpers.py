def get_prediction_label(label):

    if label == 1:

        return "AI"

    return "Human"


def probability_to_percent(prob):

    return round(prob * 100, 2)


def confidence_level(confidence):

    if confidence >= 90:

        return "Very High"

    elif confidence >= 75:

        return "High"

    elif confidence >= 60:

        return "Moderate"

    else:

        return "Low"