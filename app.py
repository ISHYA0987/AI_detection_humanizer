from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from detector.predictor import predict

from utils.preprocess import clean_text
from utils.text_stats import get_text_statistics
from utils.sentence_predictor import analyze_sentences


app = Flask(__name__)


# ===============================
# Home
# ===============================

@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# Detector Page
# ===============================

@app.route("/detector")
def detector():
    return render_template("detector.html")


# ===============================
# Prediction API
# ===============================

@app.route("/predict", methods=["POST"])
def prediction():

    data = request.get_json()

    text = data.get("text", "")

    # Clean text
    text = clean_text(text)

    # Run prediction
    result = predict(text)

    # If validation failed (less than 30 words)
    if not result["success"]:
        return jsonify(result)

    # Compute statistics
    stats = get_text_statistics(text)

    # Sentence-level analysis
    sentence_analysis = analyze_sentences(text)

    # Merge everything into one response
    result["statistics"] = stats
    result["sentence_analysis"] = sentence_analysis

    return jsonify(result)


# ===============================
# Run Flask
# ===============================

if __name__ == "__main__":
    app.run(debug=True)