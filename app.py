from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from detector.predictor import predict
from humanizer.predictor import humanize_text

from utils.preprocess import clean_text
from utils.text_stats import get_text_statistics
from utils.sentence_predictor import analyze_sentences


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/detector")
def detector():
    return render_template("detector.html")



@app.route("/predict", methods=["POST"])
def prediction():

    data = request.get_json()

    text = data.get("text", "")


    text = clean_text(text)

  
    result = predict(text)


    if not result["success"]:
        return jsonify(result)


    stats = get_text_statistics(text)


    sentence_analysis = analyze_sentences(text)

    result["statistics"] = stats
    result["sentence_analysis"] = sentence_analysis

    return jsonify(result)


@app.route("/humanize", methods=["POST"])
def humanize():

    data = request.get_json()

    text = data.get("text", "")

    text = clean_text(text)

    if len(text.split()) < 30:
        return jsonify({
            "success": False,
            "message": "Please enter at least 30 words."
        })

    output = humanize_text(text)

    return jsonify({
        "success": True,
        "humanized_text": output
    })
    
@app.route("/humanizer")
def humanizer():

    return render_template("humanizer.html")

if __name__ == "__main__":
    app.run(debug=True)