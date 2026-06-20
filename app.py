from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from detector.predictor import predict

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

    text = data["text"]

    result = predict(text)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)