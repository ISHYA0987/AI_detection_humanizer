import pandas as pd
import joblib
from preprocess import clean_text
from eval_model import evaluate_model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")

train["text"] = train["text"].apply(clean_text)
val["text"] = val["text"].apply(clean_text)

y_train = train["label"]
y_val = val["label"]

experiments = [

    {
        "name": "Baseline",
        "max_features": 100000,
        "ngram_range": (1,2),
        "min_df": 5,
        "max_df": 0.95,
        "C": 1
    },

    {
        "name": "More Features",
        "max_features": 150000,
        "ngram_range": (1,2),
        "min_df": 5,
        "max_df": 0.95,
        "C": 1
    },

    {
        "name": "Rare Words",
        "max_features": 100000,
        "ngram_range": (1,2),
        "min_df": 2,
        "max_df": 0.95,
        "C": 1
    },

    {
        "name": "Trigrams",
        "max_features": 100000,
        "ngram_range": (1,3),
        "min_df": 5,
        "max_df": 0.95,
        "C": 1
    },

    {
        "name": "Less Regularization",
        "max_features": 100000,
        "ngram_range": (1,2),
        "min_df": 5,
        "max_df": 0.95,
        "C": 10
    }

]

results = []

best_f1 = 0

best_model = None
best_vectorizer = None


for exp in experiments:

    print("\n" + "="*70)
    print(exp["name"])
    print("="*70)

    vectorizer = TfidfVectorizer(

        lowercase=True,

        max_features=exp["max_features"],

        ngram_range=exp["ngram_range"],

        min_df=exp["min_df"],

        max_df=exp["max_df"],

        sublinear_tf=True

    )

    X_train = vectorizer.fit_transform(train["text"])

    X_val = vectorizer.transform(val["text"])

    model = LogisticRegression(

        C=exp["C"],

        class_weight="balanced",

        max_iter=1000,

        random_state=42

    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_val,
        y_val,
        exp["name"]
    )

    metrics["Experiment"] = exp["name"]

    results.append(metrics)

    if metrics["f1"] > best_f1:

        best_f1 = metrics["f1"]

        best_model = model

        best_vectorizer = vectorizer


results_df = pd.DataFrame(results)

print("\n")
print(results_df.sort_values("f1", ascending=False))


joblib.dump(best_model, "models/best_logistic.pkl")

joblib.dump(best_vectorizer, "models/best_tfidf.pkl")

print("\nBest model saved.")