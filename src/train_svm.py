import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocess import clean_text
from eval_model import evaluate_model


train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")



print("Cleaning text...")

train["text"] = train["text"].apply(clean_text)
val["text"] = val["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)

y_train = train["label"]
y_val = val["label"]
y_test = test["label"]



print("Building TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=100000,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])
X_test = vectorizer.transform(test["text"])

print("Train Shape :", X_train.shape)
print("Val Shape   :", X_val.shape)
print("Test Shape  :", X_test.shape)


print("\nTraining Linear SVM...\n")

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

model.fit(X_train, y_train)



evaluate_model(model, X_val, y_val, "Validation")

evaluate_model(model, X_test, y_test, "Test")


os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/linear_svm.pkl")
joblib.dump(vectorizer, "models/svm_tfidf.pkl")

print("\nLinear SVM saved successfully!")