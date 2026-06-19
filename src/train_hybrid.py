import os
import joblib
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocess import clean_text
from features import (
    build_feature_dataframe,
    scale_features
)
from eval_model  import evaluate_model

# ==========================================================
# Load Dataset
# ==========================================================

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

# ==========================================================
# Clean Text
# ==========================================================

print("Cleaning text...")

train["text"] = train["text"].apply(clean_text)
val["text"] = val["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)

y_train = train["label"]
y_val = val["label"]
y_test = test["label"]

# ==========================================================
# TF-IDF Features
# ==========================================================

print("Building TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=100000,
    sublinear_tf=True
)

tfidf_train = vectorizer.fit_transform(train["text"])
tfidf_val = vectorizer.transform(val["text"])
tfidf_test = vectorizer.transform(test["text"])

print("TF-IDF Shape")
print("Train :", tfidf_train.shape)
print("Val   :", tfidf_val.shape)
print("Test  :", tfidf_test.shape)

# ==========================================================
# Linguistic Features
# ==========================================================

print("\nExtracting linguistic features...")

train_features = build_feature_dataframe(train["text"])
val_features = build_feature_dataframe(val["text"])
test_features = build_feature_dataframe(test["text"])

train_features, val_features, test_features, scaler = scale_features(
    train_features,
    val_features,
    test_features
)

train_features = csr_matrix(train_features)
val_features = csr_matrix(val_features)
test_features = csr_matrix(test_features)

print("Linguistic Feature Shape")
print("Train :", train_features.shape)
print("Val   :", val_features.shape)
print("Test  :", test_features.shape)

# ==========================================================
# Combine Features
# ==========================================================

print("\nCombining TF-IDF + Linguistic Features...")

X_train = hstack([tfidf_train, train_features])

X_val = hstack([tfidf_val, val_features])

X_test = hstack([tfidf_test, test_features])

print("Combined Shape")
print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)

# ==========================================================
# Train Model
# ==========================================================

print("\nTraining Linear SVM...")

model = LinearSVC(
    C=1,
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# Evaluation
# ==========================================================

evaluate_model(
    model,
    X_val,
    y_val,
    "Validation"
)

evaluate_model(
    model,
    X_test,
    y_test,
    "Test"
)

# ==========================================================
# Save Model
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/hybrid_linear_svm.pkl"
)

joblib.dump(
    vectorizer,
    "models/hybrid_tfidf.pkl"
)

joblib.dump(
    scaler,
    "models/feature_scaler.pkl"
)

print("\nHybrid Linear SVM model saved successfully!")