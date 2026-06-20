import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate a trained classification model.
    """

    y_pred = model.predict(X)

    # Probability scores if available
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_score)

    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
        roc_auc = roc_auc_score(y, y_score)

    else:
        roc_auc = None

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human","AI"],
        yticklabels=["Human","AI"]
    )

    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }