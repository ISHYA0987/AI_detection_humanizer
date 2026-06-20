import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels,
    )

    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
    )

    recall = recall_metric.compute(
        predictions=predictions,
        references=labels,
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
    )

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }