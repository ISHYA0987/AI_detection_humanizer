import evaluate
import numpy as np

from dataset import get_tokenizer

tokenizer = get_tokenizer()

rouge = None
bleu = None




def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def compute_metrics(eval_pred):
    global rouge, bleu

    if rouge is None:
        rouge = evaluate.load("rouge")

    if bleu is None:
        bleu = evaluate.load("bleu")

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds,
        decoded_labels
    )

    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )

    prediction_lengths = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in predictions
    ]

    return {
        "bleu": round(bleu_result["bleu"], 4),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
        "gen_len": round(np.mean(prediction_lengths), 2),
    }