import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)

model.to(device)
model.eval()


LABELS = {
    0: "Human",
    1: "AI"
}


def predict(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()

    print("\n==============================")
    print("Logits       :", logits.cpu().numpy())
    print("Probabilities")
    print(f"Human : {probs[0][0].item():.4f}")
    print(f"AI    : {probs[0][1].item():.4f}")
    print("==============================")

    return LABELS[pred], confidence


if __name__ == "__main__":

    while True:

        text = input("\nEnter text (or 'exit'): ")

        if text.lower() == "exit":
            break

        prediction, confidence = predict(text)

        print(f"\nPrediction : {prediction}")
        print(f"Confidence : {confidence:.4f}")