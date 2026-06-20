import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import MODEL_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model Loaded Successfully")


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probs, dim=1).item()

    ai_prob = probs[0][1].item()
    human_prob = probs[0][0].item()

    return {
        "prediction": "AI Generated" if prediction == 1 else "Human Written",
        "confidence": round(max(ai_prob, human_prob) * 100, 2),
        "ai_probability": round(ai_prob * 100, 2),
        "human_probability": round(human_prob * 100, 2),
    }