import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from config import HUMANIZER_MODEL_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading Humanizer Model...")

tokenizer = AutoTokenizer.from_pretrained(
    HUMANIZER_MODEL_PATH,
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    HUMANIZER_MODEL_PATH
)

model.to(device)
model.eval()

print("Humanizer Loaded Successfully")


def humanize_text(text):

    prompt = f"Humanize: {text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    output = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return output