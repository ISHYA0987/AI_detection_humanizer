import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

text = "Humanize the following AI-generated answer.\n\nQuestion:\nWhat is AI?\n\nAI Answer:\nArtificial intelligence is the simulation of human intelligence."

target = "Artificial intelligence refers to computer systems designed to perform tasks that usually require human intelligence."

inputs = tokenizer(text, return_tensors="pt").to(device)

labels = tokenizer(text_target=target, return_tensors="pt").input_ids.to(device)

outputs = model(**inputs, labels=labels)

print("Loss:", outputs.loss.item())

outputs.loss.backward()

grad = model.shared.weight.grad

print("Gradient None :", grad is None)
print("Gradient Mean :", grad.abs().mean().item())