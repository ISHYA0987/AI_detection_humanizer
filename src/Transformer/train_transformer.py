import torch

from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from dataset import load_datasets
from metrices import compute_metrics
from config import *


train_dataset, val_dataset, _, tokenizer = load_datasets()

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

training_args = TrainingArguments(

    output_dir=MODEL_SAVE_PATH,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    report_to="none",

    fp16=torch.cuda.is_available(),
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(MODEL_SAVE_PATH)

tokenizer.save_pretrained(MODEL_SAVE_PATH)