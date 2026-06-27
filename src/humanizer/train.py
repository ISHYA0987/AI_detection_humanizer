import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from config import *
from dataset import (
    load_datasets,
    get_data_collator,
    get_tokenizer,
)


def main():

    set_seed(SEED)

    print(f"Using Device: {DEVICE}")

    tokenizer = get_tokenizer()

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.config.use_cache = False

    train_dataset, valid_dataset, test_dataset = load_datasets()

    data_collator = get_data_collator(model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        tokenizer=tokenizer,

        data_collator=data_collator,
    )

    print("Starting training...")

    trainer.train()

    trainer.save_model(str(OUTPUT_DIR))

    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()