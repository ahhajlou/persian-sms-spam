from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from spam_common.data import ID2LABEL, LABEL2ID, MODELS_DIR, TRAIN_FILE, clean_text, load_texts_labels

FINAL_MODEL_DIR = MODELS_DIR / "parsbert_spam_final"

model_name = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(ID2LABEL),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Train only on the shared training split; data/test.jsonl stays unseen
# until 5-evaluate.py, so the score is comparable with the other models.
texts, labels = load_texts_labels(TRAIN_FILE)
texts = [clean_text(t) for t in texts]
labels = [LABEL2ID[label] for label in labels]


# Create Dataset and split off a validation set from the training data
full_dataset = Dataset.from_dict({"text": texts, "label": labels})
# Split 90% for training, 10% for validation to pick the best checkpoint
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
tokenized_datasets = split_dataset.map(tokenize, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


training_args = TrainingArguments(
    output_dir=str(MODELS_DIR / "parsbert_checkpoints"),
    num_train_epochs=4,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2, # Keep only the best + latest checkpoint

    per_device_eval_batch_size=8,
    eval_strategy="epoch", # Evaluate at the end of each epoch
    load_best_model_at_end=True, # Automatically keeps the best performing model
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"] # Validation set
)
trainer.train()

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)