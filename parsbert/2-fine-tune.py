import json

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SMS_LABELED_FILE = "/home/amirhossein/Projects/spam-filter/data/sms_labeled.jsonl"

model_name = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    id2label={0: "ham", 1: "spam"},
    label2id={"ham": 0, "spam": 1}
)

# Replace with your real logged/labeled data
texts = []
labels = []  # 1 = spam, 0 = ham

with open(SMS_LABELED_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            data = json.loads(line)
            
            # FIX 1: Clean newlines and extra spaces
            clean_text = data["text"].replace("\n", " ").replace("\r", " ").strip()
            texts.append(clean_text)
            
            # FIX 2: Convert string labels to integers (0 for ham, 1 for spam)
            label_int = 1 if data["label"].lower() == "spam" else 0
            labels.append(label_int)


# 3. Create Dataset and Split into Train/Validation (Best Practice)
full_dataset = Dataset.from_dict({"text": texts, "label": labels})
# Split 90% for training, 10% for validation to check for overfitting
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
tokenized_datasets = split_dataset.map(tokenize, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


training_args = TrainingArguments(
    output_dir="./parsbert_spam_model",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_strategy="epoch",

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

model.save_pretrained("./parsbert_spam_final")
tokenizer.save_pretrained("./parsbert_spam_final")