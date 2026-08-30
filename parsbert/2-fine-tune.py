from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

model_name = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    id2label={0: "ham", 1: "spam"},
    label2id={"ham": 0, "spam": 1}
)

# Replace with your real logged/labeled data
texts = [
    "تبریک! شما برنده یک میلیون تومان شدید، برای دریافت کلیک کنید",
    "سلام، ساعت ۵ همدیگه رو می‌بینیم؟",
    "کد تایید شما: 48213. این کد رو با کسی به اشتراک نذارید",
    "فروش ویژه امروز! تخفیف ۷۰ درصدی فقط امروز",
]
labels = [1, 0, 0, 1]  # 1 = spam, 0 = ham

dataset = Dataset.from_dict({"text": texts, "label": labels})

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./parsbert_spam_model",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("./parsbert_spam_final")
tokenizer.save_pretrained("./parsbert_spam_final")