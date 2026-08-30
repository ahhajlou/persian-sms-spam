from transformers import AutoTokenizer, AutoModel
import torch

model_name = "HooshvareLab/bert-fa-zwnj-base"  # ParsBERT v3, handles ZWNJ correctly
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "تبریک میگم! شما برنده جایزه شدید، همین الان دریافت کنید"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("//=========//")
print("//=========//")
print("//=========//")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print(outputs.last_hidden_state.shape)