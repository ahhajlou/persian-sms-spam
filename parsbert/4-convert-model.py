import litert_torch
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import FINAL_MODEL_DIR, MAX_LENGTH, TFLITE_FILE

model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
model.eval()

# Build a sample input matching your fine-tuning max_length
sample_text = "این یک متن نمونه است"
inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
sample_inputs = (inputs["input_ids"], inputs["attention_mask"])

edge_model = litert_torch.convert(model, sample_inputs)

# Sanity check: compare PyTorch output vs converted output
import numpy as np

with torch.no_grad():
    torch_output = model(*sample_inputs).logits
edge_output = edge_model(*sample_inputs)

if isinstance(edge_output, dict):
    edge_logits = edge_output["logits"]
else:
    edge_logits = edge_output

if np.allclose(torch_output.numpy(), edge_logits, atol=1e-4):
    print("Conversion verified — outputs match")
else:
    print("Warning: outputs diverge, inspect before deploying")
    print("Torch:", torch_output.numpy())
    print("Edge:", edge_logits)

edge_model.export(str(TFLITE_FILE))