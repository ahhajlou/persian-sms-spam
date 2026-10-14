from transformers import pipeline
from spam_common.data import clean_text
from spam_common.pipeline import preprocess

from config import FINAL_MODEL_DIR, MAX_LENGTH

classifier = pipeline("text-classification", model=str(FINAL_MODEL_DIR), tokenizer=str(FINAL_MODEL_DIR))

def check_spam(text, threshold=0.5):
    result = classifier(clean_text(preprocess(text)), truncation=True, max_length=MAX_LENGTH)[0]
    label, score = result["label"], result["score"]
    
    # Normalize to "probability of spam" regardless of which label won
    spam_prob = score if label == "spam" else (1 - score)
    
    is_spam = spam_prob >= threshold
    return spam_prob, is_spam

if __name__ == "__main__":
    print("Enter text to check (Ctrl+C to quit):")
    while True:
        text = input("> ")
        spam_prob, is_spam = check_spam(text)
        verdict = "SPAM" if is_spam else "HAM"
        print(f"{verdict} (spam probability: {spam_prob:.1%})")