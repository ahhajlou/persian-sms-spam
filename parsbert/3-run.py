# from transformers import pipeline

# text = "سلام"
# text = "تخفیف"
# # text = "کد تخفیف ویژه شما آماده است، همین الان دریافت کنید"
# classifier = pipeline("text-classification", model="./parsbert_spam_final", tokenizer="./parsbert_spam_final")
# result = classifier(text)
# print(f"{text=}")
# print(result)

from transformers import pipeline
from spam_common.data import MODELS_DIR, clean_text
from spam_common.pipeline import preprocess

MODEL_DIR = str(MODELS_DIR / "parsbert_spam_final")
classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)

def check_spam(text, threshold=0.5):
    result = classifier(clean_text(preprocess(text)))[0]
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