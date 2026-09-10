"""
Generate synthetic spam SMS examples using a local Ollama model (e.g.
gemma3:4b), by repeatedly sampling small batches of real spam examples
as style reference — never feeding the whole dataset into one prompt.

Prereqs:
    ollama pull gemma3:4b
    ollama serve   (usually starts automatically after install)
    pip install requests

Usage:
    python augment_spam_local_llm.py data.jsonl augmented.jsonl 200
    (generates ~200 new synthetic spam rows into augmented.jsonl)
"""
import sys
import json
import random
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma-4-e4b-local:latest"  # must match `ollama list` output exactly
SEED_BATCH_SIZE = 6       # how many real examples shown per call
GENERATE_PER_CALL = 5     # how many new examples requested per call

PROMPT_TEMPLATE = """تو در حال تولید نمونه‌های آموزشی برای یک فیلتر اسپم پیامک فارسی هستی.
در زیر چند نمونه واقعی پیامک اسپم فارسی آمده:

{examples}

حالا {n} پیامک اسپم جدید و متفاوت با همان سبک و لحن بالا بنویس. در پیامک‌های جدید:
- از نام برند، درصد تخفیف، و مبلغ‌های متفاوت استفاده کن
- به‌جای لینک‌های واقعی از "example.ir/xyz" به‌عنوان جای‌گذار استفاده کن
- سبک و لحن اسپم بودن پیام حفظ شود ولی جملات عیناً کپی نشوند

فقط خروجی را به‌صورت یک آرایه JSON از رشته‌ها برگردان، بدون هیچ توضیح اضافه. مثال فرمت خروجی:
["پیام اول", "پیام دوم"]
"""


def load_spam_examples(path):
    spam = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") == "spam":
                spam.append(row["text"])
    return spam


def call_ollama(prompt):
    print("    (calling model...)", flush=True)
    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9},
            # Keep the model resident in memory between calls instead of
            # unloading after Ollama's default ~5 min idle window — avoids
            # repeatedly paying the multi-GB cold-load cost across your
            # ~40 batch calls.
            "keep_alive": "30m",
        },
        timeout=300,  # first call includes model load time (can be slow
                      # on CPU-only / memory-constrained setups); 120s was
                      # too tight for that, not just generation itself
    )
    print(f"    (done in {time.time() - t0:.1f}s)", flush=True)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_json_array(text):
    # Models sometimes wrap JSON in markdown fences or add stray text —
    # try to locate the array bounds defensively rather than assuming
    # response is clean JSON.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def generate_batch(spam_pool):
    seeds = random.sample(spam_pool, min(SEED_BATCH_SIZE, len(spam_pool)))
    examples_block = "\n".join(f"- {s}" for s in seeds)
    prompt = PROMPT_TEMPLATE.format(examples=examples_block, n=GENERATE_PER_CALL)
    raw = call_ollama(prompt)
    return parse_json_array(raw)


def main(data_path, out_path, target_count):
    spam_pool = load_spam_examples(data_path)
    print(f"Loaded {len(spam_pool)} real spam examples as seed pool")

    generated = []
    attempts = 0
    while len(generated) < target_count and attempts < target_count * 3:
        attempts += 1
        batch = generate_batch(spam_pool)
        # Basic sanity filter: skip empty/too-short/duplicate outputs
        for text in batch:
            text = text.strip()
            if text and len(text) > 10 and text not in generated:
                generated.append(text)
        print(f"  generated so far: {len(generated)}/{target_count}  (call {attempts})")

    with open(out_path, "w", encoding="utf-8") as f:
        for text in generated[:target_count]:
            f.write(json.dumps(
                {"address": "synthetic", "text": text, "label": "spam", "synthetic": True},
                ensure_ascii=False
            ) + "\n")

    print(f"Wrote {min(len(generated), target_count)} synthetic examples to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
