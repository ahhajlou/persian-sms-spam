# Single entry point for the repo. Each model still runs in its own uv
# environment; this file only picks the right one and passes the right paths.
# Models are real file targets, so `make` retrains only when the training
# data or the training script changed.

DATA           ?= data/sms_labeled.jsonl
TEXT           ?= تخفیف ویژه فقط امروز
ANDROID_MODULE ?=
FASTTEXT_VEC   ?= classic/cc.fa.300.vec
N              ?= 200

TRAIN  := data/train.jsonl
TEST   := data/test.jsonl
MODELS := models

TFIDF    := $(MODELS)/model-tfidf.joblib
FASTTEXT := $(MODELS)/model-fasttext.bin
PARSBERT := $(MODELS)/parsbert_spam_final/model.safetensors

COMMON      := uv run --project common python
CLASSIC     := uv run --project classic python
PARSBERT_PY := uv run --project parsbert python
DATAPREP    := uv run --project data-prep python

.DEFAULT_GOAL := help
.PHONY: help sync split train-all test-all compare \
        train-tfidf test-tfidf predict-tfidf \
        train-fasttext test-fasttext predict-fasttext train-fasttext-pretrained \
        train-parsbert test-parsbert run-parsbert \
        export-android export-tflite label augment

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: DATA=$(DATA)  TEXT=\"...\"  ANDROID_MODULE=path  FASTTEXT_VEC=$(FASTTEXT_VEC)  N=$(N)"

sync: ## Create/update every uv environment
	for p in common data-prep classic parsbert; do uv sync --project $$p || exit 1; done

# --- Data -------------------------------------------------------------------

$(TRAIN) $(TEST) &: $(DATA) common/src/spam_common/split.py
	$(COMMON) -m spam_common.split $(DATA)

split: $(TRAIN) $(TEST) ## Build the shared, deduplicated train/test split from DATA

label: ## Open the label viewer GUI
	$(DATAPREP) data-prep/label_viewer_gui.py

augment: ## Generate N synthetic spam rows with a local Ollama model
	$(DATAPREP) data-prep/augment_spam.py $(DATA) data/augmented.jsonl $(N)

# --- All models -------------------------------------------------------------

train-all: train-tfidf train-fasttext train-parsbert ## Train every model (skips up-to-date ones)

test-all: test-tfidf test-fasttext test-parsbert ## Score every model on the test set and compare
	$(COMMON) -m spam_common.metrics

compare: ## Print the comparison table from the last test runs
	$(COMMON) -m spam_common.metrics

# --- TF-IDF + Linear SVM ----------------------------------------------------

$(TFIDF): $(TRAIN) classic/baseline_tfidf.py
	@mkdir -p $(MODELS)
	$(CLASSIC) classic/baseline_tfidf.py train $(TRAIN) $@

train-tfidf: $(TFIDF) ## Train TF-IDF + Linear SVM

test-tfidf: $(TFIDF) $(TEST) ## Score TF-IDF on the test set
	$(CLASSIC) classic/baseline_tfidf.py test $(TFIDF) $(TEST)

predict-tfidf: $(TFIDF) ## Classify TEXT with TF-IDF
	$(CLASSIC) classic/baseline_tfidf.py predict $(TFIDF) "$(TEXT)"

export-android: $(TFIDF) ## Export TF-IDF + hazm assets into ANDROID_MODULE
	$(if $(ANDROID_MODULE),,$(error Set ANDROID_MODULE=path/to/core/ml))
	$(CLASSIC) classic/export_android_assets.py $(ANDROID_MODULE) --model $(TFIDF)

# --- fastText ---------------------------------------------------------------

$(FASTTEXT): $(TRAIN) classic/baseline_fasttext.py
	@mkdir -p $(MODELS)
	$(CLASSIC) classic/baseline_fasttext.py train $(TRAIN) $@

train-fasttext: $(FASTTEXT) ## Train fastText (+ quantized .ftz for Android)

test-fasttext: $(FASTTEXT) $(TEST) ## Score fastText (full and quantized) on the test set
	$(CLASSIC) classic/baseline_fasttext.py test $(FASTTEXT) $(TEST)
	$(CLASSIC) classic/baseline_fasttext.py test $(FASTTEXT:.bin=_quantized.ftz) $(TEST)

predict-fasttext: $(FASTTEXT) ## Classify TEXT with fastText
	$(CLASSIC) classic/baseline_fasttext.py predict $(FASTTEXT) "$(TEXT)"

train-fasttext-pretrained: $(TRAIN) $(TEST) ## Compare fastText with/without cc.fa.300 vectors (needs FASTTEXT_VEC)
	@test -f $(FASTTEXT_VEC) || { echo "Missing $(FASTTEXT_VEC) — see classic/fasttext_with_pretrained_vector.py docstring"; exit 1; }
	@mkdir -p $(MODELS)
	$(CLASSIC) classic/fasttext_with_pretrained_vector.py train $(TRAIN) $(MODELS)/fasttext-cc.bin $(FASTTEXT_VEC)

# --- ParsBERT ---------------------------------------------------------------

$(PARSBERT): $(TRAIN) parsbert/2-fine-tune.py
	@mkdir -p $(MODELS)
	$(PARSBERT_PY) parsbert/2-fine-tune.py

train-parsbert: $(PARSBERT) ## Fine-tune ParsBERT (slow; uses the GPU if available)

test-parsbert: $(PARSBERT) $(TEST) ## Score ParsBERT on the test set
	$(PARSBERT_PY) parsbert/5-evaluate.py

run-parsbert: $(PARSBERT) ## Interactive ParsBERT spam check
	$(PARSBERT_PY) parsbert/3-run.py

export-tflite: $(PARSBERT) ## Convert ParsBERT to models/parsbert_spam.tflite
	$(PARSBERT_PY) parsbert/4-convert-model.py
