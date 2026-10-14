# Single entry point for the repo. Each model still runs in its own uv
# environment; this file only picks the right one and passes the right paths.
# Models are real file targets, so `make` retrains only when the training
# data or the training script changed.

DATA           ?= data/sms_labeled.jsonl
TEXT           ?= تخفیف ویژه فقط امروز
ANDROID_MODULE ?=
FASTTEXT_VEC   ?= classic/cc.fa.300.vec
N              ?= 200
FORCE          ?=
MODEL          ?= tfidf
IN             ?=

# Changing a preprocessing rule retrains every model and rebuilds the split.
PIPELINE := common/src/spam_common/pipeline.py common/src/spam_common/anonymize.py

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
        export-android export-tflite label augment regex-label anonymize merge extract check model-label

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: DATA=$(DATA)  TEXT=\"...\"  ANDROID_MODULE=path  FASTTEXT_VEC=$(FASTTEXT_VEC)  N=$(N)  IN=path  MODEL=$(MODEL)  FORCE=1"

check: ## Run the unit tests (anonymizer regression cases)
	uv run --project common --with pytest pytest -q common/tests

sync: ## Create/update every uv environment
	for p in common data-prep classic parsbert; do uv sync --project $$p || exit 1; done

# --- Data -------------------------------------------------------------------

$(TRAIN) $(TEST) &: $(DATA) common/src/spam_common/split.py $(PIPELINE)
	$(COMMON) -m spam_common.split $(DATA)

split: $(TRAIN) $(TEST) ## Build the shared, deduplicated train/test split from DATA

merge: ## Add the reviewed file IN to DATA (skips unlabelled and duplicate rows)
	$(if $(IN),,$(error Set IN=path/to/reviewed.jsonl))
	$(COMMON) -m spam_common.merge $(IN) --into $(DATA)

label: ## Open the label viewer GUI (on DATA only if DATA is given)
	$(DATAPREP) data-prep/label_viewer_gui.py $(if $(filter file,$(origin DATA)),,$(DATA))

extract: ## Extract data/sms_dump.txt (adb) -> data/sms_unlabeled.jsonl
	$(DATAPREP) data-prep/extract_sms.py $(if $(FORCE),--force)

anonymize: ## Mask passwords, IDs, cards, phones... in IN -> <IN>_anon.jsonl
	$(if $(IN),,$(error Set IN=path/to/export.jsonl))
	$(COMMON) -m spam_common.anonymize $(IN) $(if $(FORCE),--force)

regex-label: ## First-round label IN with the regex rules -> <IN>_regex.jsonl
	$(if $(IN),,$(error Set IN=path/to/export.jsonl))
	$(DATAPREP) data-prep/regex_label.py $(IN) $(if $(FORCE),--force)

# make model-label: which command and which (up-to-date) model file each MODEL uses.
LABEL_CMD_tfidf    = $(CLASSIC) classic/baseline_tfidf.py label $(TFIDF)
LABEL_CMD_fasttext = $(CLASSIC) classic/baseline_fasttext.py label $(FASTTEXT)
LABEL_CMD_parsbert = $(PARSBERT_PY) parsbert/6-label.py
LABEL_DEP_tfidf    = $(TFIDF)
LABEL_DEP_fasttext = $(FASTTEXT)
LABEL_DEP_parsbert = $(PARSBERT)

model-label: $(LABEL_DEP_$(MODEL)) ## First-round label IN with MODEL=tfidf|fasttext|parsbert -> <IN>_model.jsonl
	$(if $(IN),,$(error Set IN=path/to/export.jsonl))
	$(if $(LABEL_CMD_$(MODEL)),,$(error MODEL must be tfidf, fasttext or parsbert))
	$(LABEL_CMD_$(MODEL)) $(IN) $(if $(FORCE),--force)

augment: ## Generate N synthetic spam rows with a local Ollama model
	$(DATAPREP) data-prep/augment_spam.py $(DATA) data/augmented.jsonl $(N)

# --- All models -------------------------------------------------------------

train-all: train-tfidf train-fasttext train-parsbert ## Train every model (skips up-to-date ones)

test-all: test-tfidf test-fasttext test-parsbert ## Score every model on the test set and compare
	$(COMMON) -m spam_common.metrics

compare: ## Print the comparison table from the last test runs
	$(COMMON) -m spam_common.metrics

# --- TF-IDF + Linear SVM ----------------------------------------------------

$(TFIDF): $(TRAIN) classic/baseline_tfidf.py $(PIPELINE)
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

$(FASTTEXT): $(TRAIN) classic/baseline_fasttext.py $(PIPELINE)
	@mkdir -p $(MODELS)
	$(CLASSIC) classic/baseline_fasttext.py train $(TRAIN) $@

train-fasttext: $(FASTTEXT) ## Train fastText (+ quantized .ftz for Android)

test-fasttext: $(FASTTEXT) $(TEST) ## Score fastText (full and quantized) on the test set
	$(CLASSIC) classic/baseline_fasttext.py test $(FASTTEXT) $(TEST)
	$(CLASSIC) classic/baseline_fasttext.py test $(FASTTEXT:.bin=_quantized.ftz) $(TEST)

predict-fasttext: $(FASTTEXT) ## Classify TEXT with fastText
	$(CLASSIC) classic/baseline_fasttext.py predict $(FASTTEXT) "$(TEXT)"

train-fasttext-pretrained: $(TRAIN) $(TEST) ## Train + score fastText with cc.fa.300 vectors (needs FASTTEXT_VEC); see make compare
	@test -f $(FASTTEXT_VEC) || { echo "Missing $(FASTTEXT_VEC) — see the classic/baseline_fasttext.py docstring"; exit 1; }
	@mkdir -p $(MODELS)
	$(CLASSIC) classic/baseline_fasttext.py train $(TRAIN) $(MODELS)/model-fasttext-cc.bin $(FASTTEXT_VEC)
	$(CLASSIC) classic/baseline_fasttext.py test $(MODELS)/model-fasttext-cc.bin $(TEST)

# --- ParsBERT ---------------------------------------------------------------

$(PARSBERT): $(TRAIN) parsbert/2-fine-tune.py $(PIPELINE)
	@mkdir -p $(MODELS)
	$(PARSBERT_PY) parsbert/2-fine-tune.py

train-parsbert: $(PARSBERT) ## Fine-tune ParsBERT (slow; uses the GPU if available)

test-parsbert: $(PARSBERT) $(TEST) ## Score ParsBERT on the test set
	$(PARSBERT_PY) parsbert/5-evaluate.py

run-parsbert: $(PARSBERT) ## Interactive ParsBERT spam check
	$(PARSBERT_PY) parsbert/3-run.py

export-tflite: $(PARSBERT) ## Convert ParsBERT to models/parsbert_spam.tflite
	$(PARSBERT_PY) parsbert/4-convert-model.py
