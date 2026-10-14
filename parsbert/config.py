"""Settings shared by the ParsBERT scripts, so training, evaluation and export agree."""
from spam_common.data import MODELS_DIR

BASE_MODEL = "HooshvareLab/bert-fa-zwnj-base"  # ParsBERT v3, handles ZWNJ correctly
FINAL_MODEL_DIR = MODELS_DIR / "parsbert_spam_final"
CHECKPOINT_DIR = MODELS_DIR / "parsbert_checkpoints"
TFLITE_FILE = MODELS_DIR / "parsbert_spam.tflite"
MAX_LENGTH = 128  # tokens; longer messages are truncated
