from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"

TRAIN_FILE = SPLITS_DIR / "train.csv"
VALID_FILE = SPLITS_DIR / "valid.csv"
TEST_FILE = SPLITS_DIR / "test.csv"

MODEL_NAME = "google/flan-t5-base"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 5

WARMUP_RATIO = 0.06

BEAM_SIZE = 4

SEED = 42

OUTPUT_DIR = BASE_DIR / "models" / "flan_t5_base"

LOGGING_DIR = BASE_DIR / "logs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_TOTAL_LIMIT = 2

LOGGING_STEPS = 100

EVAL_STRATEGY = "epoch"

SAVE_STRATEGY = "epoch"

PREDICT_WITH_GENERATE = True

LOAD_BEST_MODEL_AT_END = True

METRIC_FOR_BEST_MODEL = "rougeL"

GREATER_IS_BETTER = True

NUM_BEAMS = 2