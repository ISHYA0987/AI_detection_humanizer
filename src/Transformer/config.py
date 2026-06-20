"""
Configuration file for Transformer-based AI Text Detection
Compatible with:
- PyTorch 2.12+
- Transformers 4.57+
"""

MODEL_NAME = "roberta-base"

# Data
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

# Output
MODEL_SAVE_PATH = "models/roberta_detector"

# Labels
NUM_LABELS = 2

# Tokenization
MAX_LENGTH = 256

# Training
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3

# Misc
SEED = 42