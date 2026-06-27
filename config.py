import os

# ==========================================
# Base Directory
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ==========================================
# Model Paths
# ==========================================

# Detector Model
MODEL_PATH = os.path.join(BASE_DIR, "models", "detector")

# Humanizer Model
HUMANIZER_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "humanizer",
    "flan_t5_base"
)

# ==========================================
# Model Parameters
# ==========================================

MAX_LENGTH = 512

# ==========================================
# Labels
# ==========================================

LABELS = {
    0: "Human",
    1: "AI"
}