import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# Detector Model
MODEL_PATH = os.path.join(BASE_DIR, "models", "detector")

# Humanizer Model
HUMANIZER_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "humanizer",
    "flan_t5_base"
)


MAX_LENGTH = 512



LABELS = {
    0: "Human",
    1: "AI"
}