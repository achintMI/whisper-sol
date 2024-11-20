from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "training_data"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
KNN_MODEL_PATH = MODEL_DIR / "knn_model.json"
