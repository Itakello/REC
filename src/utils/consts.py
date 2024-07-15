from pathlib import Path

import torch

# from torch.optim import Adam, AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "REC-Test"
CLIP_MODEL = "RN50"

# Default parameters
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_NUM_CANDIDATES = 5

# Model parameters

NUM_HEADS = 8

# Selection modality parameters

BLUR_INTENSITY = 5
LINE_WIDTH = 2

# Hyperarameters

HIGHLIGHTING_METHODS = ["blur", "ellipse", "crop", "blackout", "rectangle"]
SENTENCES_TYPES = ["original_sentences", "comprehensive_sentence", "combined_sentences"]
LEARNING_RATES = [0.001, 0.01, 0.1]
YOLO_VERSIONS = [
    "yolov5nu",
    "yolov5su",
    "yolov5mu",
    "yolov5lu",
    "yolov5xu",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
]
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# Consts
LLM_MODEL = "llama3"

# Paths
DATA_PATH = Path("./data")
PROCESSED_ANNOTATIONS_PATH = DATA_PATH / "processed_annotations"
EMBEDDINGS_PATH = DATA_PATH / "embeddings"
IMAGES_PATH = DATA_PATH / "images"
MODELS_PATH = Path("./models")
LLM_SYSTEM_PROMPT_PATH = Path("./prompts/referential-expression-prompt.txt")
STATS_PATH = Path("./stats")

# URLS
DATASET_URL = "https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq"
