from pathlib import Path

import torch

# from torch.optim import Adam, AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "REC"
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
IOU_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


# Consts
LLM_MODEL = "llama3"

# Paths
DATA_PATH = Path("./data")
MODELS_PATH = Path("./models")
LLM_SYSTEM_PROMPT_PATH = Path("prompts/referential-expression-prompt.txt")

# URLS
DATASET_URL = "https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq"
