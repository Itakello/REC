from pathlib import Path

import torch

# from torch.optim import Adam, AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "REC-June"

# Default parameters
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_NUM_CANDIDATES = 5

# Model parameters

NUM_HEADS = 8

# Selection modality parameters

BLUR_INTENSITY = 5
LINE_WIDTH = 4

# Hyperarameters

SELECTION_MODALITIES = ["blur", "ellipse", "crop", "blackout", "rectangle"]
SENTENCES_TYPES = ["default", "new", "combined"]
LEARNING_RATES = [0.001, 0.01, 0.1]

# Consts
LLM_MODEL = "llama3"

# Paths
DATA_PATH = Path("./data")
IMAGES_PATH = Path("./data/images")
ANNOTATIONS_PATH = Path("./data/annotations")
LLM_SYSTEM_PROMPT_PATH = Path("prompts/referential-expression-prompt.txt")

# URLS
DATASET_URL = "https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq"

# Hyperparameters
YOLO_VERSIONS = [
    "yolov5nu.pt",
    "yolov5su.pt",
    "yolov5mu.pt",
    "yolov5lu.pt",
    "yolov5xu.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]
IOU_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
