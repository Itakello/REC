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
