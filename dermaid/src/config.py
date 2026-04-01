import torch
from pathlib import Path

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

CLASS_COUNTS = {
    'nv': 6705,
    'mel': 1113,
    'bkl': 1099,
    'bcc': 514,
    'akiec': 327,
    'vasc': 142,
    'df': 115
}

SEVERITY_MAP = {
    'nv': 'Low Risk',
    'df': 'Low Risk',
    'bkl': 'Low Risk',
    'vasc': 'Refer Soon',
    'akiec': 'Refer Soon',
    'bcc': 'Refer Immediately',
    'mel': 'Refer Immediately'
}

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR_HEADS = 1e-3
LR_BACKBONE = 3e-5
LR_HEADS_STAGE2 = 1e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.4
DROPOUT_RATE = 0.3

CONFIDENCE_THRESHOLD = 0.60
UNCERTAINTY_THRESHOLD = 0.15
MC_DROPOUT_PASSES = 20

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
EXPORT_DIR = BASE_DIR / 'export'
