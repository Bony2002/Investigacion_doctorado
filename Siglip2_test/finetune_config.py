"""User settings for SigLIP2 finetuning.

Edit this file when switching datasets or training stage.

Training modes
--------------
**legacy_class** (default here) — smoke-test the pipeline on CIFAR-10 you already
have. Synthetic captions from class names; no JSONL required.

**text** — real work: ``data/train.jsonl`` with natural-language captions.

**full** — same as text but also finetunes the image tower.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BIG_VISION_ROOT = PROJECT_ROOT / "big_vision"

# ---------------------------------------------------------------------------
# Training mode — set legacy_class to test the pipeline on CIFAR-10
# ---------------------------------------------------------------------------
TRAINING_MODE = "legacy_class"   # "legacy_class" | "text" | "full"

# ---------------------------------------------------------------------------
# CIFAR-10 pipeline test (TRAINING_MODE == "legacy_class")
# ---------------------------------------------------------------------------
CIFAR_ROOT = PROJECT_ROOT / "datasets" / "cifar-10-batches-py"
# One or more templates; one is chosen at random per image each epoch.
CIFAR_PROMPTS = (
    "a photo of a {class_name}",
    "a picture of a {class_name}",
    "an image showing a {class_name}",
)

# ---------------------------------------------------------------------------
# Real captions (TRAINING_MODE == "text" or "full")
# ---------------------------------------------------------------------------
TEXT_JSONL_TRAIN = PROJECT_ROOT / "data" / "train.jsonl"
TEXT_JSONL_VAL = PROJECT_ROOT / "data" / "val.jsonl"
IMAGE_ROOT = PROJECT_ROOT / "data" / "images"

# ---------------------------------------------------------------------------
# Model (224 fits 8 GB VRAM for training)
# ---------------------------------------------------------------------------
VARIANT = "B/16"
RES = 224
CKPT_DIR = Path("/tmp")

# ---------------------------------------------------------------------------
# Training — CIFAR pipeline test uses full train set (50k); steps ≈ 1 epoch
# ---------------------------------------------------------------------------
WORKDIR = PROJECT_ROOT / "workdirs" / "siglip2_cifar_test"
BATCH_SIZE = 16
# 50k / batch 16 ≈ 3125 steps per epoch; 3000 is a meaningful finetune test.
TOTAL_STEPS = 3000 if TRAINING_MODE == "legacy_class" else 2_000
LOG_STEPS = 25
CKPT_STEPS = 500
SEQLEN = 64

FREEZE_IMAGE_TOWER = TRAINING_MODE != "full"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# Delete old checkpoints and train from scratch (set False to resume).
FRESH_TRAIN = True

# CIFAR zero-shot eval after training (None = all 10k test images)
EVAL_MAX_IMAGES = None  # full CIFAR-10 test set (10k images)

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def checkpoint_name(variant: str = VARIANT, res: int = RES) -> str:
  return f"siglip2_{variant.lower().replace('/', '')}_{res}.npz"


def checkpoint_path(variant: str = VARIANT, res: int = RES) -> Path:
  return CKPT_DIR / checkpoint_name(variant, res)


def text_variant(variant: str = VARIANT) -> str:
  v, _ = variant.split("/")
  return "So400m" if v == "g-opt" else v


def embed_dim(variant: str = VARIANT) -> int:
  v, _ = variant.split("/")
  return {"B": 768, "L": 1024, "So400m": 1152, "g-opt": 1536}[v]


def dataset_mode_for_trainer() -> str:
  if TRAINING_MODE == "legacy_class":
    return "cifar10"
  return "text"
