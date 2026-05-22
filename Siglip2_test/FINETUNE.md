# SigLIP2 finetuning

## Pipeline test on CIFAR-10 (start here)

No JSONL needed. In `finetune_config.py`:

```python
TRAINING_MODE = "legacy_class"
```

Your data is already at `datasets/cifar-10-batches-py`. Run **`finetune_siglip2.ipynb`** cells **1 → 6 in order** (do not skip cell 5 — training).

- 50k train images → synthetic captions like `"a photo of a cat"` (random template per step)
- **3000 steps** (~1 epoch), checkpoints under `workdirs/siglip2_cifar_test/`
- `FRESH_TRAIN = True` wipes old checkpoints so each run starts from the base model
- Same trainer + text tower as real caption data

This checks: checkpoint download → data → preprocess → sigmoid loss → save.

After training, run **cell 6** in `finetune_siglip2.ipynb` to print **base vs finetuned** zero-shot accuracy on CIFAR test images. By default this uses the **full 10k test set** (`EVAL_MAX_IMAGES = None` in `finetune_config.py`); set `EVAL_MAX_IMAGES = 2000` for a faster smoke test.

---

## Text-centric finetuning (real captions)

Train SigLIP2 on **real language** paired with images — not as a 10-class classifier.

## Text problem vs classification

| | Classification-style (old) | **Text-centric (default)** |
|---|---------------------------|----------------------------|
| Caption | `"a photo of a {class}"` from label | Your own sentence per image |
| What learns | Text templates tied to 10 words | **Text tower** maps language → image space |
| Image tower | Frozen or full | **Frozen** by default (`TRAINING_MODE = "text"`) |
| Data | CIFAR pickles | **`data/train.jsonl`** |

Example **good** captions:

```text
"the green line peaks in March while the blue bars stay flat"
"man in red jacket standing left of the truck"
```

Example **avoid** (unless you really want class-style prompts):

```text
"a photo of a cat"
```

## Quick start

1. Copy the example and edit:

   ```bash
   cp data/train.jsonl.example data/train.jsonl
   # edit paths + captions
   ```

2. In `finetune_config.py` keep:

   ```python
   TRAINING_MODE = "text"
   TEXT_JSONL_TRAIN = PROJECT_ROOT / "data" / "train.jsonl"
   ```

3. Run **`finetune_siglip2.ipynb`**.

## JSONL format

One JSON object per line. See [`data/README.md`](data/README.md).

```json
{"image": "/path/to/img.jpg", "text": "your natural language description"}
{"image": "rel/path.png", "captions": ["caption A", "caption B"]}
```

Multiple strings under `"captions"` → one is sampled randomly each step.

## Training modes (`finetune_config.py`)

| `TRAINING_MODE` | Meaning |
|-----------------|--------|
| **`text`** | JSONL captions; freeze image tower (default) |
| **`full`** | Same JSONL; train image + text towers |
| **`legacy_class`** | CIFAR + `"a photo of a …"` templates (quick test only) |

## Weights

| | Location |
|---|----------|
| **Base model** | `/tmp/siglip2_b16_224.npz` (from `gs://big_vision/siglip2/…`) |
| **After finetune** | `workdirs/siglip2_text_finetune/checkpoint.bv-LAST` |

Base `.npz` is never overwritten. Finetuned weights use Orbax format under `WORKDIR`.

## What gets trained in text mode

```text
Frozen:  image tower (ViT) — pretrained vision features stay fixed
Trained: text tower (Transformer) + temperature t + bias b
```

The model learns: *"this kind of sentence should align with this image embedding."*

## Switching datasets later

Only change **`data/train.jsonl`** (and optionally `IMAGE_ROOT`). No code changes.

For RefCOCO / COCO / exports: write a script that emits the same JSONL schema, or add a new `bv:…` dataset module that yields `captions/text`.

## CLI

```bash
conda activate siglip2
cd big_vision

export SIGLIP_DATASET_MODE=text
export SIGLIP_JSONL_TRAIN=/path/to/train.jsonl
export SIGLIP_CKPT_PATH=/tmp/siglip2_b16_224.npz
export SIGLIP_FREEZE_IMAGE=1

python -m big_vision.trainers.proj.image_text.siglip \
  --config big_vision/configs/proj/image_text/siglip2_finetune_local.py:runlocal,freeze_image=True \
  --workdir ../workdirs/siglip2_text_finetune
```

## Files

| File | Role |
|------|------|
| `finetune_config.py` | `TRAINING_MODE`, JSONL paths, hyperparameters |
| `finetune_siglip2.ipynb` | Run training |
| `big_vision/.../image_text_jsonl.py` | Loads caption JSONL |
| `big_vision/.../siglip2_finetune_local.py` | Trainer config |
