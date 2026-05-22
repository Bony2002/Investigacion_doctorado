#!/usr/bin/env python3
"""Export image–text pairs to JSONL for SigLIP2 finetuning (bv:jsonl dataset).

Each line::

    {"image": "/abs/path/img.jpg", "captions/text": "your caption here"}

Example — CIFAR-10 pickles to train/val JSONL::

    python tools/export_jsonl.py cifar \\
        --root datasets/cifar-10-batches-py \\
        --out-dir data
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _unpickle(path: Path):
  with open(path, "rb") as f:
    return pickle.load(f, encoding="bytes")


def export_cifar(root: Path, out_dir: Path, prompt: str) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  root = Path(root)
  meta = _unpickle(root / "batches.meta")
  names = [n.decode() for n in meta[b"label_names"]]

  def write_split(batch_files, out_path):
    with open(out_path, "w", encoding="utf-8") as fp:
      for bf in batch_files:
        batch = _unpickle(bf)
        for i, label in enumerate(batch[b"labels"]):
          # Store relative paths; fopen_keys resolves from JSONL location.
          rel = f"cifar/{bf.stem}_{i:05d}.png"
          caption = prompt.format(class_name=names[label])
          fp.write(json.dumps({
              "image": rel,
              "captions/text": caption,
          }) + "\n")

  train_batches = [root / f"data_batch_{i}" for i in range(1, 6)]
  write_split(train_batches, out_dir / "train.jsonl")
  write_split([root / "test_batch"], out_dir / "val.jsonl")
  print(f"Wrote {out_dir / 'train.jsonl'} and {out_dir / 'val.jsonl'}")
  print("Note: jsonl mode expects image files on disk. For pickle-only CIFAR,")
  print("      use DATASET_MODE='cifar10' in finetune_config.py instead.")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)

  cifar = sub.add_parser("cifar", help="Export CIFAR-10 caption JSONL (paths only).")
  cifar.add_argument("--root", type=Path, required=True)
  cifar.add_argument("--out-dir", type=Path, default=Path("data"))
  cifar.add_argument("--prompt", default="a photo of a {class_name}")

  args = parser.parse_args()
  if args.cmd == "cifar":
    export_cifar(args.root, args.out_dir, args.prompt)


if __name__ == "__main__":
  main()
