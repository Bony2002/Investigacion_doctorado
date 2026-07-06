"""
Step 1 – Inference (siglip2 env)
================================
Loads the overfitting checkpoint, runs a forward pass on the target image,
and saves the predicted points + colors to a .pt file for rendering.

Usage
-----
    python infer_curves.py [--ckpt overfit_ckpt.pt] [--out predicted_curves.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from chart_encoder import ChartEncoderPipeline, load_vision_encoder
from chart_decoder import TinyPrimitiveDecoder

CKPT_MODEL  = "google/siglip2-base-patch16-naflex"
IMAGE_PATH  = ("/home/valenbonas/Documents/Investigacion_doctorado"
               "/diffvg/apps/imgs/charts/area0_xiv.jpg")
CANVAS_W    = 1024
CANVAS_H    = 629
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def main(ckpt_path: str, out_path: str) -> None:
    print(f"Device   : {DEVICE}")
    print(f"Checkpoint: {ckpt_path}")

    # ── processor + image ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(CKPT_MODEL)
    img    = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # ── build model ──────────────────────────────────────────────────────
    vision_enc_raw = load_vision_encoder(CKPT_MODEL, device=DEVICE)
    enc_pipeline   = ChartEncoderPipeline(
        vision_encoder    = vision_enc_raw,
        n_unfrozen_blocks = 2,
        n_adapter_layers  = 2,
        adapter_reduction = 8,
        decoder_dim       = None,
    ).to(DEVICE)

    enc_dim = enc_pipeline.vision_enc.hidden_size
    decoder = TinyPrimitiveDecoder(
        d_model=enc_dim, n_queries=512, n_heads=8, n_layers=4
    ).to(DEVICE)

    # ── load checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    enc_pipeline.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    enc_pipeline.eval()
    decoder.eval()
    print(f"Loaded checkpoint  (step={ckpt['step']}, "
          f"loss={ckpt['final_loss']:.6f})")

    # ── forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        tokens, mask = enc_pipeline(
            pixel_values         = inputs["pixel_values"],
            pixel_attention_mask = inputs.get("pixel_attention_mask"),
            spatial_shapes       = inputs.get("spatial_shapes"),
        )
        out = decoder(tokens, mask)

    # points: [512, 9, 2] normalised, colors: [512, 4] RGBA
    pred_pts = out.points[0].cpu()
    pred_clr = out.colors[0].cpu()

    torch.save({
        "points":   pred_pts,    # [512, 9, 2]  in [0,1]
        "colors":   pred_clr,    # [512, 4]     in [0,1]
        "canvas_w": CANVAS_W,
        "canvas_h": CANVAS_H,
    }, out_path)
    print(f"Saved predicted curves → {out_path}")
    print(f"  points : {tuple(pred_pts.shape)}")
    print(f"  colors : {tuple(pred_clr.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="overfit_ckpt.pt")
    parser.add_argument("--out",  default="predicted_curves.pt")
    args = parser.parse_args()
    main(args.ckpt, args.out)
