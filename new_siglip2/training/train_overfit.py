"""
Single-image overfitting sanity check for TinyChartSplat-DETR-v0.
=========================================================================
Trains the encoder (last 2 blocks) + decoder (all params) on one chart
image using the pre-optimised iter_399.svg as ground-truth curve targets.

Loss
----
Hungarian-matched MSE on control points + RGBA colours.
  - Cost matrix built on detached predictions  [N, N].
  - scipy.optimize.linear_sum_assignment finds the optimal bijective
    assignment. Assignment is recomputed every REASSIGN_EVERY steps.
  - Differentiable MSE computed on the matched pairs.

Usage
-----
    python train_overfit.py
"""
from __future__ import annotations

import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor

# Make sure our modules are on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from chart_encoder import ChartEncoderPipeline, load_vision_encoder
from chart_decoder import TinyPrimitiveDecoder, BezierOutput

# ── config ──────────────────────────────────────────────────────────────────
CKPT       = "google/siglip2-base-patch16-naflex"
IMAGE_PATH = ("/home/valenbonas/Documents/Investigacion_doctorado"
              "/diffvg/apps/imgs/charts/area0_xiv.jpg")
SVG_PATH   = ("/home/valenbonas/Documents/Investigacion_doctorado"
              "/diffvg/apps/results/L2_area0_rendering_512(3)/iter_399.svg")
CANVAS_W   = 1024
CANVAS_H   = 629
N_CURVES   = 512

N_ITER          = 300          # training steps
LR_ENC          = 1e-5        # fine-tuning LR for unfrozen encoder blocks
LR_DEC          = 1e-4        # decoder (all params)
COLOR_WEIGHT    = 0.5         # weight of colour loss relative to point loss
REASSIGN_EVERY  = 5           # recompute Hungarian assignment every N steps
LOG_EVERY       = 10          # print loss every N steps
SAVE_CKPT       = True
CKPT_PATH       = str(Path(__file__).parent.parent / "outputs" / "overfit_ckpt.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── SVG parsing ─────────────────────────────────────────────────────────────

def parse_svg_curves(svg_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse a pydiffvg blob-mode SVG and return ground-truth tensors.

    Each closed cubic blob path has the layout::

        M  x0 y0
        C  x1 y1  x2 y2  x3 y3      (segment 0)
        C  x4 y4  x5 y5  x6 y6      (segment 1)
        C  x7 y7  x8 y8  x0 y0      (segment 2, closes to anchor0)

    We extract points 0-8 (9 × 2 floats) and normalise by canvas size.
    Colour is parsed from ``fill="rgb(r,g,b)"`` and ``opacity="a"``.

    Returns
    -------
    points : [N, 9, 2]  float32  normalised to [0, 1]
    colors : [N, 4]     float32  RGBA in [0, 1]
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    # Handle namespace-qualified tags
    ns = root.tag.split('}')[0][1:] if '}' in root.tag else ''
    tag = f'{{{ns}}}path' if ns else 'path'
    paths = root.findall(f'.//{tag}')
    assert len(paths) == N_CURVES, \
        f"Expected {N_CURVES} paths in SVG, found {len(paths)}"

    all_pts, all_clr = [], []
    for p in paths:
        # ── points ──────────────────────────────────────────────────────
        d    = p.get('d', '')
        nums = [float(x) for x in
                re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', d)]
        # First 18 floats = 9 points (last 2 floats = closing repeat of pt0)
        pts = [[nums[2 * i] / CANVAS_W, nums[2 * i + 1] / CANVAS_H]
               for i in range(9)]
        all_pts.append(pts)

        # ── colour ──────────────────────────────────────────────────────
        fill  = p.get('fill', 'rgb(0,0,0)')
        rgb   = [int(c) / 255.0 for c in re.findall(r'\d+', fill)]
        alpha = float(p.get('opacity', '1.0'))
        all_clr.append(rgb + [alpha])

    pts_t = torch.tensor(all_pts, dtype=torch.float32)   # [N, 9, 2]
    clr_t = torch.tensor(all_clr, dtype=torch.float32)   # [N, 4]
    return pts_t, clr_t


# ── Hungarian matching loss ──────────────────────────────────────────────────

class HungarianMatcher:
    """
    Maintains a cached assignment that is refreshed every ``reassign_every``
    calls so Hungarian matching does not dominate wall-clock time.
    """

    def __init__(self, reassign_every: int = 5, color_weight: float = 0.5):
        self.reassign_every = reassign_every
        self.color_weight   = color_weight
        self._step          = 0
        self._col_ind: torch.Tensor | None = None

    @torch.no_grad()
    def _compute_assignment(
        self,
        pred_pts: torch.Tensor,
        pred_clr: torch.Tensor,
        gt_pts:   torch.Tensor,
        gt_clr:   torch.Tensor,
    ) -> torch.Tensor:
        """Build cost matrix and return col_ind from linear_sum_assignment."""
        N = pred_pts.shape[0]
        p_flat = pred_pts.detach().view(N, -1).float().cpu()   # [N, 18]
        g_flat = gt_pts.view(N, -1).float().cpu()              # [N, 18]

        cost_pts = torch.cdist(p_flat, g_flat, p=2).pow(2)    # [N, N]
        cost_clr = torch.cdist(
            pred_clr.detach().float().cpu(),
            gt_clr.float().cpu(), p=2
        ).pow(2)                                                # [N, N]

        cost = cost_pts + self.color_weight * cost_clr
        _, col_ind = linear_sum_assignment(cost.numpy())
        return torch.from_numpy(col_ind).long()

    def __call__(
        self,
        pred_pts: torch.Tensor,
        pred_clr: torch.Tensor,
        gt_pts:   torch.Tensor,
        gt_clr:   torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Returns
        -------
        total_loss, loss_pts (float), loss_clr (float)
        """
        # Refresh assignment periodically
        if self._col_ind is None or self._step % self.reassign_every == 0:
            self._col_ind = self._compute_assignment(
                pred_pts, pred_clr, gt_pts, gt_clr)
        self._step += 1

        col_ind = self._col_ind.to(gt_pts.device)
        matched_gt_pts = gt_pts[col_ind]   # [N, 9, 2]
        matched_gt_clr = gt_clr[col_ind]   # [N, 4]

        loss_pts = F.mse_loss(pred_pts, matched_gt_pts)
        loss_clr = F.mse_loss(pred_clr, matched_gt_clr)
        total    = loss_pts + self.color_weight * loss_clr

        return total, loss_pts.item(), loss_clr.item()


# ── main training loop ───────────────────────────────────────────────────────

def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"Image  : {IMAGE_PATH}")
    print(f"SVG    : {SVG_PATH}\n")

    # ── load ground-truth ────────────────────────────────────────────────
    print("Parsing SVG ground-truth …")
    gt_pts, gt_clr = parse_svg_curves(SVG_PATH)
    gt_pts = gt_pts.to(DEVICE)   # [512, 9, 2]
    gt_clr = gt_clr.to(DEVICE)   # [512, 4]
    print(f"  gt_pts : {tuple(gt_pts.shape)}, "
          f"range [{gt_pts.min():.3f}, {gt_pts.max():.3f}]")
    print(f"  gt_clr : {tuple(gt_clr.shape)}, "
          f"range [{gt_clr.min():.3f}, {gt_clr.max():.3f}]\n")

    # ── processor + image ────────────────────────────────────────────────
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(CKPT)
    img    = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    print(f"  pixel_values : {tuple(inputs['pixel_values'].shape)}\n")

    # ── build model ──────────────────────────────────────────────────────
    print("Loading encoder …")
    vision_enc_raw = load_vision_encoder(CKPT, device=DEVICE)
    enc_pipeline   = ChartEncoderPipeline(
        vision_encoder    = vision_enc_raw,
        n_unfrozen_blocks = 2,
        n_adapter_layers  = 2,
        adapter_reduction = 8,
        decoder_dim       = None,   # no projection → [B, T, 768]
    ).to(DEVICE)

    enc_dim = enc_pipeline.vision_enc.hidden_size   # 768
    print(f"  encoder hidden size : {enc_dim}")

    print("Building decoder …")
    decoder = TinyPrimitiveDecoder(
        d_model        = enc_dim,
        n_queries      = N_CURVES,
        n_heads        = 8,
        n_layers       = 4,
        dim_feedforward= 2048,
    ).to(DEVICE)
    print(f"  decoder params : {decoder.total_params:,}\n")

    # ── optimiser – separate LRs for encoder vs decoder ──────────────────
    # Collect only the trainable encoder params (unfrozen blocks + adapters)
    enc_trainable = [p for p in enc_pipeline.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": enc_trainable,          "lr": LR_ENC},
        {"params": decoder.parameters(),   "lr": LR_DEC},
    ], weight_decay=1e-4)

    matcher = HungarianMatcher(
        reassign_every=REASSIGN_EVERY,
        color_weight=COLOR_WEIGHT,
    )

    # ── training loop ────────────────────────────────────────────────────
    print(f"Training for {N_ITER} iterations …\n")
    print(f"{'Step':>5}  {'Loss':>10}  {'Pts':>10}  {'Clr':>10}  {'Time(s)':>8}")
    print("─" * 55)

    t0 = time.time()
    for step in range(1, N_ITER + 1):
        optimizer.zero_grad()

        # Forward
        tokens, mask = enc_pipeline(
            pixel_values         = inputs["pixel_values"],
            pixel_attention_mask = inputs.get("pixel_attention_mask"),
            spatial_shapes       = inputs.get("spatial_shapes"),
        )                               # tokens: [1, T, 768]

        out: BezierOutput = decoder(tokens, mask)
        # Squeeze batch dim → [N, 9, 2] and [N, 4]
        pred_pts = out.points[0]
        pred_clr = out.colors[0]

        # Loss with Hungarian assignment
        loss, l_pts, l_clr = matcher(pred_pts, pred_clr, gt_pts, gt_clr)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc_trainable) + list(decoder.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"{step:>5}  {loss.item():>10.6f}  "
                  f"{l_pts:>10.6f}  {l_clr:>10.6f}  {elapsed:>8.1f}")

    print("─" * 55)
    print(f"Final loss : {loss.item():.6f}")

    if SAVE_CKPT:
        ckpt = {
            "encoder_state": enc_pipeline.state_dict(),
            "decoder_state": decoder.state_dict(),
            "step": N_ITER,
            "final_loss": loss.item(),
        }
        torch.save(ckpt, CKPT_PATH)
        print(f"Checkpoint saved → {CKPT_PATH}")


if __name__ == "__main__":
    main()
