"""
TinyChartSplat-DETR-v0  –  Differentiable Rendering Training
=============================================================
Trains the encoder + decoder end-to-end using pydiffvg (GPU mode) as the
differentiable renderer.

KEY initialisation rule discovered through debugging
-----------------------------------------------------
  pydiffvg.set_use_gpu(True) + one warm-up render  MUST happen BEFORE any
  PyTorch model is moved onto the GPU.  Otherwise cudaMallocManaged (used
  internally by pydiffvg) conflicts with PyTorch's CUDA virtual-memory pool
  and produces "illegal memory access" errors.

  Additionally, pydiffvg GPU rendering crashes with all 512 curves at once
  (internal buffer overflow).  We therefore split into two passes of 256
  curves and alpha-composite the results.

Architecture of one training step
----------------------------------
  1. pydiffvg GPU warm-up is done once at startup before model loading.
  2. Encoder / decoder forward on CUDA → pred_pts [N,9,2]  pred_clr [N,4]
  3. Create CPU leaf tensors (exactly like chart_rendering.py)
  4. Two-pass GPU render:
       pass 1 → curves 0-255   → img1 [H,W,4]
       pass 2 → curves 256-511 → img2 [H,W,4]
     alpha-composite img2 over img1 → final_rgba [H,W,4]
  5. Pixel MSE vs target resized to same dims → loss.backward()
       → pts_leaf.grad / clr_leaf.grad  (accumulated from both passes)
  6. Surrogate in decoder graph:
         surrogate = Σ(pred_pts · pts_leaf_grad)
                   + Σ(pred_clr · clr_leaf_grad)
         surrogate.backward() → decoder / encoder weights updated

Run in the im2v conda environment (has pydiffvg + transformers):
    /home/valenbonas/miniconda3/envs/im2v/bin/python -u train_diffvg.py

Flags
-----
--warmup_ckpt  overfit_ckpt.pt  warm-start from parameter-supervised ckpt
--n_iter       500
--render_w     256     rendering canvas width  (height = w * 629/1024)
--lr_dec       1e-4
--lr_enc       1e-5
--reg_weight   0.0     Hungarian GT-SVG regularisation  (0 = disable)
--save_every   50
--log_every    10
"""
from __future__ import annotations

# ── pydiffvg MUST be imported first, before torch, and GPU mode set here ──────
import pydiffvg
pydiffvg.set_use_gpu(True)

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from chart_encoder import ChartEncoderPipeline, load_vision_encoder
from chart_decoder import TinyPrimitiveDecoder, BezierOutput

# ── static config ─────────────────────────────────────────────────────────────
CKPT_MODEL   = "google/siglip2-base-patch16-naflex"
IMAGE_PATH   = ("/home/valenbonas/Documents/Investigacion_doctorado"
                "/diffvg/apps/imgs/charts/area0_xiv.jpg")
SVG_PATH     = ("/home/valenbonas/Documents/Investigacion_doctorado"
                "/diffvg/apps/results/L2_area0_rendering_512(3)/iter_399.svg")
ORIG_W       = 1024
ORIG_H       = 629
N_CURVES     = 512
HALF         = N_CURVES // 2          # 256 – max curves per pydiffvg pass
OUT_DIR      = Path(__file__).parent.parent / "outputs" / "diffvg_train_out"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
_NUM_CTRL    = torch.zeros(3, dtype=torch.int32) + 2   # CPU [2,2,2]
_STROKE_W    = torch.tensor(0.0)                        # CPU scalar


# ── pydiffvg warm-up helper ────────────────────────────────────────────────────

def _pydiffvg_warmup(rw: int, rh: int) -> None:
    """
    Perform one GPU render of random curves to initialise pydiffvg's
    cudaMallocManaged pool BEFORE any PyTorch model is moved onto the GPU.

    This must be called at the top of main(), before load_vision_encoder().
    """
    pts = torch.rand(HALF, 9, 2)
    clr = torch.rand(HALF, 4)
    scale = torch.tensor([float(rw), float(rh)])
    shapes = [pydiffvg.Path(
        num_control_points=_NUM_CTRL,
        points=(pts[i] * scale).contiguous(),
        stroke_width=_STROKE_W,
        is_closed=True) for i in range(HALF)]
    groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([i]),
        fill_color=clr[i].contiguous()) for i in range(HALF)]
    sa = pydiffvg.RenderFunction.serialize_scene(rw, rh, shapes, groups)
    pydiffvg.RenderFunction.apply(rw, rh, 2, 2, 0, None, *sa)


# ── SVG parsing ───────────────────────────────────────────────────────────────

def parse_svg_curves(svg_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    tree  = ET.parse(svg_path)
    root  = tree.getroot()
    ns    = root.tag.split('}')[0][1:] if '}' in root.tag else ''
    tag   = f'{{{ns}}}path' if ns else 'path'
    paths = root.findall(f'.//{tag}')
    all_pts, all_clr = [], []
    for p in paths:
        d    = p.get('d', '')
        nums = [float(x) for x in
                re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', d)]
        pts  = [[nums[2*i] / ORIG_W, nums[2*i+1] / ORIG_H] for i in range(9)]
        all_pts.append(pts)
        fill = p.get('fill', 'rgb(0,0,0)')
        rgb  = [int(c) / 255.0 for c in re.findall(r'\d+', fill)]
        a    = float(p.get('opacity', '1.0'))
        all_clr.append(rgb + [a])
    return (torch.tensor(all_pts, dtype=torch.float32),
            torch.tensor(all_clr, dtype=torch.float32))


# ── differentiable renderer (GPU, 256-curve passes) ───────────────────────────

def _render_half(pts_leaf_half: torch.Tensor, clr_leaf_half: torch.Tensor,
                 rw: int, rh: int, seed: int) -> torch.Tensor:
    """
    Render exactly 256 curves with pydiffvg GPU.

    pts_leaf_half : [256, 9, 2] CPU, part of leaf autograd graph
    clr_leaf_half : [256, 4]   CPU, part of leaf autograd graph
    Returns [rh, rw, 4] CPU RGBA, grad-connected via pydiffvg backward.
    """
    n = pts_leaf_half.shape[0]
    scale = torch.tensor([float(rw), float(rh)])
    shapes = [pydiffvg.Path(
        num_control_points=_NUM_CTRL,
        points=(pts_leaf_half[i] * scale).contiguous(),
        stroke_width=_STROKE_W,
        is_closed=True) for i in range(n)]
    groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([i]),
        fill_color=clr_leaf_half[i].contiguous()) for i in range(n)]
    sa = pydiffvg.RenderFunction.serialize_scene(rw, rh, shapes, groups)
    return pydiffvg.RenderFunction.apply(rw, rh, 2, 2, seed, None, *sa)


def render_two_pass(pts_leaf: torch.Tensor, clr_leaf: torch.Tensor,
                    rw: int, rh: int, seed: int = 0) -> torch.Tensor:
    """
    Full 512-curve render via two 256-curve passes, alpha-composited.

    The result is differentiable with respect to pts_leaf and clr_leaf
    (gradients accumulate in pts_leaf.grad / clr_leaf.grad after backward).
    """
    r1 = _render_half(pts_leaf[:HALF],  clr_leaf[:HALF],  rw, rh, seed)
    r2 = _render_half(pts_leaf[HALF:],  clr_leaf[HALF:],  rw, rh, seed + 1)
    # alpha-composite r2 over r1 (pass 2 curves appear on top)
    a1 = r1[:, :, 3:4]; a2 = r2[:, :, 3:4]
    a_out  = a1 + a2 * (1.0 - a1)
    rgb_out = (r1[:, :, :3] * a1 + r2[:, :, :3] * a2 * (1.0 - a1)) / (a_out + 1e-8)
    return torch.cat([rgb_out, a_out], dim=-1)   # [H,W,4]


def composite_over_white(img_rgba: torch.Tensor) -> torch.Tensor:
    """[H,W,4] → [H,W,3] alpha-composited over white background."""
    a = img_rgba[:, :, 3:4]
    return a * img_rgba[:, :, :3] + (1.0 - a)


# ── Hungarian regularisation ──────────────────────────────────────────────────

class HungarianReg:
    def __init__(self, gt_pts, gt_clr, color_w=0.5, reassign_every=10):
        self.gt_pts = gt_pts; self.gt_clr = gt_clr
        self.color_w = color_w; self.reassign_every = reassign_every
        self._step = 0; self._col = None

    @torch.no_grad()
    def _assign(self, pp, pc):
        N = pp.shape[0]
        cost = (torch.cdist(pp.view(N,-1).cpu(), self.gt_pts.view(N,-1).cpu()).pow(2)
                + self.color_w * torch.cdist(pc.cpu(), self.gt_clr.cpu()).pow(2))
        _, col = linear_sum_assignment(cost.numpy())
        return torch.from_numpy(col).long()

    def __call__(self, pred_pts, pred_clr):
        if self._col is None or self._step % self.reassign_every == 0:
            self._col = self._assign(pred_pts, pred_clr)
        self._step += 1
        ci = self._col.to(pred_pts.device)
        return (F.mse_loss(pred_pts, self.gt_pts[ci].to(pred_pts.device))
                + self.color_w *
                F.mse_loss(pred_clr, self.gt_clr[ci].to(pred_clr.device)))


# ── save helpers ──────────────────────────────────────────────────────────────

def save_png(pts_leaf: torch.Tensor, clr_leaf: torch.Tensor,
             rw: int, rh: int, path: Path) -> None:
    """Render and save as PNG."""
    with torch.no_grad():
        img_rgba = render_two_pass(pts_leaf.detach(), clr_leaf.detach(), rw, rh, seed=0)
    rgb = composite_over_white(img_rgba).cpu()
    arr = (rgb.clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((ORIG_W, ORIG_H), Image.NEAREST)
    pil.save(path)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    RENDER_W = args.render_w
    RENDER_H = round(RENDER_W * ORIG_H / ORIG_W)

    # ── CRITICAL: pydiffvg GPU warm-up BEFORE any model loading ──────────
    print(f"pydiffvg GPU warm-up at {RENDER_W}×{RENDER_H} …")
    _pydiffvg_warmup(RENDER_W, RENDER_H)
    print(f"  done.  Device={DEVICE}  render={RENDER_W}×{RENDER_H}\n")

    # ── target at render resolution ─────────────────────────────────────
    # pydiffvg GPU mode returns CUDA tensors → keep target on CPU, move after render
    target_cpu = torch.from_numpy(
        np.array(Image.open(IMAGE_PATH).convert("RGB")
                 .resize((RENDER_W, RENDER_H), Image.LANCZOS))
        .astype(np.float32) / 255.0)   # [rh, rw, 3]  CPU

    # ── GT SVG regularisation ────────────────────────────────────────────
    reg = None
    if args.reg_weight > 0:
        print("Parsing GT SVG …")
        gt_pts, gt_clr = parse_svg_curves(SVG_PATH)
        reg = HungarianReg(gt_pts.to(DEVICE), gt_clr.to(DEVICE))

    # ── processor + image ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(CKPT_MODEL)
    inputs    = {k: v.to(DEVICE) for k, v in
                 processor(images=[Image.open(IMAGE_PATH).convert("RGB")],
                            return_tensors="pt").items()}

    # ── model ────────────────────────────────────────────────────────────
    print("Loading encoder …")
    enc_pipeline = ChartEncoderPipeline(
        vision_encoder    = load_vision_encoder(CKPT_MODEL, device=DEVICE),
        n_unfrozen_blocks = 2, n_adapter_layers = 2,
        adapter_reduction = 8, decoder_dim = None,
    ).to(DEVICE)
    decoder = TinyPrimitiveDecoder(
        d_model=enc_pipeline.vision_enc.hidden_size,
        n_queries=N_CURVES, n_heads=8, n_layers=4,
    ).to(DEVICE)

    if args.warmup_ckpt and Path(args.warmup_ckpt).exists():
        ck = torch.load(args.warmup_ckpt, map_location=DEVICE)
        enc_pipeline.load_state_dict(ck["encoder_state"])
        decoder.load_state_dict(ck["decoder_state"])
        print(f"Warm-started from {args.warmup_ckpt}  (loss={ck['final_loss']:.6f})")
    else:
        print("Training from random init")

    enc_trainable = [p for p in enc_pipeline.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": enc_trainable,        "lr": args.lr_enc},
        {"params": decoder.parameters(), "lr": args.lr_dec},
    ], weight_decay=1e-4)

    # ── training loop ─────────────────────────────────────────────────────
    print(f"\nTraining {args.n_iter} iters  "
          f"(render_loss + {args.reg_weight}×reg)\n")
    hdr = f"{'Step':>5}  {'Render':>10}  {'Reg':>10}  {'s/step':>8}  {'Total(s)':>9}"
    print(hdr); print("─" * len(hdr))

    t_total = time.time()
    t_step  = time.time()
    last_pts_leaf = None
    last_clr_leaf = None

    for step in range(1, args.n_iter + 1):
        optimizer.zero_grad()

        # 1. decoder forward on CUDA
        tokens, mask = enc_pipeline(**inputs)
        out: BezierOutput = decoder(tokens, mask)
        pred_pts = out.points[0]   # [N,9,2] CUDA, grad-connected
        pred_clr = out.colors[0]   # [N,4]   CUDA, grad-connected

        # 2. CPU leaf tensors – values in [0,1], mirror chart_rendering.py
        pts_leaf = pred_pts.detach().float().cpu(); pts_leaf.requires_grad_(True)
        clr_leaf = pred_clr.detach().float().cpu(); clr_leaf.requires_grad_(True)

        # 3. Two-pass GPU render (256 curves per pass)
        img_rgba    = render_two_pass(pts_leaf, clr_leaf, RENDER_W, RENDER_H,
                                       seed=step)
        # img_rgba is CUDA (GPU mode); move to CPU for loss so target stays consistent
        img_rgb     = composite_over_white(img_rgba).cpu()   # [rh,rw,3] CPU
        loss_render = F.mse_loss(img_rgb, target_cpu)

        # 4. backward through pydiffvg graph → fill pts_leaf.grad / clr_leaf.grad
        loss_render.backward()

        # 5. chain render grads back through decoder via surrogate
        pts_g = pts_leaf.grad.to(DEVICE) if pts_leaf.grad is not None \
                else torch.zeros_like(pred_pts)
        clr_g = clr_leaf.grad.to(DEVICE) if clr_leaf.grad is not None \
                else torch.zeros_like(pred_clr)
        surrogate = (pred_pts * pts_g).sum() + (pred_clr * clr_g).sum()

        # 6. optional Hungarian regularisation
        loss_reg = reg(pred_pts, pred_clr) if reg is not None else \
                   torch.tensor(0.0, device=DEVICE)

        (surrogate + args.reg_weight * loss_reg).backward()
        torch.nn.utils.clip_grad_norm_(
            enc_trainable + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()

        last_pts_leaf = pts_leaf.detach()
        last_clr_leaf = clr_leaf.detach()

        if step % args.log_every == 0 or step == 1:
            sps = time.time() - t_step
            print(f"{step:>5}  {loss_render.item():>10.6f}  "
                  f"{loss_reg.item():>10.6f}  "
                  f"{sps/args.log_every:>8.2f}  "
                  f"{time.time()-t_total:>9.1f}")
            t_step = time.time()

        if step % args.save_every == 0 or step == args.n_iter:
            save_png(last_pts_leaf, last_clr_leaf, RENDER_W, RENDER_H,
                     out_dir / f"step_{step:04d}.png")
            torch.save({"encoder_state": enc_pipeline.state_dict(),
                        "decoder_state": decoder.state_dict(),
                        "step": step, "final_loss": loss_render.item()},
                       out_dir / "ckpt_latest.pt")

    # ── final comparison ──────────────────────────────────────────────────
    print("\nSaving final comparison …")
    snap      = render_two_pass(last_pts_leaf, last_clr_leaf,
                                RENDER_W, RENDER_H, seed=0)
    pred_u8   = (composite_over_white(snap).cpu().clamp(0,1).numpy()*255).astype(np.uint8)
    pred_u8   = np.array(Image.fromarray(pred_u8).resize((ORIG_W, ORIG_H),
                                                          Image.NEAREST))
    target_u8 = np.array(Image.open(IMAGE_PATH).convert("RGB").resize((ORIG_W, ORIG_H)))
    sep = np.ones((ORIG_H, 8, 3), dtype=np.uint8) * 120

    def lbl(a, t):
        from PIL import ImageDraw
        p = Image.fromarray(a); d = ImageDraw.Draw(p)
        d.rectangle([0, 0, p.width, 24], fill=(20, 20, 20))
        d.text((6, 5), t, fill=(255, 255, 255))
        return np.array(p)

    Image.fromarray(np.concatenate(
        [lbl(target_u8, "Target"),
         sep,
         lbl(pred_u8, f"Prediction  step={args.n_iter}  {RENDER_W}×{RENDER_H}")],
        axis=1)).save(out_dir / "final_comparison.png")
    print(f"Done.  Outputs → {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--warmup_ckpt",
                   default=str(Path(__file__).parent.parent / "outputs" / "overfit_ckpt.pt"))
    p.add_argument("--n_iter",      type=int,   default=500)
    p.add_argument("--render_w",    type=int,   default=256,
                   help="Render canvas width (height auto-scaled)")
    p.add_argument("--lr_dec",      type=float, default=1e-4)
    p.add_argument("--lr_enc",      type=float, default=1e-5)
    p.add_argument("--reg_weight",  type=float, default=0.0)
    p.add_argument("--save_every",  type=int,   default=50)
    p.add_argument("--log_every",   type=int,   default=10)
    p.add_argument("--out_dir",     type=str,
                   default=str(Path(__file__).parent.parent / "outputs" / "diffvg_train_out"))
    args = p.parse_args()
    main(args)
