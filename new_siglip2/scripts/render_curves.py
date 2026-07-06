"""
Step 2 – Rendering (im2v env, pydiffvg)
=======================================
Loads predicted_curves.pt produced by infer_curves.py and renders the
512 blob Bézier curves using pydiffvg, following the same pattern as
chart_rendering.py.  Saves side-by-side target vs. prediction PNG.

Usage
-----
    python render_curves.py [--curves predicted_curves.pt] [--out render_out.png]
"""
from __future__ import annotations

import argparse

import pydiffvg
import torch
import skimage.io
import numpy as np

TARGET_PATH = ("/home/valenbonas/Documents/Investigacion_doctorado"
               "/diffvg/apps/imgs/charts/area0_xiv.jpg")


def render_curves(pts: torch.Tensor, clr: torch.Tensor,
                  canvas_w: int, canvas_h: int) -> torch.Tensor:
    """
    Build and render 512 closed cubic blob paths with pydiffvg.

    Parameters
    ----------
    pts      : [N, 9, 2]  normalised to [0, 1]
    clr      : [N, 4]     RGBA in [0, 1]
    canvas_w : int
    canvas_h : int

    Returns
    -------
    img : [H, W, 4]  float32  RGBA image on CPU
    """
    dev = pydiffvg.get_device()
    num_ctrl = torch.zeros(3, dtype=torch.int32) + 2   # [2, 2, 2]

    shapes, shape_groups = [], []
    for i in range(pts.shape[0]):
        p = pts[i].clone().float()          # [9, 2]
        p[:, 0] *= canvas_w
        p[:, 1] *= canvas_h
        p = p.to(dev)

        path = pydiffvg.Path(
            num_control_points = num_ctrl,
            points             = p,
            stroke_width       = torch.tensor(0.0),
            is_closed          = True,
        )
        shapes.append(path)

        fill = clr[i].clone().float().to(dev)
        shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids  = torch.tensor([i]),
            fill_color = fill,
        ))

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_w, canvas_h, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_w, canvas_h, 2, 2, 0, None, *scene_args)
    return img.cpu()      # [H, W, 4]


def composite_on_white(img_rgba: torch.Tensor) -> np.ndarray:
    """Alpha-composite RGBA image over white background → uint8 [H, W, 3]."""
    alpha = img_rgba[:, :, 3:4]
    rgb   = img_rgba[:, :, :3]
    white = torch.ones_like(rgb)
    composited = alpha * rgb + (1 - alpha) * white
    return (composited.clamp(0, 1).numpy() * 255).astype(np.uint8)


def main(curves_path: str, out_path: str) -> None:
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    dev = pydiffvg.get_device()
    print(f"pydiffvg device : {dev}")

    # ── load predicted curves ─────────────────────────────────────────────
    data     = torch.load(curves_path, map_location="cpu")
    pts      = data["points"].to(dev)    # [512, 9, 2]
    clr      = data["colors"].to(dev)    # [512, 4]
    canvas_w = int(data["canvas_w"])
    canvas_h = int(data["canvas_h"])
    print(f"Curves loaded  : {tuple(pts.shape)}  canvas={canvas_w}×{canvas_h}")

    # ── render prediction ─────────────────────────────────────────────────
    print("Rendering predicted curves …")
    img_rgba = render_curves(pts, clr, canvas_w, canvas_h)   # [H, W, 4]
    pred_rgb = composite_on_white(img_rgba)                   # [H, W, 3] uint8

    # ── load target image ─────────────────────────────────────────────────
    target_raw = skimage.io.imread(TARGET_PATH)
    if target_raw.ndim == 2:
        target_raw = np.stack([target_raw] * 3, axis=-1)
    target_rgb = target_raw[:, :, :3]

    # ── side-by-side ──────────────────────────────────────────────────────
    # Resize target to canvas size if needed
    from PIL import Image as PILImage
    target_pil = PILImage.fromarray(target_rgb).resize(
        (canvas_w, canvas_h), PILImage.LANCZOS)
    target_resized = np.array(target_pil)

    separator = np.ones((canvas_h, 10, 3), dtype=np.uint8) * 180
    side_by_side = np.concatenate(
        [target_resized, separator, pred_rgb], axis=1)

    # Save outputs
    PILImage.fromarray(pred_rgb).save(out_path)
    print(f"Prediction saved → {out_path}")

    side_path = out_path.replace(".png", "_comparison.png")
    PILImage.fromarray(side_by_side).save(side_path)
    print(f"Comparison saved → {side_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default="predicted_curves.pt")
    parser.add_argument("--out",    default="render_out.png")
    args = parser.parse_args()
    main(args.curves, args.out)
