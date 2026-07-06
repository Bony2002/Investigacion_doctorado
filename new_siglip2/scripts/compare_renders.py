"""
Three-way visual comparison: target image | GT SVG (iter_399) | prediction
Run in im2v env.
"""
import re
import xml.etree.ElementTree as ET
import numpy as np
import torch
import pydiffvg
import skimage.io
from PIL import Image

TARGET_PATH  = ("/home/valenbonas/Documents/Investigacion_doctorado"
                "/diffvg/apps/imgs/charts/area0_xiv.jpg")
SVG_PATH     = ("/home/valenbonas/Documents/Investigacion_doctorado"
                "/diffvg/apps/results/L2_area0_rendering_512(3)/iter_399.svg")
CURVES_PATH  = "predicted_curves.pt"
OUT_PATH     = "comparison_3way.png"
CANVAS_W, CANVAS_H = 1024, 629


def render_paths(shapes, shape_groups, w, h):
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(w, h, 2, 2, 0, None, *scene_args)
    return img.cpu()   # [H, W, 4]


def composite(img_rgba):
    alpha = img_rgba[:, :, 3:4]
    rgb   = img_rgba[:, :, :3]
    out   = alpha * rgb + (1 - alpha) * torch.ones_like(rgb)
    return (out.clamp(0, 1).numpy() * 255).astype(np.uint8)


def parse_and_render_svg(svg_path, w, h):
    tree  = ET.parse(svg_path)
    root  = tree.getroot()
    ns    = root.tag.split('}')[0][1:] if '}' in root.tag else ''
    tag   = f'{{{ns}}}path' if ns else 'path'
    paths = root.findall(f'.//{tag}')

    num_ctrl = torch.zeros(3, dtype=torch.int32) + 2
    dev = pydiffvg.get_device()
    shapes, groups = [], []

    for i, p in enumerate(paths):
        d    = p.get('d', '')
        nums = [float(x) for x in
                re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', d)]
        pts  = torch.tensor([[nums[2*j], nums[2*j+1]] for j in range(9)],
                            dtype=torch.float32).to(dev)
        fill = p.get('fill', 'rgb(0,0,0)')
        rgb  = [int(c) / 255.0 for c in re.findall(r'\d+', fill)]
        a    = float(p.get('opacity', '1.0'))
        col  = torch.tensor(rgb + [a], dtype=torch.float32).to(dev)

        shapes.append(pydiffvg.Path(
            num_control_points=num_ctrl, points=pts,
            stroke_width=torch.tensor(0.0), is_closed=True))
        groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]), fill_color=col))

    return render_paths(shapes, groups, w, h)


def render_predicted(curves_path, w, h):
    dev  = pydiffvg.get_device()
    data = torch.load(curves_path, map_location='cpu')
    pts  = data['points'].to(dev)
    clr  = data['colors'].to(dev)
    num_ctrl = torch.zeros(3, dtype=torch.int32) + 2
    shapes, groups = [], []

    for i in range(pts.shape[0]):
        p = pts[i].clone().float()
        p[:, 0] *= w;  p[:, 1] *= h
        shapes.append(pydiffvg.Path(
            num_control_points=num_ctrl, points=p,
            stroke_width=torch.tensor(0.0), is_closed=True))
        groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=clr[i].clone().float()))

    return render_paths(shapes, groups, w, h)


def mse(a, b):
    return float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())


pydiffvg.set_use_gpu(torch.cuda.is_available())

# ── render GT SVG ─────────────────────────────────────────────────────────────
print("Rendering GT SVG (iter_399) …")
gt_rgba  = parse_and_render_svg(SVG_PATH, CANVAS_W, CANVAS_H)
gt_rgb   = composite(gt_rgba)

# ── render prediction ─────────────────────────────────────────────────────────
print("Rendering prediction …")
pr_rgba  = render_predicted(CURVES_PATH, CANVAS_W, CANVAS_H)
pr_rgb   = composite(pr_rgba)

# ── load target ───────────────────────────────────────────────────────────────
target   = np.array(Image.open(TARGET_PATH).convert('RGB')
                    .resize((CANVAS_W, CANVAS_H), Image.LANCZOS))

# ── quantitative similarity ───────────────────────────────────────────────────
mse_gt_vs_target  = mse(gt_rgb,  target)
mse_pred_vs_gt    = mse(pr_rgb,  gt_rgb)
mse_pred_vs_target= mse(pr_rgb,  target)
print(f"\nPixel MSE (0-255²):")
print(f"  GT SVG  vs target : {mse_gt_vs_target:8.2f}")
print(f"  Pred    vs GT SVG : {mse_pred_vs_gt:8.2f}")
print(f"  Pred    vs target : {mse_pred_vs_target:8.2f}")

# ── build comparison image ────────────────────────────────────────────────────
sep = np.ones((CANVAS_H, 8, 3), dtype=np.uint8) * 120

def add_label(img, text):
    from PIL import ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 22], fill=(30, 30, 30))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.array(pil)

target_l = add_label(target,  "Target image")
gt_l     = add_label(gt_rgb,  "GT SVG (iter_399)")
pr_l     = add_label(pr_rgb,  f"Prediction (300 steps)  MSE vs GT={mse_pred_vs_gt:.1f}")

out = np.concatenate([target_l, sep, gt_l, sep, pr_l], axis=1)
Image.fromarray(out).save(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")
