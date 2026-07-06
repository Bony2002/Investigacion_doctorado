"""
Smoke-test for chart_encoder.py + chart_decoder.py
Checks: imports, CanvasMetadata helpers, each module independently,
        the full ChartEncoderPipeline, TinyPrimitiveDecoder, and the
        end-to-end encoder→decoder→to_diffvg_paths pipeline.
No real images needed – uses synthetic RGB PIL images.
"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from transformers import AutoProcessor

CKPT = "google/siglip2-base-patch16-naflex"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg):   print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}"); sys.exit(1)

# ── 0. imports ──────────────────────────────────────────────────────────────
section("0. Imports")
try:
    from chart_encoder import (
        CanvasMetadata, build_canvas_metadata,
        SigLIP2NaFlexVisionEncoder, load_vision_encoder,
        ChartEncoderAdapter, TokenProjector,
        ChartEncoderPipeline,
    )
    ok("All symbols imported from chart_encoder")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 1. Processor + CanvasMetadata ───────────────────────────────────────────
section("1. Processor & build_canvas_metadata")
try:
    processor = AutoProcessor.from_pretrained(CKPT)
    img = Image.new("RGB", (320, 240), color=(128, 64, 32))  # synthetic chart
    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    ok(f"Processor output keys: {list(inputs.keys())}")
    ok(f"pixel_values shape    : {inputs['pixel_values'].shape}")
    ok(f"spatial_shapes        : {inputs['spatial_shapes'].tolist()}")

    metas = build_canvas_metadata([img], inputs["spatial_shapes"].cpu())
    assert len(metas) == 1
    m = metas[0]
    assert m.original_size == (240, 320)
    assert m.canvas_size[0] == inputs["spatial_shapes"][0, 0].item() * 16
    assert m.canvas_size[1] == inputs["spatial_shapes"][0, 1].item() * 16
    ok(f"CanvasMetadata        : orig={m.original_size}  canvas={m.canvas_size}  "
       f"scale=({m.scale_h:.3f}, {m.scale_w:.3f})")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 2. SigLIP2NaFlexVisionEncoder ───────────────────────────────────────────
section("2. SigLIP2NaFlexVisionEncoder (Module 2)")
try:
    vision_enc_raw = load_vision_encoder(CKPT, device=DEVICE)
    enc = SigLIP2NaFlexVisionEncoder(vision_enc_raw, n_unfrozen_blocks=2).to(DEVICE)
    ok(f"Trainable params : {enc.trainable_params:,}")
    ok(f"Total params     : {enc.total_params:,}")
    ok(f"Frozen fraction  : {1 - enc.trainable_params/enc.total_params:.1%}")

    with torch.no_grad():
        tokens = enc(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
        )
    assert tokens.ndim == 3 and tokens.shape[0] == 1
    ok(f"Output shape     : {tuple(tokens.shape)}  (expected [1, T, {enc.hidden_size}])")
    T, C = tokens.shape[1], tokens.shape[2]
    assert C == enc.hidden_size, f"Hidden size mismatch: got {C}, expected {enc.hidden_size}"
    ok("Shape assertion passed")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 3. ChartEncoderAdapter ───────────────────────────────────────────────────
section("3. ChartEncoderAdapter (Module 3)")
try:
    adapter = ChartEncoderAdapter(d_model=C, n_layers=2, reduction=8).to(DEVICE)
    with torch.no_grad():
        adapted = adapter(tokens)
    assert adapted.shape == tokens.shape
    ok(f"Output shape     : {tuple(adapted.shape)}  (shape-preserving ✓)")

    # zero-init check: at init the adapter should be identity (output ≈ input)
    fresh_adapter = ChartEncoderAdapter(d_model=C, n_layers=2, reduction=8).to(DEVICE)
    dummy = torch.randn(1, T, C, device=DEVICE)
    with torch.no_grad():
        out_fresh = fresh_adapter(dummy)
    max_diff = (out_fresh - dummy).abs().max().item()
    assert max_diff < 1e-5, f"Zero-init identity check failed, max diff={max_diff}"
    ok(f"Zero-init identity check: max_diff={max_diff:.2e}  ✓")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 4. TokenProjector ────────────────────────────────────────────────────────
section("4. TokenProjector (Module 4)")
try:
    decoder_dim = 256
    proj = TokenProjector(in_dim=C, out_dim=decoder_dim).to(DEVICE)
    with torch.no_grad():
        projected = proj(adapted)
    assert projected.shape == (1, T, decoder_dim)
    ok(f"Output shape     : {tuple(projected.shape)}  (expected [1, {T}, {decoder_dim}])")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 5a. ChartEncoderPipeline  WITH projector (legacy path) ───────────────────
section("5a. ChartEncoderPipeline  with decoder_dim=256")
try:
    vision_enc_raw2 = load_vision_encoder(CKPT, device=DEVICE)
    pipeline_proj = ChartEncoderPipeline(
        vision_encoder=vision_enc_raw2,
        n_unfrozen_blocks=2,
        n_adapter_layers=2,
        adapter_reduction=8,
        decoder_dim=256,
    ).to(DEVICE)

    with torch.no_grad():
        out_tokens, out_mask = pipeline_proj(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
        )

    assert out_tokens.shape == (1, T, 256), f"Got {out_tokens.shape}"
    ok(f"tokens shape     : {tuple(out_tokens.shape)}")
    ok(f"mask shape       : {tuple(out_mask.shape) if out_mask is not None else 'None'}")

    img2 = Image.new("RGB", (224, 224), color=(200, 100, 50))
    inputs2 = processor(images=[img, img2], return_tensors="pt")
    inputs2 = {k: v.to(DEVICE) for k, v in inputs2.items()}
    with torch.no_grad():
        t2, m2 = pipeline_proj(
            pixel_values=inputs2["pixel_values"],
            pixel_attention_mask=inputs2.get("pixel_attention_mask"),
            spatial_shapes=inputs2.get("spatial_shapes"),
        )
    assert t2.shape[0] == 2
    ok(f"Batch-2 tokens   : {tuple(t2.shape)}")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 5b. ChartEncoderPipeline  WITHOUT projector (new default) ────────────────
section("5b. ChartEncoderPipeline  with decoder_dim=None  (no projection)")
try:
    vision_enc_raw3 = load_vision_encoder(CKPT, device=DEVICE)
    pipeline_noProj = ChartEncoderPipeline(
        vision_encoder=vision_enc_raw3,
        n_unfrozen_blocks=2,
        n_adapter_layers=2,
        adapter_reduction=8,
        # decoder_dim defaults to None → no TokenProjector
    ).to(DEVICE)

    assert pipeline_noProj.projector is None, "projector should be None"
    ok("projector is None ✓")

    with torch.no_grad():
        enc_tokens, enc_mask = pipeline_noProj(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
        )

    assert enc_tokens.shape == (1, T, C), f"Got {enc_tokens.shape}, expected [1, {T}, {C}]"
    ok(f"tokens shape     : {tuple(enc_tokens.shape)}  (full {C}-dim preserved ✓)")
    ok(f"mask shape       : {tuple(enc_mask.shape) if enc_mask is not None else 'None'}")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 6. TinyPrimitiveDecoder imports ──────────────────────────────────────────
section("6. chart_decoder imports")
try:
    from chart_decoder import BezierOutput, TinyPrimitiveDecoder, to_diffvg_paths
    ok("BezierOutput, TinyPrimitiveDecoder, to_diffvg_paths imported")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 7. TinyPrimitiveDecoder – standalone ─────────────────────────────────────
section("7. TinyPrimitiveDecoder  (standalone)")
try:
    N_QUERIES = 512
    decoder = TinyPrimitiveDecoder(
        d_model=C,
        n_queries=N_QUERIES,
        n_heads=8,
        n_layers=4,
        dim_feedforward=2048,
    ).to(DEVICE)
    ok(f"Trainable params : {decoder.trainable_params:,}")
    ok(f"Total params     : {decoder.total_params:,}")

    # Single image
    with torch.no_grad():
        out = decoder(enc_tokens, enc_mask)

    assert isinstance(out, BezierOutput)
    assert out.points.shape == (1, N_QUERIES, 9, 2), f"points: {out.points.shape}"
    assert out.colors.shape == (1, N_QUERIES, 4),    f"colors: {out.colors.shape}"
    ok(f"points shape     : {tuple(out.points.shape)}")
    ok(f"colors shape     : {tuple(out.colors.shape)}")

    # Values must be in [0, 1] (sigmoid outputs)
    assert out.points.min() >= 0.0 and out.points.max() <= 1.0, "points out of [0,1]"
    assert out.colors.min() >= 0.0 and out.colors.max() <= 1.0, "colors out of [0,1]"
    ok("All values in [0, 1] ✓")

    # batch_size / n_curves properties
    assert out.batch_size == 1 and out.n_curves == N_QUERIES
    ok(f"batch_size={out.batch_size}  n_curves={out.n_curves} ✓")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 8. TinyPrimitiveDecoder – batch > 1 ──────────────────────────────────────
section("8. TinyPrimitiveDecoder  (batch size 2)")
try:
    with torch.no_grad():
        enc_tokens2, enc_mask2 = pipeline_noProj(
            pixel_values=inputs2["pixel_values"],
            pixel_attention_mask=inputs2.get("pixel_attention_mask"),
            spatial_shapes=inputs2.get("spatial_shapes"),
        )
        out2 = decoder(enc_tokens2, enc_mask2)

    assert out2.points.shape == (2, N_QUERIES, 9, 2), f"Got {out2.points.shape}"
    assert out2.colors.shape == (2, N_QUERIES, 4),    f"Got {out2.colors.shape}"
    ok(f"points shape     : {tuple(out2.points.shape)}")
    ok(f"colors shape     : {tuple(out2.colors.shape)}")
except Exception as e:
    traceback.print_exc(); fail(str(e))

# ── 9. to_diffvg_paths ────────────────────────────────────────────────────────
section("9. to_diffvg_paths  (encoder→decoder→diffvg objects)")
try:
    import pydiffvg
    CANVAS_W, CANVAS_H = 512, 512

    shapes, groups = to_diffvg_paths(out, canvas_w=CANVAS_W, canvas_h=CANVAS_H, image_idx=0)

    assert len(shapes) == N_QUERIES, f"Expected {N_QUERIES} shapes, got {len(shapes)}"
    assert len(groups) == N_QUERIES, f"Expected {N_QUERIES} groups, got {len(groups)}"
    ok(f"Number of paths  : {len(shapes)} ✓")

    # Spot-check first path
    p = shapes[0]
    assert p.points.shape == (9, 2),      f"path points: {p.points.shape}"
    assert p.is_closed is True,           "path must be closed"
    assert p.num_control_points.tolist() == [2, 2, 2], "num_control_points mismatch"
    assert p.points[:, 0].max() <= CANVAS_W * 1.01
    assert p.points[:, 1].max() <= CANVAS_H * 1.01
    ok("Path geometry checks passed  (9 pts, closed, scaled to canvas) ✓")

    # Spot-check first group
    g = groups[0]
    assert g.fill_color.shape == (4,), f"fill_color: {g.fill_color.shape}"
    assert 0.0 <= g.fill_color.min() and g.fill_color.max() <= 1.0
    ok("ShapeGroup RGBA fill colour in [0, 1] ✓")

    # Full serialize_scene round-trip (no actual render, just serialisation)
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        CANVAS_W, CANVAS_H, shapes, groups)
    ok("pydiffvg.RenderFunction.serialize_scene  succeeded ✓")
except ImportError:
    ok("pydiffvg not available – skipping diffvg round-trip check")
except Exception as e:
    traceback.print_exc(); fail(str(e))

section("ALL TESTS PASSED")
