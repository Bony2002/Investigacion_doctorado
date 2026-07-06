"""
Example: how to use chart_encoder.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import torch
from transformers import AutoProcessor

from chart_encoder import (
    build_canvas_metadata,
    load_vision_encoder,
    ChartEncoderPipeline,
)

CKPT   = "google/siglip2-base-patch16-naflex"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Load the processor (once, outside any loop)
# ---------------------------------------------------------------------------
processor = AutoProcessor.from_pretrained(CKPT)

# ---------------------------------------------------------------------------
# 2. Load the encoder pipeline (once, outside any loop)
# ---------------------------------------------------------------------------
vision_enc = load_vision_encoder(CKPT, device=DEVICE)

pipeline = ChartEncoderPipeline(
    vision_enc,
    n_unfrozen_blocks=2,   # last 2 transformer blocks are trainable
    n_adapter_layers=2,    # 2 bottleneck adapter blocks
    decoder_dim=256,       # output token dimension
).to(DEVICE)

# ---------------------------------------------------------------------------
# 3. Pre-process a batch of chart images
# ---------------------------------------------------------------------------
images = [
    Image.open("path/to/chart_1.png").convert("RGB"),
    Image.open("path/to/chart_2.png").convert("RGB"),
]

inputs = processor(images=images, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# Optional: build coordinate metadata if you need to map predictions
# back to the original image (used later by PrimitiveAssembler).
metas = build_canvas_metadata(images, inputs["spatial_shapes"])
print(f"img[0] canvas: {metas[0].canvas_size}, scale: ({metas[0].scale_h:.3f}, {metas[0].scale_w:.3f})")

# ---------------------------------------------------------------------------
# 4. Run the encoder pipeline
# ---------------------------------------------------------------------------
with torch.no_grad():
    tokens, mask = pipeline(**inputs)

# tokens : [B, T, 256]  – projected chart tokens, ready for the decoder
# mask   : [B, T]        – 1 for real patches, 0 for padding
print(f"tokens : {tuple(tokens.shape)}")
print(f"mask   : {tuple(mask.shape)}")
