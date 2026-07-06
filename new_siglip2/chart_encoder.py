"""
TinyChartSplat-DETR-v0  –  Encoder Pipeline (Modules 2–4)
==========================================================
Implements the encoder-centric front-end of the architecture:

  Preprocessing  HuggingFace AutoProcessor  – used directly; no wrapper.
                 CanvasMetadata + build_canvas_metadata()  – coordinate helper.
  Module 2  SigLIP2NaFlexVisionEncoder – pretrained SigLIP2 NaFlex vision tower
                                         returning all patch tokens [B, T, C].
  Module 3  ChartEncoderAdapter      – lightweight bottleneck adapters for
                                       chart-domain specialisation.
  Module 4  TokenProjector           – linear projection C → decoder_dim.
  Pipeline  ChartEncoderPipeline     – composes modules 2-4.

Preprocessing pattern::

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
    inputs = processor(images=pil_images, return_tensors="pt")
    # inputs keys: pixel_values [B,N,768], pixel_attention_mask [B,N],
    #              spatial_shapes [B,2]

Checkpoint: google/siglip2-base-patch16-naflex
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor


# ===========================================================================
# Coordinate helper – CanvasMetadata
# ===========================================================================

@dataclass
class CanvasMetadata:
    """
    Per-image geometry that maps normalised predicted coordinates back to
    pixels in the original image.

    Attributes
    ----------
    original_size : (H_orig, W_orig) in pixels.
    canvas_size   : (H_canvas, W_canvas) – the pixel grid NaFlex resized
                    the image to before patchifying.
    scale_h       : H_canvas / H_orig
    scale_w       : W_canvas / W_orig
    """
    original_size: tuple[int, int]
    canvas_size:   tuple[int, int]
    scale_h:       float
    scale_w:       float


def build_canvas_metadata(
    images: List[Image.Image],
    spatial_shapes: torch.Tensor,
    patch_size: int = 16,
) -> List[CanvasMetadata]:
    """
    Build :class:`CanvasMetadata` for a batch of images using the
    ``spatial_shapes`` tensor already returned by the HuggingFace processor.

    Call this right after ``processor(images=..., return_tensors="pt")``
    if you need coordinate mapping later (e.g. in PrimitiveAssembler).

    Parameters
    ----------
    images         : Original PIL images (same order as the processor batch).
    spatial_shapes : ``inputs["spatial_shapes"]``  [B, 2]  int64  – (nh, nw).
    patch_size     : Patch size in pixels (16 for patch16 models).

    Returns
    -------
    List[CanvasMetadata], one per image.

    Example
    -------
    >>> processor = AutoProcessor.from_pretrained(CKPT)
    >>> inputs    = processor(images=pil_images, return_tensors="pt")
    >>> metas     = build_canvas_metadata(pil_images, inputs["spatial_shapes"])
    """
    metas: List[CanvasMetadata] = []
    for img, (nh, nw) in zip(images, spatial_shapes.tolist()):
        w_orig, h_orig = img.size          # PIL gives (W, H)
        h_c = nh * patch_size
        w_c = nw * patch_size
        metas.append(CanvasMetadata(
            original_size=(h_orig, w_orig),
            canvas_size=(h_c, w_c),
            scale_h=h_c / h_orig,
            scale_w=w_c / w_orig,
        ))
    return metas


# ===========================================================================
# Module 2 – SigLIP2 NaFlex Vision Encoder
# ===========================================================================

class SigLIP2NaFlexVisionEncoder(nn.Module):
    """
    Pretrained SigLIP2 NaFlex vision tower that returns **all patch tokens**
    (``last_hidden_state``) rather than the pooled CLS embedding used in the
    reference notebook.

    The encoder can be partially fine-tuned: all parameters are frozen by
    default, and the last ``n_unfrozen_blocks`` transformer blocks plus the
    final LayerNorm are made trainable.

    Parameters
    ----------
    vision_encoder   : The ``vision_model`` extracted from AutoModel
                       (see :func:`load_vision_encoder`).
    n_unfrozen_blocks: Number of trailing transformer blocks to keep
                       trainable (0 = fully frozen). Default 2.

    Input
    -----
    pixel_values          : [B, N, P²·3]  float32  – patch tokens from processor.
    pixel_attention_mask  : [B, N]         int32    – valid-patch mask.
    spatial_shapes        : [B, 2]         int64    – (nh, nw) per image.

    Output
    ------
    last_hidden_state : [B, T, C]  where C = hidden_size (1152 for base).
                        T ≤ N; only valid (non-padding) positions are kept
                        by the NaFlex encoder internally.

    Example
    -------
    >>> enc = load_vision_encoder()
    >>> model = SigLIP2NaFlexVisionEncoder(enc, n_unfrozen_blocks=2)
    >>> tokens = model(**inputs)   # inputs from AutoProcessor
    >>> tokens.shape               # e.g. (2, 256, 768)
    """

    def __init__(self,vision_encoder: nn.Module,n_unfrozen_blocks: int = 2 ) -> None:
        super().__init__()
        self.encoder     = vision_encoder
        self.hidden_size: int = vision_encoder.config.hidden_size

        self._apply_freeze_strategy(n_unfrozen_blocks)

    # ------------------------------------------------------------------
    def _apply_freeze_strategy(self, n_unfrozen_blocks: int) -> None:
        """Freeze everything, then selectively unfreeze trailing blocks."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if n_unfrozen_blocks > 0:
            transformer_layers = self.encoder.encoder.layers
            for layer in transformer_layers[-n_unfrozen_blocks:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

        # Always keep the post-encoder LayerNorm trainable when any block
        # is unfrozen, so the adapted representations are renormalized.
        if n_unfrozen_blocks > 0:
            for p in self.encoder.post_layernorm.parameters():
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None ) -> torch.Tensor:
        """Returns last_hidden_state [B, T, C]."""
        
        out = self.encoder(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        return out.last_hidden_state     # [B, T, C]

    # ------------------------------------------------------------------
    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_vision_encoder(
    checkpoint: str = "google/siglip2-base-patch16-naflex",
    device: str | torch.device = "cpu") -> nn.Module:
    """
    Load only the vision tower from a SigLIP2 checkpoint and discard the
    text encoder to free GPU memory.

    Parameters
    ----------
    checkpoint : HuggingFace model ID.
    device     : Torch device to move the encoder to after loading.

    Returns
    -------
    vision_model : ``nn.Module`` – the raw HuggingFace vision encoder.
    """
    full_model   = AutoModel.from_pretrained(checkpoint)
    vision_model = full_model.vision_model.to(device)
    del full_model
    torch.cuda.empty_cache()
    return vision_model


# ===========================================================================
# Module 3 – Chart Encoder Adapter
# ===========================================================================

class _BottleneckAdapter(nn.Module):
    """
    Single bottleneck adapter block: down-project → GELU → up-project,
    with a residual connection and pre-norm.

    Architecture:  x  →  LayerNorm  →  Linear(C, r)  →  GELU  →  Linear(r, C)
                      ↘________________________________________________↗  + x

    Parameters
    ----------
    d_model   : Token dimension C.
    reduction : Bottleneck factor; r = d_model // reduction.
    """

    def __init__(self, d_model: int, reduction: int = 8) -> None:
        super().__init__()
        r = max(1, d_model // reduction)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.act  = nn.GELU()
        self.up   = nn.Linear(r, d_model)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]  →  [B, T, C]"""
        return x + self.up(self.act(self.down(self.norm(x))))


class ChartEncoderAdapter(nn.Module):
    """
    Lightweight chart-specialisation module composed of stacked
    :class:`_BottleneckAdapter` blocks applied token-wise after the vision
    encoder.  All weights are initialised from scratch (the up-projection of
    each block is zero-initialised so the module starts as an identity).

    Parameters
    ----------
    d_model    : Input/output token dimension (must match vision encoder
                 hidden_size, i.e. 1152 for base).
    n_layers   : Number of bottleneck adapter blocks to stack (default 2).
    reduction  : Bottleneck reduction factor  r = d_model // reduction.

    Input / Output
    --------------
    [B, T, d_model]  →  [B, T, d_model]   (shape-preserving)

    Example
    -------
    >>> adapter = ChartEncoderAdapter(d_model=1152, n_layers=2)
    >>> out = adapter(patch_tokens)    # [B, T, 1152] → [B, T, 1152]
    """

    def __init__(self,d_model: int = 1152, n_layers: int = 2, reduction: int = 8) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[_BottleneckAdapter(d_model, reduction) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]  →  [B, T, C]"""
        return self.layers(x)


# ===========================================================================
# Module 4 – Token Projector
# ===========================================================================

class TokenProjector(nn.Module):
    """
    Projects visual tokens from the encoder dimension to the (smaller)
    decoder dimension using a single linear layer followed by LayerNorm.

    Parameters
    ----------
    in_dim  : Source dimension (1152 for SigLIP2 base).
    out_dim : Target dimension for the TinyPrimitiveDecoder (256).

    Input / Output
    --------------
    [B, T, in_dim]  →  [B, T, out_dim]   (T is unchanged)

    Example
    -------
    >>> proj = TokenProjector(in_dim=1152, out_dim=256)
    >>> out  = proj(adapted_tokens)    # [B, T, 1152] → [B, T, 256]
    """

    def __init__(self, in_dim: int = 1152, out_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_dim]  →  [B, T, out_dim]"""
        return self.norm(self.proj(x))


# ===========================================================================
# ChartEncoderPipeline  (composes Modules 2–4)
# ===========================================================================

class ChartEncoderPipeline(nn.Module):
    """
    End-to-end encoder pipeline: vision encoder → adapter → projector.

    Takes the raw HuggingFace ``AutoProcessor`` output and returns projected
    patch tokens ready for the TinyPrimitiveDecoder, together with the
    attention mask forwarded unchanged so downstream modules can handle
    variable-T batches.

    Parameters
    ----------
    vision_encoder    : Raw HuggingFace vision model (use
                        :func:`load_vision_encoder` to obtain it).
    n_unfrozen_blocks : Trailing transformer blocks to fine-tune (default 2).
    n_adapter_layers  : Adapter blocks in :class:`ChartEncoderAdapter`
                        (default 2).
    adapter_reduction : Bottleneck factor for the adapter (default 8).
    decoder_dim       : Target dimension of :class:`TokenProjector`.
                        Pass ``None`` (default) to skip projection entirely
                        and return tokens at the full encoder hidden size
                        (768 for base), which is preferred when the downstream
                        decoder operates at the same dimension.

    Input (keyword arguments matching processor output)
    ---------------------------------------------------
    pixel_values          : [B, N, P²·3]
    pixel_attention_mask  : [B, N]
    spatial_shapes        : [B, 2]

    Output
    ------
    tokens : [B, T, hidden]       when decoder_dim is None  (default)
    tokens : [B, T, decoder_dim]  when decoder_dim is set
    mask   : [B, T]               attention mask (same as input)

    Example
    -------
    >>> pipeline = ChartEncoderPipeline(vision_encoder)          # no projection
    >>> tokens, mask = pipeline(**inputs)
    >>> tokens.shape   # e.g. (2, 256, 768)
    >>> mask.shape     # e.g. (2, 256)

    >>> pipeline = ChartEncoderPipeline(vision_encoder, decoder_dim=256)
    >>> tokens, mask = pipeline(**inputs)
    >>> tokens.shape   # e.g. (2, 256, 256)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        n_unfrozen_blocks: int = 2,
        n_adapter_layers: int = 2,
        adapter_reduction: int = 8,
        decoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vision_enc = SigLIP2NaFlexVisionEncoder(
            vision_encoder, n_unfrozen_blocks=n_unfrozen_blocks
        )
        hidden = self.vision_enc.hidden_size
        self.adapter    = ChartEncoderAdapter(
            d_model=hidden,
            n_layers=n_adapter_layers,
            reduction=adapter_reduction,
        )
        self.projector  = (
            TokenProjector(in_dim=hidden, out_dim=decoder_dim)
            if decoder_dim is not None else None
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens = self.vision_enc(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )                                          # [B, T, C]
        tokens = self.adapter(tokens)              # [B, T, C]
        if self.projector is not None:
            tokens = self.projector(tokens)        # [B, T, decoder_dim]
        return tokens, pixel_attention_mask

