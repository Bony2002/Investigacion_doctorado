"""
TinyChartSplat-DETR-v0  –  Decoder Pipeline (Module 5)
=======================================================
Implements the DETR-style parallel decoder that consumes the patch-token
sequence produced by :class:`ChartEncoderPipeline` and predicts a fixed set
of 512 cubic Bézier blob curves, each described by:

  • 9 control points  (normalised to [0, 1])
  • 4-channel RGBA fill colour  (normalised to [0, 1])

The predicted curves are directly consumable by pydiffvg via the helper
:func:`to_diffvg_paths`.

Blob curve geometry (matches chart_rendering.py blob mode)
----------------------------------------------------------
Each blob curve is a **closed** cubic Bézier path with 3 segments:

    num_control_points = [2, 2, 2]   (2 interior control pts per segment)
    is_closed = True

Point layout per curve (9 points × 2 coords = 18 floats):

    idx  role
    0    anchor 0  (start / implicit close target)
    1    ctrl  0-1
    2    ctrl  0-2
    3    anchor 1
    4    ctrl  1-1
    5    ctrl  1-2
    6    anchor 2
    7    ctrl  2-1
    8    ctrl  2-2
    (closure back to anchor 0 is implicit in pydiffvg is_closed=True)

Architecture
------------
::

    encoder tokens [B, T, 768]
           │  (keys / values)
    ┌──────▼───────────────────────────────────────────────┐
    │  TinyPrimitiveDecoder                                │
    │                                                      │
    │  query_embed  Embedding(512, 768)                    │
    │         │                                            │
    │  ┌──────▼──────────────────────────────────────┐     │
    │  │  TransformerDecoderLayer  × n_layers         │     │
    │  │    Self-Attention  (queries ↔ queries)       │     │
    │  │    Cross-Attention (queries ↔ enc tokens)    │     │
    │  │    FFN                                       │     │
    │  └──────────────────────────────────────────────┘     │
    │         │  [B, 512, 768]                              │
    │  points_head  Linear(768→18) → Sigmoid               │
    │  color_head   Linear(768→4)  → Sigmoid               │
    └──────────────────────────────────────────────────────┘
           │
    BezierOutput  points [B,512,9,2]  colors [B,512,4]
           │
    to_diffvg_paths()
           │
    List[pydiffvg.Path], List[pydiffvg.ShapeGroup]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ===========================================================================
# BezierOutput  –  structured decoder output
# ===========================================================================

@dataclass
class BezierOutput:
    """
    Structured output of :class:`TinyPrimitiveDecoder`.

    Attributes
    ----------
    points : [B, N, 9, 2]  float32
        Predicted control-point coordinates, normalised to ``[0, 1]``.
        Scale by ``(canvas_width, canvas_height)`` before passing to
        pydiffvg.
    colors : [B, N, 4]  float32
        Predicted RGBA fill colours, normalised to ``[0, 1]``.

    N is ``n_queries`` (default 512).
    """
    points: torch.Tensor   # [B, N, 9, 2]
    colors: torch.Tensor   # [B, N, 4]

    @property
    def batch_size(self) -> int:
        return self.points.shape[0]

    @property
    def n_curves(self) -> int:
        return self.points.shape[1]


# ===========================================================================
# Module 5  –  TinyPrimitiveDecoder
# ===========================================================================

class TinyPrimitiveDecoder(nn.Module):
    """
    DETR-style parallel decoder for Bézier blob primitives.

    Takes the variable-length patch-token sequence from the encoder and
    decodes a **fixed** set of ``n_queries`` curves in a single parallel
    forward pass (no autoregression).

    Parameters
    ----------
    d_model        : Token dimension – must match the encoder output
                     (768 for SigLIP2 base with no TokenProjector).
    n_queries      : Number of Bézier curves to predict (default 512).
    n_heads        : Attention heads in each decoder layer (default 8).
    n_layers       : Number of TransformerDecoderLayer blocks (default 4).
    dim_feedforward: Inner dimension of the FFN (default 2048).
    dropout        : Dropout rate for transformer layers (default 0.1).

    Input
    -----
    tokens : [B, T, d_model]  – encoder patch tokens.
    mask   : [B, T]  bool / int  – 1 for valid tokens, 0 for padding.
             (This is the ``pixel_attention_mask`` forwarded unchanged by
             :class:`ChartEncoderPipeline`.)

    Output
    ------
    :class:`BezierOutput` with:
      points : [B, n_queries, 9, 2]
      colors : [B, n_queries, 4]

    Example
    -------
    >>> from chart_encoder import ChartEncoderPipeline, load_vision_encoder
    >>> from chart_decoder import TinyPrimitiveDecoder
    >>> enc = load_vision_encoder()
    >>> pipeline = ChartEncoderPipeline(enc)             # no projector
    >>> decoder  = TinyPrimitiveDecoder(d_model=768)
    >>> tokens, mask = pipeline(**processor_inputs)      # [B, T, 768]
    >>> out = decoder(tokens, mask)                      # BezierOutput
    >>> out.points.shape                                 # [B, 512, 9, 2]
    >>> out.colors.shape                                 # [B, 512, 4]
    """

    # 9 control points × 2 coords
    _POINTS_DIM: int = 18
    # RGBA
    _COLOR_DIM: int = 4

    def __init__(
        self,
        d_model: int = 768,
        n_queries: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.n_queries = n_queries

        # Learned object queries – one embedding per predicted curve
        self.query_embed = nn.Embedding(n_queries, d_model)

        # Standard DETR decoder: self-attn + cross-attn + FFN, stacked
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for improved training stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output heads
        self.points_head = nn.Linear(d_model, self._POINTS_DIM)
        self.color_head  = nn.Linear(d_model, self._COLOR_DIM)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.points_head.weight)
        nn.init.zeros_(self.points_head.bias)
        nn.init.xavier_uniform_(self.color_head.weight)
        nn.init.zeros_(self.color_head.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> BezierOutput:
        """
        Parameters
        ----------
        tokens : [B, T, d_model]
        mask   : [B, T]  1=valid, 0=padding  (pixel_attention_mask from
                 the processor / encoder pipeline).

        Returns
        -------
        BezierOutput
        """
        B = tokens.shape[0]

        # Expand learned queries to the batch dimension
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # [B, n_queries, d_model]

        # Build key-padding mask for the encoder memory.
        # nn.TransformerDecoder expects True where tokens should be IGNORED.
        memory_key_padding_mask: Optional[torch.Tensor] = None
        if mask is not None:
            # mask: 1=valid → invert so True=ignore padding
            memory_key_padding_mask = (mask == 0)   # [B, T]

        # Decode: queries attend to encoder tokens
        out = self.decoder(
            tgt=queries,
            memory=tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )   # [B, n_queries, d_model]

        # Predict control points and colours, squash to [0, 1]
        points = torch.sigmoid(self.points_head(out))   # [B, N, 18]
        colors = torch.sigmoid(self.color_head(out))    # [B, N, 4]

        # Reshape points to [B, N, 9, 2]
        points = points.view(B, self.n_queries, 9, 2)

        return BezierOutput(points=points, colors=colors)

    # ------------------------------------------------------------------
    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# to_diffvg_paths  –  BezierOutput → pydiffvg scene objects
# ===========================================================================

def to_diffvg_paths(
    output: BezierOutput,
    canvas_w: int,
    canvas_h: int,
    image_idx: int = 0,
) -> Tuple[list, list]:
    """
    Convert the predicted curves for a **single image** into pydiffvg scene
    objects ready for :func:`pydiffvg.RenderFunction.serialize_scene`.

    This function does **not** import pydiffvg at module level so the rest of
    the codebase remains usable in environments where pydiffvg is not
    installed.  The import happens lazily inside the function.

    Parameters
    ----------
    output    : :class:`BezierOutput` with batch dimension B ≥ image_idx+1.
    canvas_w  : Canvas width  in pixels (e.g. ``target.shape[3]``).
    canvas_h  : Canvas height in pixels (e.g. ``target.shape[2]``).
    image_idx : Which item in the batch to convert (default 0).

    Returns
    -------
    shapes       : List of ``pydiffvg.Path`` objects (length = n_curves).
    shape_groups : List of ``pydiffvg.ShapeGroup`` objects (length = n_curves).

    Example
    -------
    >>> shapes, groups = to_diffvg_paths(out, canvas_w=512, canvas_h=512)
    >>> scene_args = pydiffvg.RenderFunction.serialize_scene(
    ...     canvas_w, canvas_h, shapes, groups)
    >>> img = pydiffvg.RenderFunction.apply(canvas_w, canvas_h, 2, 2, 0,
    ...                                     None, *scene_args)
    """
    import pydiffvg  # lazy import

    points_batch = output.points[image_idx]   # [N, 9, 2]  in [0, 1]
    colors_batch = output.colors[image_idx]   # [N, 4]     in [0, 1]

    # num_control_points for 3 cubic segments = [2, 2, 2]
    num_ctrl = torch.zeros(3, dtype=torch.int32) + 2

    shapes: list       = []
    shape_groups: list = []

    for i in range(points_batch.shape[0]):
        # Scale normalised coords → canvas pixels
        pts = points_batch[i].clone().detach()          # [9, 2]
        pts[:, 0] = pts[:, 0] * canvas_w
        pts[:, 1] = pts[:, 1] * canvas_h

        path = pydiffvg.Path(
            num_control_points=num_ctrl,
            points=pts,
            stroke_width=torch.tensor(0.0),
            is_closed=True,
        )
        shapes.append(path)

        fill_color = colors_batch[i].clone().detach()   # [4]  RGBA
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=fill_color,
            stroke_color=None,
        )
        shape_groups.append(group)

    return shapes, shape_groups
