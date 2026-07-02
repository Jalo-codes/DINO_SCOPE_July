"""experiments.labs.viz — visualization helpers for DINO_SCOPE_final.

All functions return matplotlib figures or numpy images — nothing is saved to
disk unless the caller passes an output path.  No GT masks are read here; all
inputs are already-decoded patch masks and ModelInfo objects.

Typical usage::

    from experiments.labs.viz import plot_prediction, plot_attention_grid
    fig = plot_prediction(img_pil, patch_mask, info, title='test')
    fig.savefig('out.png', dpi=150, bbox_inches='tight')
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ── Image helpers ──────────────────────────────────────────────────────────────

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def mask_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a boolean patch mask on an RGB image.

    Args:
        img:   (H, W, 3) uint8 image.
        mask:  (h, w) bool patch-resolution mask — resized to (H, W) internally.
        color: RGB overlay colour.
        alpha: Blend factor for the overlay region.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    from PIL import Image as PILImage

    H, W = img.shape[:2]
    mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255, mode='L')
    mask_hw  = np.array(mask_pil.resize((W, H), PILImage.NEAREST)) > 127

    overlay = img.copy().astype(np.float32)
    for c, v in enumerate(color):
        overlay[mask_hw, c] = overlay[mask_hw, c] * (1 - alpha) + v * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# Inferno-like anchors (pos, RGB 0–255) for a matplotlib-free heat LUT. Used on
# hosts without matplotlib (e.g. the 2080 Ti box) so attention_overlay still
# colourises. Close enough to mpl 'inferno' for diagnostic reads.
_INFERNO_ANCHORS = (
    (0.00, (0,   0,   4)),   (0.13, (28,  12,  69)),
    (0.25, (74,  12,  107)), (0.38, (120, 28,  109)),
    (0.50, (165, 44,  96)),  (0.63, (207, 68,  70)),
    (0.75, (237, 105, 37)),  (0.88, (251, 155, 6)),
    (1.00, (252, 255, 164)),
)


class _LutColormap:
    """Callable mimicking a matplotlib colormap: float[..] in [0,1] → RGBA[...,4]
    floats in [0,1], via a precomputed 256-entry LUT. No matplotlib needed."""

    def __init__(self, anchors=_INFERNO_ANCHORS):
        xs = np.array([a[0] for a in anchors])
        cols = np.array([a[1] for a in anchors], dtype=np.float64) / 255.0
        grid = np.linspace(0.0, 1.0, 256)
        self._lut = np.stack(
            [np.interp(grid, xs, cols[:, c]) for c in range(3)], axis=-1
        )  # (256, 3)

    def __call__(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
        idx = (a * 255).astype(np.int64)
        rgb = self._lut[idx]                                   # (..., 3)
        return np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1)


_LUT_CMAP = None


def _colormap(name: str):
    """Colormap lookup. Prefers matplotlib; on hosts without it, returns a
    LUT-based inferno fallback so attention overlays still render."""
    global _LUT_CMAP
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except ImportError:
        pass
    except Exception:  # pragma: no cover - old matplotlib API
        try:
            import matplotlib.cm as cm
            return cm.get_cmap(name)
        except ImportError:
            pass
    if _LUT_CMAP is None:
        _LUT_CMAP = _LutColormap()
    return _LUT_CMAP


def attention_overlay(
    img: np.ndarray,
    attn: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    cmap: str = 'inferno',
    gamma: float = 0.45,
    alpha_max: float = 0.72,
    bg_min: float = 0.30,
    boxes=None,
    box_colors=None,
) -> np.ndarray:
    """Blend an attention map over an RGB image, tuned to surface faint warmth.

    Pool attention is a softmax over patches, so a couple of peaks dominate and
    everything else looks black on a raw 'hot' map.  We min-max normalise then
    apply gamma < 1, which lifts low-but-nonzero patches into visible range, and
    use a per-pixel alpha so cold regions stay transparent (image shows through)
    while warm regions glow.  Upsampling is NEAREST (locked to patch boundaries)
    so the heat reads aligned to the patch grid.  The background is also darkened
    where there is low attention so the heat glow stands out clearly.

    Args:
        img:        (H, W, 3) uint8 image.
        attn:       (N,) or (h, w) attention weights.
        grid_hw:    (h, w) patch grid the attention came from.
        cmap:       matplotlib colormap name.
        gamma:      <1 boosts faint patches; lower = more sensitive.
        alpha_max:  peak overlay opacity.
        bg_min:     baseline brightness fraction for background (where attention is 0).
        boxes:      optional fractional bboxes to draw on top.
        box_colors: per-box RGB (or single tuple).

    Returns:
        (H, W, 3) uint8 image with the attention blended in.
    """
    from PIL import Image as PILImage

    H, W = img.shape[:2]
    n = grid_hw[0] * grid_hw[1]
    a = np.asarray(attn, dtype=np.float64).reshape(-1)[:n].reshape(grid_hw)

    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    a = np.power(a, gamma)

    a_img = np.asarray(
        PILImage.fromarray((a * 255).astype(np.uint8)).resize((W, H), PILImage.NEAREST),
        dtype=np.float64,
    ) / 255.0

    rgb = _colormap(cmap)(a_img)[..., :3] * 255.0          # (H, W, 3)
    alpha = (a_img * alpha_max)[..., None]                  # (H, W, 1)

    # Darken the background image where there is low attention
    bg_dim = bg_min + (1.0 - bg_min) * a_img[..., None]

    out = img.astype(np.float64) * bg_dim * (1.0 - alpha) + rgb * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    if boxes:
        out = draw_bboxes(out, boxes, colors=box_colors)
    return out


# ── Bounding boxes ─────────────────────────────────────────────────────────────

def draw_bboxes(
    img: np.ndarray,
    boxes,
    *,
    colors=None,
    labels=None,
    width: int = 3,
):
    """Draw fractional bboxes (y0, x0, y1, x1) on an RGB image.

    Args:
        img:    (H, W, 3) uint8 image.
        boxes:  iterable of (y0, x0, y1, x1) in [0, 1] fractions.
        colors: per-box RGB tuple, or a single tuple for all (default red).
        labels: optional per-box text labels drawn at the top-left corner.
        width:  rectangle line thickness in pixels.

    Returns:
        (H, W, 3) uint8 image with rectangles drawn (a copy).
    """
    from PIL import Image as PILImage, ImageDraw

    H, W = img.shape[:2]
    pil  = PILImage.fromarray(img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(pil)

    boxes = list(boxes)
    if colors is None:
        colors = [(220, 30, 30)] * len(boxes)
    elif isinstance(colors, tuple):
        colors = [colors] * len(boxes)

    for i, (y0, x0, y1, x1) in enumerate(boxes):
        left, right = int(round(x0 * W)), int(round(x1 * W))
        upper, lower = int(round(y0 * H)), int(round(y1 * H))
        col = tuple(colors[i % len(colors)])
        draw.rectangle([left, upper, right, lower], outline=col, width=width)
        if labels and i < len(labels) and labels[i]:
            draw.text((left + 2, max(0, upper - 12)), str(labels[i]), fill=col)

    return np.array(pil)


def _composite_panels_pil(panels, *, title='', pad=10, header=22,
                          bg=(30, 30, 30)):
    """Paste a list of (title, HxWx3 uint8 array) panels into one labelled PIL
    canvas. Panels are scaled to a common height (preserving aspect, so zoom
    crops are not distorted) and laid out left-to-right. Returns a PIL.Image.
    The matplotlib-free counterpart to a plt.subplots(1, n) row.

    The common panel height is capped by the VIZ_MAX_PANEL_H env var (px) when
    set — this keeps batch-generated diagnostic PNGs small (the model only sees
    448px anyway), so committing hundreds of them to git stays manageable."""
    import os
    from PIL import Image, ImageDraw, ImageFont

    H = max(int(np.asarray(a).shape[0]) for _, a in panels)
    _cap = os.environ.get('VIZ_MAX_PANEL_H')
    if _cap:
        try:
            H = min(H, int(_cap))
        except ValueError:
            pass
    pil_panels = []
    for t, arr in panels:
        p = Image.fromarray(np.asarray(arr).astype(np.uint8)).convert('RGB')
        if p.height != H:
            w = max(1, int(round(p.width * H / p.height)))
            p = p.resize((w, H), Image.BILINEAR)
        pil_panels.append((t, p))

    n = len(pil_panels)
    foot = 18 if title else 0
    canvas_w = sum(p.width for _, p in pil_panels) + (n + 1) * pad
    canvas_h = H + header + pad + foot
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x = pad
    for t, p in pil_panels:
        canvas.paste(p, (x, header))
        if font is not None and t:
            try:
                tw = draw.textlength(t, font=font)
            except AttributeError:
                try:
                    tw, _ = draw.textsize(t, font=font)
                except Exception:
                    tw = len(t) * 6
            draw.text((x + max(0, (p.width - int(tw)) // 2), 6), t,
                      fill=(235, 235, 235), font=font)
        x += p.width + pad

    if title and font is not None:
        draw.text((pad, canvas_h - 14), title, fill=(180, 180, 180), font=font)
    return canvas


def plot_hdbscan_result(
    img_pil,
    patch_mask: np.ndarray,
    info,
    *,
    gt_mask: Optional[np.ndarray] = None,
    zoom_mask: Optional[np.ndarray] = None,
    gt_box=None,
    crop_box=None,
    crop_pil=None,
    attn_crop: Optional[np.ndarray] = None,
    crop_grid_hw: Optional[Tuple[int, int]] = None,
    title: str = '',
    figsize: Tuple[float, float] = (28, 4),
    decoder_name: str = 'hdbscan',
    **_ignored,
):
    """Seven-panel HDBSCAN result, built around shape + the zoom's-eye view:

        input+window | MIL attention | zoom crop | zoom attention |
        flat HDBSCAN | zoom HDBSCAN | GT

    Attention maps are blended over the image (sensitive to faint warmth), with
    the zoom window drawn on top so the box can be sanity-checked against the
    heat that produced it.  The zoom crop panel shows what the model actually
    sees on pass 2, and the zoom-attention panel shows the pass-2 attention on
    that crop.  Masks are overlaid as shapes — the zoom mask is pixel-resolution.

    Args:
        img_pil:      PIL RGB input image.
        patch_mask:   flat (full-frame) HDBSCAN prediction (n_side, n_side) bool.
        info:         ModelInfo — full-frame attention + grid_hw.
        gt_mask:      (H, W) bool pixel GT — GT panel.
        zoom_mask:    (H, W) bool pixel-res placed-back zoom prediction; None when
                      no real zoom happened.
        gt_box:       GT bbox (green) for panel 1.
        crop_box:     attention-zoom window bbox (yellow) for panels 1 & 2.
        crop_pil:     the pass-2 crop image (what the model sees zoomed in).
        attn_crop:    pass-2 attention (N,) for the zoom-attention panel.
        crop_grid_hw: pass-2 patch grid shape.
        title:        Suptitle.

    Returns:
        matplotlib.figure.Figure if matplotlib is importable, else PIL.Image.Image
        (same 7 panels either way — the PIL path is for hosts without matplotlib).
    """
    img = np.array(img_pil.convert('RGB'))
    yellow = (240, 200, 0)

    def _attn_map(src, gh_fallback=None):
        attn = src.get('attention') if isinstance(src, dict) else getattr(src, 'attention', None)
        if attn is None:
            return None, None
        gh = (src.get('grid_hw') if isinstance(src, dict) else getattr(src, 'grid_hw', None))
        attn = np.asarray(attn).reshape(-1)
        gh = gh or gh_fallback or _square_grid(attn.size)
        return attn, gh

    crop_img = np.array(crop_pil.convert('RGB')) if crop_pil is not None else None

    # Panel 1 — input + zoom window (yellow) + GT box (green)
    boxes, colors, labels = [], [], []
    if crop_box is not None:
        boxes.append(crop_box); colors.append(yellow); labels.append('zoom')
    if gt_box is not None:
        boxes.append(gt_box); colors.append((30, 200, 30)); labels.append('gt')
    p1 = draw_bboxes(img, boxes, colors=colors, labels=labels) if boxes else img

    # Panel 2 — MIL attention blended over the image, with the zoom window on top
    full_attn, full_gh = _attn_map(info)
    if full_attn is not None:
        win = [crop_box] if crop_box is not None else None
        p2, t2 = attention_overlay(img, full_attn, full_gh,
                                   boxes=win, box_colors=yellow), 'MIL attention'
    else:
        p2, t2 = img, '(no attention head)'

    # Panel 3 — what the model sees at the zoom (pass-2 crop)
    if crop_img is not None:
        p3, t3 = crop_img, 'zoom crop (model input)'
    else:
        p3, t3 = img, 'zoom crop (no zoom)'

    # Panel 4 — pass-2 attention over the crop
    if crop_img is not None and attn_crop is not None:
        gh = crop_grid_hw or _square_grid(np.asarray(attn_crop).size)
        p4, t4 = attention_overlay(crop_img, attn_crop, gh), 'zoom attention'
    else:
        p4, t4 = (img if crop_img is None else crop_img), 'zoom attention (n/a)'

    # Panel 5 — flat full-frame cluster overlay
    p5 = mask_overlay(img, patch_mask, color=(220, 30, 30))
    t5 = f'flat {decoder_name.upper()}'

    # Panel 6 — pixel-resolution placed-back zoom cluster overlay
    if zoom_mask is not None:
        p6, t6 = mask_overlay(img, zoom_mask, color=(0, 180, 220)), \
            f'zoom {decoder_name.upper()} (pixel)'
    else:
        p6, t6 = img, f'zoom {decoder_name.upper()} (fell back)'

    # Panel 7 — GT mask overlay
    if gt_mask is not None:
        p7, t7 = mask_overlay(img, gt_mask, color=(30, 200, 30)), 'GT mask'
    else:
        p7, t7 = img, '(no GT — real image)'

    panels = [('input + zoom window', p1), (t2, p2), (t3, p3), (t4, p4),
              (t5, p5), (t6, p6), (t7, p7)]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # No matplotlib on this host (e.g. the box) — composite with PIL.
        return _composite_panels_pil(panels, title=title)

    fig, axes = plt.subplots(1, 7, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=11)
    for ax, (t, arr) in zip(axes, panels):
        ax.imshow(arr); ax.set_title(t); ax.axis('off')
    plt.tight_layout()
    return fig


def plot_multi_zoom_result(
    img_pil,
    attn1: Optional[np.ndarray],
    grid_hw: Tuple[int, int],
    boxes,
    union_mask: Optional[np.ndarray],
    *,
    attn2: Optional[np.ndarray] = None,
    attn2_grid_hw: Optional[Tuple[int, int]] = None,
    attn_combined: Optional[np.ndarray] = None,
    attn_combined_grid_hw: Optional[Tuple[int, int]] = None,
    gt_mask: Optional[np.ndarray] = None,
    gt_box=None,
    box_labels=None,
    box_colors=None,
    frames=None,
    full_score: Optional[float] = None,
    title: str = '',
    decoder_name: str = 'kmeans',
    panel_size: float = 5.0,
):
    """Dynamic multi-panel zoom result — one panel per signal that's available, so
    you can see exactly what drove the boxes.

    Panels (only those with data are drawn):
        input + boxes(+GT box) | pass-1 attention | post-hide attention |
        combined (additive) attention | per-frame crop predictions (scored) |
        union zoom mask (pixel) | GT mask

    `frames` is a list of per-crop dicts (the candidate zoom windows): each draws
    the crop with ITS OWN predicted mask overlaid, titled with the crop's MIL
    image score (model-only) and whether the gate KEPT or DROPPED it.  This is how
    you read the selection logic.

    NO ORACLE, ever: per-frame scores are the model's `image_logit`/score for that
    crop — never a GT-IoU ranking, never a "best window" chosen by GT.  GT is used
    ONLY for the final GT-mask panel / box (the caller passes the already-scored
    gt_mask); it never selects, orders, or labels a window.

    The final boxes are overlaid on every attention panel so you can check them
    against each stage.

    Args:
        img_pil:        PIL RGB input.
        attn1:          (N,)|(h,w) pass-1 MIL attention (None ⇒ panel skipped).
        grid_hw:        patch grid for attn1.
        boxes:          final fractional bboxes that were zoomed (drawn everywhere).
        union_mask:     (H,W) bool pixel union of placed-back masks (or None).
        attn2:          post-hide re-pool attention (second_best).
        attn_combined:  fused pass-1+post-hide attention (combined mode).
        gt_mask/gt_box: GT panel / box.
        box_labels:     per-box labels (e.g. 'top','2nd','r0','hidden','guided').
        box_colors:     per-box RGB; defaults to a fixed palette.
        panel_size:     inches per square panel.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    img = np.array(img_pil.convert('RGB'))
    boxes = list(boxes or [])

    if box_colors is None:
        palette = [(240, 200, 0), (0, 200, 240), (240, 100, 0), (180, 0, 220)]
        box_colors = [palette[i % len(palette)] for i in range(max(1, len(boxes)))]
    elif isinstance(box_colors, tuple):
        box_colors = [box_colors] * max(1, len(boxes))
    bc = box_colors[:len(boxes)] or None

    # Assemble (title, image) panels — only those with data.
    panels: List[Tuple[str, np.ndarray]] = []

    p1_boxes = list(boxes)
    p1_colors = list(box_colors[:len(boxes)])
    p1_labels = list(box_labels) if box_labels else [None] * len(boxes)
    if gt_box is not None:
        p1_boxes = p1_boxes + [gt_box]
        p1_colors = p1_colors + [(30, 200, 30)]
        p1_labels = p1_labels + ['gt']
    panels.append(('input + boxes',
                   draw_bboxes(img, p1_boxes, colors=p1_colors, labels=p1_labels)
                   if p1_boxes else img))

    if attn1 is not None:
        panels.append(('pass-1 attention',
                       attention_overlay(img, attn1, grid_hw, boxes=boxes or None, box_colors=bc)))
    if attn2 is not None:
        panels.append(('post-hide attention',
                       attention_overlay(img, attn2, attn2_grid_hw or grid_hw,
                                         boxes=boxes or None, box_colors=bc)))
    if attn_combined is not None:
        panels.append(('combined (additive)',
                       attention_overlay(img, attn_combined, attn_combined_grid_hw or grid_hw,
                                         boxes=boxes or None, box_colors=bc)))

    # Per-frame candidate windows: each crop with its OWN prediction + the model's
    # MIL score and the gate verdict.  Scores are model-only (no GT / no oracle).
    for fr in (frames or []):
        crop_pil = fr.get('crop_pil')
        if crop_pil is None:
            continue
        crop_arr = np.array(crop_pil.convert('RGB'))
        mask_c = fr.get('mask_crop')
        kept = fr.get('kept', True)
        verdict_color = (0, 180, 220) if kept else (210, 70, 70)
        pimg = (mask_overlay(crop_arr, np.asarray(mask_c, dtype=bool), color=verdict_color)
                if mask_c is not None else crop_arr)
        score = fr.get('score')
        score_txt = f'p={score:.2f}' if score is not None else 'p=n/a'
        verdict = 'KEPT' if kept else 'DROPPED'
        panels.append((f"{fr.get('label', 'box')} · {score_txt} · {verdict}", pimg))

    if union_mask is not None:
        panels.append((f'union zoom {decoder_name.upper()} (px)',
                       mask_overlay(img, np.asarray(union_mask, dtype=bool), color=(0, 180, 220))))
    else:
        panels.append((f'zoom {decoder_name.upper()} (fell back)', img))

    if gt_mask is not None and np.asarray(gt_mask).any():
        panels.append(('GT mask',
                       mask_overlay(img, np.asarray(gt_mask, dtype=bool), color=(30, 200, 30))))
    else:
        panels.append(('(no GT — real image)', img))

    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(panel_size * k, panel_size))
    if k == 1:
        axes = [axes]
    suptitle = title
    if full_score is not None:
        ref = f'full-frame MIL p={full_score:.2f} (gate reference)'
        suptitle = f'{title}\n{ref}' if title else ref
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    for ax, (ptitle, pimg) in zip(axes, panels):
        ax.imshow(pimg)
        ax.set_title(ptitle, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_box_policy_result(
    img_pil,
    boxes,
    keep_prob: Optional[np.ndarray],
    grid_hw: Tuple[int, int],
    *,
    candidates: Optional[np.ndarray] = None,
    attn: Optional[np.ndarray] = None,
    union_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    gt_box=None,
    title: str = '',
    panel_size: float = 5.0,
):
    """Box-policy result: chosen boxes + the keep-probability field that drove them.

    Panels:
        input + chosen boxes (+GT box) | keep-prob heatmap (+candidates outline) |
        MIL attention | union zoom mask (pixel) | GT mask

    The keep-prob panel is the policy's actual per-patch "put a box here" field —
    the prediction the user asked to see alongside the chosen boxes.  Candidate
    patches (the prefilter) are tinted so you can tell which locations were even
    eligible.  No GT drives any panel except the final GT overlay/box.
    """
    import matplotlib.pyplot as plt

    img = np.array(img_pil.convert('RGB'))
    boxes = list(boxes or [])
    red = (220, 30, 30)
    green = (30, 200, 30)

    panels: List[Tuple[str, np.ndarray]] = []

    # Panel 1 — input + chosen boxes (+ GT box)
    p1_boxes = list(boxes)
    p1_colors = [red] * len(boxes)
    p1_labels = [f'b{i}' for i in range(len(boxes))]
    if gt_box is not None:
        p1_boxes.append(gt_box); p1_colors.append(green); p1_labels.append('gt')
    panels.append(('input + boxes',
                   draw_bboxes(img, p1_boxes, colors=p1_colors, labels=p1_labels)
                   if p1_boxes else img))

    # Panel 2 — keep-probability field (the box prediction) + candidate tint
    if keep_prob is not None:
        n = grid_hw[0] * grid_hw[1]
        kp = np.asarray(keep_prob, dtype=np.float64).reshape(-1)[:n].copy()
        if candidates is not None:
            cand = np.asarray(candidates, dtype=bool).reshape(-1)[:n]
            kp[~cand] = 0.0   # show only eligible locations' probability
        panels.append(('keep-prob (boxes)',
                       attention_overlay(img, kp, grid_hw, cmap='viridis',
                                         boxes=boxes or None, box_colors=red)))

    # Panel 3 — MIL attention
    if attn is not None:
        panels.append(('MIL attention',
                       attention_overlay(img, attn, grid_hw,
                                         boxes=boxes or None, box_colors=(240, 200, 0))))

    # Panel 4 — union zoom mask (pixel resolution)
    if union_mask is not None:
        panels.append(('union zoom (px)',
                       mask_overlay(img, np.asarray(union_mask, dtype=bool), color=(0, 180, 220))))
    else:
        panels.append(('zoom (fell back)', img))

    # Panel 5 — GT
    if gt_mask is not None and np.asarray(gt_mask).any():
        panels.append(('GT mask', mask_overlay(img, np.asarray(gt_mask, dtype=bool), color=green)))
    else:
        panels.append(('(no GT — real)', img))

    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(panel_size * k, panel_size))
    if k == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=11)
    for ax, (ptitle, pimg) in zip(axes, panels):
        ax.imshow(pimg)
        ax.set_title(ptitle, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    return fig


def _square_grid(n: int) -> Tuple[int, int]:
    s = int(round(n ** 0.5))
    return s, s


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_prediction(
    img_pil,
    patch_mask: np.ndarray,
    info,
    *,
    title: str = '',
    gt_mask: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (12, 4),
):
    """Three/four-panel figure: input | prediction | [attention] | [GT mask].

    If matplotlib is available, returns a matplotlib.figure.Figure.
    Otherwise, returns a PIL.Image.Image.
    """
    show_attn = info is not None and info.attention is not None
    show_gt = gt_mask is not None

    try:
        import matplotlib.pyplot as plt
        use_matplotlib = True
    except ImportError:
        use_matplotlib = False

    if use_matplotlib:
        img = np.array(img_pil.convert('RGB'))
        H, W = img.shape[:2]

        panels = ['input', 'prediction']
        if show_attn:
            panels.append('attention')
        if show_gt:
            panels.append('gt')

        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4.0))
        if n_panels == 1:
            axes = [axes]
        if title:
            fig.suptitle(title, fontsize=11)

        axes[0].imshow(img)
        axes[0].set_title('input')
        axes[0].axis('off')

        pred_vis = mask_overlay(img, patch_mask, color=(220, 30, 30), alpha=0.45)
        axes[1].imshow(pred_vis)
        axes[1].set_title('prediction')
        axes[1].axis('off')

        curr_idx = 2
        if show_attn:
            n  = info.grid_hw[0] * info.grid_hw[1]
            attn_map = info.attention[:n].reshape(info.grid_hw)
            axes[curr_idx].imshow(attn_map, cmap='hot', interpolation='nearest')
            axes[curr_idx].set_title('attention')
            axes[curr_idx].axis('off')
            curr_idx += 1

        if show_gt:
            gt_vis = mask_overlay(img, gt_mask, color=(30, 200, 30), alpha=0.45)
            axes[curr_idx].imshow(gt_vis)
            axes[curr_idx].set_title('GT mask')
            axes[curr_idx].axis('off')
            curr_idx += 1

        plt.tight_layout()
        return fig
    else:
        from PIL import Image, ImageDraw, ImageFont

        # Panel 1: Original input image
        p1 = img_pil.convert('RGB')
        W, H = p1.size

        # Panel 2: Prediction overlay (translucent red)
        pred_patch = np.asarray(patch_mask, dtype=bool)
        n_side = info.grid_hw[0] if info is not None else int(round(pred_patch.size ** 0.5))
        pred_2d = pred_patch.reshape(n_side, n_side) if pred_patch.ndim == 1 else pred_patch
        
        mask_pil = Image.fromarray((pred_2d.astype(np.uint8) * 255), mode='L')
        mask_hw = mask_pil.resize((W, H), Image.NEAREST)
        
        red_img = Image.new('RGB', (W, H), color=(220, 30, 30))
        alpha_mask = Image.fromarray((np.array(mask_hw) * 0.45).astype(np.uint8), mode='L')
        p2 = Image.composite(red_img, p1, alpha_mask)

        # Assemble list of panels: (title, image)
        panels_to_paste = [
            ("input", p1),
            ("prediction", p2)
        ]

        if show_attn:
            n = info.grid_hw[0] * info.grid_hw[1]
            attn = info.attention[:n].reshape(info.grid_hw)
            lo, hi = float(attn.min()), float(attn.max())
            attn_norm = (attn - lo) / (hi - lo) if hi > lo else np.zeros_like(attn)
            attn_norm = np.power(attn_norm, 0.45) # gamma correction
            
            attn_pil = Image.fromarray((attn_norm * 255).astype(np.uint8), mode='L')
            attn_hw = np.array(attn_pil.resize((W, H), Image.NEAREST)) / 255.0

            # Heatmap color computation
            r = np.clip(3.0 * attn_hw, 0.0, 1.0)
            g = np.clip(3.0 * attn_hw - 1.0, 0.0, 1.0)
            b = np.clip(3.0 * attn_hw - 2.0, 0.0, 1.0)
            rgb_heatmap = np.stack([r, g, b], axis=-1) * 255.0
            
            # Blend
            alpha_max = 0.72
            bg_min = 0.30
            bg_dim = bg_min + (1.0 - bg_min) * attn_hw
            img_arr = np.array(p1, dtype=np.float32)
            dimmed_img = img_arr * bg_dim[..., None]
            
            alpha = (attn_hw * alpha_max)[..., None]
            out_arr = dimmed_img * (1.0 - alpha) + rgb_heatmap * alpha
            p_attn = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), mode='RGB')
            panels_to_paste.append(("attention", p_attn))

        if show_gt:
            green_img = Image.new('RGB', (W, H), color=(30, 200, 30))
            gt_mask_pil = Image.fromarray(gt_mask.astype(np.uint8) * 255, mode='L')
            if gt_mask_pil.size != (W, H):
                gt_mask_pil = gt_mask_pil.resize((W, H), Image.NEAREST)
            alpha_gt = Image.fromarray((np.array(gt_mask_pil) * 0.45).astype(np.uint8), mode='L')
            p_gt = Image.composite(green_img, p1, alpha_gt)
            panels_to_paste.append(("GT mask", p_gt))

        # Canvas preparation
        num_p = len(panels_to_paste)
        pad_x = 10
        header_y = 40
        canvas_w = num_p * W + (num_p + 1) * pad_x
        canvas_h = H + header_y + pad_x
        
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Paste each panel and write text
        for idx, (p_title, p_img) in enumerate(panels_to_paste):
            x_offset = (idx + 1) * pad_x + idx * W
            canvas.paste(p_img, (x_offset, header_y))
            
            if font is not None:
                try:
                    tw = draw.textlength(p_title, font=font)
                except AttributeError:
                    try:
                        tw, _ = draw.textsize(p_title, font=font)
                    except Exception:
                        tw = len(p_title) * 6
                tx = x_offset + (W - tw) // 2
                draw.text((tx, 12), p_title, fill=(240, 240, 240), font=font)

        if title:
            draw.text((pad_x, canvas_h - 20), title, fill=(180, 180, 180), font=font)

        return canvas


def plot_attention_grid(
    infos: List,
    *,
    n_cols: int = 4,
    figsize_per: Tuple[float, float] = (3, 3),
    cmap: str = 'hot',
    titles: Optional[List[str]] = None,
):
    """Grid of attention maps for a list of ModelInfo objects.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n = len(infos)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
    )
    axes = np.array(axes).reshape(-1)

    for i, info in enumerate(infos):
        ax = axes[i]
        if info.attention is not None:
            nm = info.grid_hw[0] * info.grid_hw[1]
            attn_map = info.attention[:nm].reshape(info.grid_hw)
            ax.imshow(attn_map, cmap=cmap, interpolation='nearest')
        else:
            ax.text(0.5, 0.5, 'no attn', ha='center', va='center',
                    transform=ax.transAxes)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
        ax.axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = 'patch embeddings',
    figsize: Tuple[float, float] = (7, 6),
):
    """2D UMAP scatter of patch embeddings coloured by label.

    Requires `umap-learn` installed.

    Args:
        embeddings: (N, D) float32 L2-normalized patch embeddings.
        labels:     (N,) int {0=real, 1=splice} or cluster ids.
        title:      Figure title.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    try:
        import umap
    except ImportError as exc:
        raise ImportError('plot_embedding_umap requires umap-learn: pip install umap-learn') from exc

    reducer = umap.UMAP(n_components=2, random_state=0)
    z2d     = reducer.fit_transform(embeddings.astype(np.float32))

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(z2d[:, 0], z2d[:, 1], c=labels, cmap='tab10',
                         s=10, alpha=0.7, linewidths=0)
    plt.colorbar(scatter, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ── inline display (notebook + graphics terminals) ───────────────────────────────

def figure_to_png_bytes(fig, *, dpi: int = 130) -> bytes:
    """Render a matplotlib Figure to PNG bytes (no file written)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def display_image_inline(image, *, max_cols: Optional[int] = None) -> bool:
    """Show a PNG (bytes) or matplotlib Figure inline in the current frontend.

    Tries, in order: an IPython/Colab kernel, then a graphics-capable terminal —
    iTerm2's inline-image protocol, kitty's ``icat``, or ``chafa`` (sixel/unicode).
    Returns True if it actually displayed, False if no channel was available so
    the caller can fall back to saving / printing a path.

    Reusable beyond the zoom labs — any script with a Figure can call this to get
    a picture into the terminal without assuming a notebook.
    """
    from PIL import Image as PILImage
    if hasattr(image, 'savefig'):
        png = figure_to_png_bytes(image)
    elif isinstance(image, PILImage.Image):
        import io
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        png = buf.getvalue()
    elif isinstance(image, (bytes, bytearray)):
        png = bytes(image)
    else:
        raise TypeError('display_image_inline expects a Figure, PIL Image, or PNG bytes')

    # 1) IPython / Colab kernel (works when called from a notebook cell).
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import Image as _Img, display as _display
            _display(_Img(data=png))
            return True
    except Exception:
        pass

    # 2) graphics-capable terminal — only when stdout is an interactive TTY.
    import sys
    if not sys.stdout.isatty():
        return False

    import base64
    import os
    import shutil
    import subprocess
    import tempfile

    term_program = os.environ.get('TERM_PROGRAM', '')
    term = os.environ.get('TERM', '')

    # iTerm2 inline-image escape sequence.
    if term_program == 'iTerm.app' or os.environ.get('LC_TERMINAL') == 'iTerm2':
        b64 = base64.b64encode(png).decode('ascii')
        sys.stdout.write(f'\033]1337;File=inline=1;size={len(png)};width=auto:{b64}\a\n')
        sys.stdout.flush()
        return True

    def _via_tool(argv) -> bool:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
            tf.write(png)
            path = tf.name
        try:
            subprocess.run(argv + [path], check=False)
            return True
        except Exception:
            return False
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # kitty graphics protocol via the icat kitten.
    if os.environ.get('KITTY_WINDOW_ID') or 'kitty' in term:
        exe = shutil.which('kitten') or shutil.which('kitty')
        if exe:
            argv = [exe, 'icat', '--align', 'left'] if exe.endswith('kitten') \
                else [exe, '+kitten', 'icat', '--align', 'left']
            return _via_tool(argv)

    # chafa: sixel / unicode blocks, works in many terminals.
    if shutil.which('chafa'):
        argv = ['chafa'] + (['--size', f'{max_cols}x'] if max_cols else [])
        return _via_tool(argv)

    return False
