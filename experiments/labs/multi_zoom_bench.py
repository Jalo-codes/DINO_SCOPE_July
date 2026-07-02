"""experiments.labs.multi_zoom_bench — A/B the zoom architectures.

Runs the same items through the zoom modes and prints a side-by-side comparison
so we can see whether multi-window / second-best actually beats the single-box
baseline:

    single      — one attention bbox (experiments.labs.attention_zoom)
    multi       — efficient box cover over the attention hot set, gated (workhorse)
    second_best — top bbox + a second box from MIL-hiding region 1 (PAUSED)

All go through the shared fetch → decode → metric → aggregate seam; GT is
touched only inside metric() (I3).  Model forward stays in fetch.model_info (I2)
— this lab calls the *_eval functions, never the model directly.
"""

from typing import Dict, List, Sequence

import torch

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.aggregate import decoder_bench, summarize
from lab_utils.eval.record import EvalRecord
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model

# The mode registry is owned by the lab so any caller can dispatch by name;
# this harness just consumes it.
from experiments.labs.attention_zoom import (
    ZOOM_EVAL_FNS,
    ZOOM_MODES,
    ZOOM_SINGLE_FNS,
)

_MODES = ZOOM_EVAL_FNS
_SINGLE_FNS = ZOOM_SINGLE_FNS
ALL_MODES = ZOOM_MODES


@torch.no_grad()
def multi_zoom_bench(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    decoder='kmeans',
    modes: Sequence[str] = ALL_MODES,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    single_zoom_kwargs: dict = None,
    log_tag: str = '[zoom]',
) -> Dict[str, List[EvalRecord]]:
    """Run each requested zoom mode over `items`; print per-mode summaries and a
    cross-mode bench table.  Returns {mode: [EvalRecord]}.

    `single_zoom_kwargs` (e.g. {'attn_min_pad_frac': ...}) are forwarded ONLY to
    the vanilla 'single' mode — the crop-window tuning lives on
    attention_zoom_single; the multi/second/hide single-fns take fixed
    keyword-only args and would reject them.
    """
    bare = unwrap_model(model)
    bare.eval()
    single_zoom_kwargs = single_zoom_kwargs or {}

    records_by_mode: Dict[str, List[EvalRecord]] = {}
    for mode in modes:
        fn = _MODES.get(mode)
        if fn is None:
            raise ValueError(f'multi_zoom_bench: unknown mode {mode!r} '
                             f'(choose from {sorted(_MODES)})')
        extra = single_zoom_kwargs if mode == 'single' else {}
        log_line(f'{log_tag} mode={mode} decoder={decoder} n_items={len(items)}')
        recs = fn(
            bare, items, res,
            device=device, use_amp=use_amp, decoder=decoder,
            log_tag=f'{log_tag}:{mode}', summarize_results=False, amp_dtype=amp_dtype,
            **extra,
        )
        summarize(recs, log_tag=log_tag, tag=mode)
        records_by_mode[mode] = recs

    if len(records_by_mode) > 1:
        log_line(f'{log_tag} mode bench:')
        decoder_bench(records_by_mode, log_tag=log_tag)
    return records_by_mode


# ── per-item visualisation ───────────────────────────────────────────────────────

def _normalize_debug(mode: str, debug: dict):
    """Per-mode debug dict → (boxes, box_labels, attn2, attn2_grid_hw, attn_combined)."""
    gh = debug.get('grid_hw')
    if mode == 'single':
        b = debug.get('bbox')
        boxes = [b] if b is not None else []
        return boxes, ['zoom'][:len(boxes)], None, None, None
    if mode == 'multi':
        boxes = list(debug.get('bboxes') or [])
        return boxes, [f'w{i}' for i in range(len(boxes))], None, None, None
    if mode == 'second_best':
        boxes = [b for b in (debug.get('bbox1'), debug.get('bbox2')) if b is not None]
        return boxes, ['top', '2nd'][:len(boxes)], debug.get('attn2'), gh, None
    return [], [], None, None, None


def _sigmoid(x):
    """sigmoid(logit) → prob, or None when the image head is disabled."""
    import math
    if x is None:
        return None
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if float(x) < 0 else 1.0


def _frames_from_debug(debug: dict, labels) -> List[dict]:
    """Per-crop candidate windows for the viz: crop image + its own prediction +
    the model's MIL score + the gate verdict.

    Scores are the model's per-crop image_logit ONLY — no GT, no oracle ranking.
    `gated_boxes` (indices kept by the gate) marks KEPT vs DROPPED; None means
    gating was off/skipped → every crop is KEPT.
    """
    per_box = debug.get('per_box') or []
    gated = debug.get('gated_boxes')
    kept_set = set(gated) if gated is not None else set(range(len(per_box)))
    frames: List[dict] = []
    for i, pb in enumerate(per_box):
        frames.append({
            'crop_pil':  pb.get('crop_pil'),
            'mask_crop': pb.get('mask_crop'),
            'score':     _sigmoid(pb.get('image_logit')),
            'logit':     pb.get('image_logit'),
            'label':     labels[i] if labels and i < len(labels) else f'box{i}',
            'kept':      i in kept_set,
        })
    return frames


def _emit_figure(fig, out_path, show: bool) -> bool:
    """Save fig to out_path (if given) and/or display it inline.  Returns True if
    it was actually displayed — possible in an IPython/Colab kernel OR a
    graphics-capable terminal (iTerm2 / kitty / chafa)."""
    import matplotlib.pyplot as plt
    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    shown = False
    if show:
        from experiments.labs.viz import display_image_inline
        try:
            shown = display_image_inline(fig)
        except Exception:
            shown = False
    plt.close(fig)
    return shown


@torch.no_grad()
def run_zoom_viz(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    mode: str = 'second_best',
    device: torch.device,
    decoder='kmeans',
    out_dir=None,
    viz_n: int = 0,
    show: bool = False,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    single_zoom_kwargs: dict = None,
    log_tag: str = '[zoom]',
) -> List[EvalRecord]:
    """Run one zoom mode over `items`, scoring all of them and rendering the
    first `viz_n` non-real items with `plot_multi_zoom_result`.

    Figures are saved to `out_dir` (if given) and/or displayed inline when
    `show=True` AND the process is inside an IPython/Colab kernel.  Under a plain
    ``!python`` subprocess inline display is impossible — figures still save, and
    a one-time note explains how to display (call this in a cell).

    GT is read only from the scored EvalRecord (rec.gt_mask) for the GT panel —
    it never influences a prediction.
    """
    from pathlib import Path

    from tqdm import tqdm

    from lab_utils.eval.aggregate import summarize
    from lab_utils.eval.zoom import mask_to_bbox
    from experiments.labs.viz import plot_multi_zoom_result

    single_fn = _SINGLE_FNS.get(mode)
    if single_fn is None:
        raise ValueError(f'run_zoom_viz: unknown mode {mode!r} (choose from {sorted(_SINGLE_FNS)})')

    bare = unwrap_model(model)
    bare.eval()
    # crop-window tuning is forwarded only to the vanilla 'single' finder.
    extra = (single_zoom_kwargs or {}) if mode == 'single' else {}

    out_path_dir = None
    if out_dir is not None:
        out_path_dir = Path(out_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)

    records: List[EvalRecord] = []
    n_viz = 0
    warned = False

    for i, item in enumerate(tqdm(items, desc=f'{log_tag} {mode}', unit='item')):
        want_viz = (n_viz < viz_n) and (not item.is_real)
        try:
            out = single_fn(
                bare, item, res, device=device, use_amp=use_amp,
                amp_dtype=amp_dtype, decoder=decoder, return_debug=want_viz,
                **extra,
            )
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')
            continue

        rec, debug = out if want_viz else (out, None)
        records.append(rec)

        if not want_viz:
            continue

        boxes, labels, attn2, attn2_gh, attn_combined = _normalize_debug(mode, debug)
        frames = _frames_from_debug(debug, labels)
        full_score = _sigmoid(debug.get('full_logit'))
        gt_box = None
        if rec.gt_mask is not None and rec.gt_mask.any():
            gt_box = mask_to_bbox(rec.gt_mask.astype(bool))
        title = (f'{item.item_id} src={item.source} mode={mode} '
                 f'f1={rec.f1:.3f} iou={rec.iou:.3f} bucket={rec.bucket} '
                 f'nbox={len(boxes)} nframe={len(frames)}')
        fig = plot_multi_zoom_result(
            debug['img_pil'], debug.get('attn1'), debug['grid_hw'],
            boxes, debug.get('mask_zoom'),
            attn2=attn2, attn2_grid_hw=attn2_gh,
            attn_combined=attn_combined, attn_combined_grid_hw=attn2_gh,
            gt_mask=rec.gt_mask.astype(bool) if rec.gt_mask is not None else None,
            gt_box=gt_box, box_labels=labels, frames=frames, full_score=full_score,
            title=title, decoder_name=decoder,
        )
        out_path = (out_path_dir / f'{i:04d}_{item.source}_{item.item_id}_{mode}.png'
                    if out_path_dir is not None else None)
        shown = _emit_figure(fig, out_path, show)
        if show and not shown and not warned:
            log_line(f'{log_tag} note: --show found no display channel (need an '
                     f'IPython/Colab kernel or a graphics terminal: iTerm2/kitty/'
                     f'chafa). Figures still save to --out_dir. Under "!python" in '
                     f'Colab, call run_zoom_viz / main() from a notebook cell.')
            warned = True
        n_viz += 1

    summarize(records, log_tag=log_tag, tag=mode)
    if out_path_dir is not None:
        log_line(f'{log_tag} wrote {n_viz} figures → {out_path_dir}')
    return records
