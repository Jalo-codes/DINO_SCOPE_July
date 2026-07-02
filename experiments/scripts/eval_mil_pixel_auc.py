"""experiments.scripts.eval_mil_pixel_auc — pixel-level AUC from MIL attention.

Evaluates one or two checkpoints on a fixed mixed test set
(IMD2020 + TGIF FR + TGIF SP) using the raw gated-attention weights from
the AttentionPool head as a continuous per-pixel localization score.
No decoder — the attention IS the prediction.

Pixel AUC is computed per splice item at model input resolution (res.image_size):
attention is bilinearly upsampled from the patch grid (e.g. 32×32 for res=448)
to pixel space, then scored against the binarised GT mask.  The per-item AUC
values are averaged over all splice items in each source group.  Real items
(no GT mask) are excluded from the pixel AUC but DO contribute to the
image-level AUC.

Ada usage — source env_ada.sh first, then:

Step 1 — eval existing baseline (r16 + contrastive, epoch 4):

    BASELINE=/home/studentresearch2/runs/ablation/lora_rank_sweep/r016/epoch_0004.pt

    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.scripts.eval_mil_pixel_auc \\
        --checkpoint  $BASELINE \\
        --label       "r16+contrastive e4" \\
        --imd2020_root    $DATA_ROOT/IMD2020 \\
        --tgif2_root      "$DATA_ROOT/content/flux_originals" \\
        --imd_n 5000 --tgif_fr_n 2500 --tgif_sp_n 2500 \\
        --amp_dtype bfloat16 \\
        --out_json $RUNS_ROOT/mil_only_r16_448/baseline_pixel_auc.json

Step 2 — train MIL-only (no contrastive, same data recipe, epoch 4):

    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.scripts.train \\
        --checkpoint_root $RUNS_ROOT/mil_only_r16_448 \\
        --image_size 448 \\
        --lora_rank 16 --lora_alpha 32 \\
        --contrastive_dim 0 \\
        --pool_hidden 256 \\
        --lambda_image_bce 1.0 \\
        --paste_frac 0.5 \\
        --noise_prob 0.8 --jpeg_prob 0.55 \\
        --train_samples 3000 \\
        --batch_size 8 --grad_accum 1 \\
        --num_epochs 4 --min_epochs 4 \\
        --warmup_epochs 1.0 \\
        --early_stop_patience 2 --early_stop_reduce mean \\
        --num_workers 8 \\
        --val_zoom --imd_val_only --casia_train \\
        --tgif_val_models flux1dev --val_per_cell 100 \\
        --imd2020_root      $DATA_ROOT/IMD2020 \\
        --casia_root        $DATA_ROOT/casia \\
        --sagid_root        $DATA_ROOT/SAGI_D \\
        --coco_inpaint_root "$DATA_ROOT/INPAINT_COCO/content/inpaint_coco/images" \\
        --tgif2_root        "$DATA_ROOT/content/flux_originals"

Step 3 — compare baseline vs MIL-only on the same fixed test set:

    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.scripts.eval_mil_pixel_auc \\
        --checkpoint   $BASELINE \\
        --checkpoint_b $RUNS_ROOT/mil_only_r16_448/epoch_0004.pt \\
        --label "r16+contrastive e4" "r16 MIL-only e4" \\
        --imd2020_root    $DATA_ROOT/IMD2020 \\
        --tgif2_root      "$DATA_ROOT/content/flux_originals" \\
        --imd_n 5000 --tgif_fr_n 2500 --tgif_sp_n 2500 \\
        --amp_dtype bfloat16 \\
        --out_json $RUNS_ROOT/mil_only_r16_448/pixel_auc_compare.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as TF

from lab_utils.compat import trapz
from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.fetch import ModelInfo, model_info as run_model
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_score(image_logit: Optional[float]) -> float:
    if image_logit is None or not math.isfinite(image_logit):
        return float('nan')
    return float(1.0 / (1.0 + math.exp(-image_logit)))


def _trapz_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """ROC AUC via trapezoid rule.  Returns None when degenerate (no pos or neg)."""
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores)
    sl  = labels[order]
    tpr = np.cumsum(sl) / n_pos
    fpr = np.cumsum(1 - sl) / n_neg
    auc = float(trapz(tpr, fpr))
    return 1.0 + auc if auc < 0 else auc


def _pixel_auc(info: ModelInfo, item: Item, res: Resolution) -> Optional[float]:
    """Per-image pixel-level AUC from bilinearly upsampled attention vs GT mask.

    Returns None for real items (no GT) or when the model has no pool head.
    The attention grid is upsampled to (res.image_size, res.image_size) in
    float32 (no quantization).  The GT mask is binarised at 0.5 and resized
    with NEAREST to the same resolution so each patch maps to its pixel block.
    """
    if item.is_real or item.mask is None:
        return None
    if info.attention is None:
        return None

    n_side = info.grid_hw[0]
    S = res.image_size

    # Bilinear upsample: (n_side, n_side) → (S, S) in float32, no quantization
    attn_t = (
        torch.from_numpy(info.attention.reshape(n_side, n_side))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, n_side, n_side)
    attn_up = TF.interpolate(attn_t, size=(S, S), mode='bilinear', align_corners=False)
    attn_px = attn_up.squeeze().numpy().flatten()  # (S*S,)

    # GT at model resolution (NEAREST so each patch = 1 block of pixels)
    from PIL import Image
    gt_pil = Image.open(item.mask).convert('L')
    if gt_pil.size != (S, S):
        gt_pil = gt_pil.resize((S, S), Image.NEAREST)
    gt_px = (np.asarray(gt_pil, dtype=np.float32) / 255.0 >= 0.5).astype(np.int32).flatten()

    return _trapz_auc(attn_px, gt_px)


# ── Test-set builder ──────────────────────────────────────────────────────────

def _collect_imd(root: Path, res: Resolution, n: int) -> List[Item]:
    """All IMD2020 val items (real + splice), capped at n."""
    _, val_ds = REGISTRY['imd2020'](root, res=res, val_split=1.0)
    items = val_ds.items
    log_line(f'[data] imd2020: {len(items)} total → capping at {n}')
    return items[:n]


def _collect_tgif(
    root: Path,
    res: Resolution,
    fr_n: int,
    sp_n: int,
) -> Tuple[List[Item], List[Item]]:
    """TGIF2 val items split into FR and SP, each capped.

    We load all items (no eval_per_cell quota) then filter by tgif_type so the
    two caps are independent and easy to adjust.
    """
    _, val_ds = REGISTRY['tgif2'](root, res=res)
    all_items = val_ds.items

    fr_items = [it for it in all_items if it.meta.get('tgif_type') == 'fr']
    sp_items = [it for it in all_items if it.meta.get('tgif_type') == 'sp']
    real_items = [it for it in all_items if it.is_real]

    log_line(
        f'[data] tgif2: {len(fr_items)} FR, {len(sp_items)} SP, '
        f'{len(real_items)} reals → capping FR={fr_n}, SP={sp_n}'
    )
    # Reals are included in both FR and SP groups so img_AUC has negatives.
    # They contribute label=0 to image-level AUC only (no GT mask → skipped
    # by _pixel_auc), so pixel_AUC is unaffected.
    return fr_items[:fr_n] + real_items, sp_items[:sp_n] + real_items


# ── Per-source eval ───────────────────────────────────────────────────────────

def _eval_source(
    model,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    source_label: str,
    max_items: Optional[int] = None,
) -> Dict:
    """Forward pass over all items, return pixel/image AUC stats for the source."""
    if max_items:
        items = items[:max_items]

    pixel_aucs: List[float] = []
    img_scores:  List[float] = []
    img_labels:  List[int]   = []

    disable_tqdm = not sys.stdout.isatty()
    try:
        from tqdm import tqdm
        iterator = tqdm(items, desc=f'[eval] {source_label}', unit='item', disable=disable_tqdm)
    except ImportError:
        iterator = items
        if disable_tqdm:
            log_line(f'[eval] {source_label}: processing {len(items)} items…')

    for item in iterator:
        try:
            img_t = load_image_tensor(item, res, device=device)
            info  = run_model(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
        except Exception as exc:
            log_line(f'[eval] WARN: skipped {item.item_id}: {exc}')
            continue

        # image-level
        s = _image_score(info.image_logit)
        if math.isfinite(s):
            img_scores.append(s)
            img_labels.append(0 if item.is_real else 1)

        # pixel-level (splice only)
        pauc = _pixel_auc(info, item, res)
        if pauc is not None:
            pixel_aucs.append(pauc)

    img_auc = None
    if img_scores:
        img_auc = _trapz_auc(
            np.array(img_scores, dtype=np.float64),
            np.array(img_labels,  dtype=np.int32),
        )

    return {
        'source':     source_label,
        'n_total':    len(items),
        'n_splice':   len(pixel_aucs),
        'pixel_auc':  float(np.mean(pixel_aucs))  if pixel_aucs else None,
        'pixel_std':  float(np.std(pixel_aucs))   if pixel_aucs else None,
        'img_auc':    img_auc,
        'pixel_aucs': pixel_aucs,
        'img_scores': img_scores,
        'img_labels': img_labels,
    }


def _print_single(results: List[Dict], label: str) -> None:
    log_line(f'[eval] ══════ {label} ══════')
    hdr  = f"{'Source':<16}{'N_splice':>10}{'pixel_AUC':>12}{'±std':>8}{'img_AUC':>10}"
    print(hdr)
    print('─' * len(hdr))

    all_pixel: List[float] = []
    all_img_s: List[float] = []
    all_img_l: List[int]   = []

    for r in results:
        pa  = f"{r['pixel_auc']:.4f}" if r['pixel_auc'] is not None else '  N/A  '
        std = f"{r['pixel_std']:.4f}" if r['pixel_std'] is not None else '  N/A  '
        ia  = f"{r['img_auc']:.4f}"   if r['img_auc']   is not None else '  N/A  '
        print(f"{r['source']:<16}{r['n_splice']:>10}{pa:>12}{std:>8}{ia:>10}")
        all_pixel.extend(r['pixel_aucs'])
        all_img_s.extend(r['img_scores'])
        all_img_l.extend(r['img_labels'])

    print('─' * len(hdr))
    overall_pixel_auc = f"{np.mean(all_pixel):.4f}" if all_pixel else 'N/A'
    overall_std       = f"{np.std(all_pixel):.4f}"  if all_pixel else 'N/A'
    overall_img_auc   = _trapz_auc(
        np.array(all_img_s, dtype=np.float64),
        np.array(all_img_l, dtype=np.int32),
    )
    ia_str = f'{overall_img_auc:.4f}' if overall_img_auc is not None else 'N/A'
    print(
        f"{'OVERALL':<16}{len(all_pixel):>10}{overall_pixel_auc:>12}"
        f"{overall_std:>8}{ia_str:>10}"
    )


def _print_compare(
    results_a: List[Dict],
    results_b: List[Dict],
    label_a: str,
    label_b: str,
) -> None:
    log_line(f'[eval] ══════ {label_a}  vs  {label_b} ══════')
    hdr = (
        f"{'Source':<16}"
        f"{'A pixel':>10}{'B pixel':>10}{'Δpixel':>9}"
        f"{'A img':>9}{'B img':>9}{'Δimg':>7}"
    )
    print(hdr)
    print('─' * len(hdr))

    all_pa, all_pb = [], []
    all_ia_s, all_ia_l = [], []
    all_ib_s, all_ib_l = [], []

    for ra, rb in zip(results_a, results_b):
        assert ra['source'] == rb['source']
        pa = ra['pixel_auc']
        pb = rb['pixel_auc']
        ia = ra['img_auc']
        ib = rb['img_auc']
        delta_p = f"{pb - pa:+.4f}" if (pa is not None and pb is not None) else '  N/A  '
        delta_i = f"{ib - ia:+.4f}" if (ia is not None and ib is not None) else '  N/A  '
        pa_s = 'N/A' if pa is None else f'{pa:.4f}'
        pb_s = 'N/A' if pb is None else f'{pb:.4f}'
        ia_s = 'N/A' if ia is None else f'{ia:.4f}'
        ib_s = 'N/A' if ib is None else f'{ib:.4f}'
        print(
            f"{ra['source']:<16}"
            f"{pa_s:>10}{pb_s:>10}{delta_p:>9}"
            f"{ia_s:>9}{ib_s:>9}{delta_i:>7}"
        )
        if ra['pixel_aucs']:
            all_pa.extend(ra['pixel_aucs'])
        if rb['pixel_aucs']:
            all_pb.extend(rb['pixel_aucs'])
        all_ia_s.extend(ra['img_scores']); all_ia_l.extend(ra['img_labels'])
        all_ib_s.extend(rb['img_scores']); all_ib_l.extend(rb['img_labels'])

    print('─' * len(hdr))
    overall_pa  = f"{np.mean(all_pa):.4f}" if all_pa else 'N/A'
    overall_pb  = f"{np.mean(all_pb):.4f}" if all_pb else 'N/A'
    delta_total = (
        f"{np.mean(all_pb) - np.mean(all_pa):+.4f}"
        if (all_pa and all_pb) else 'N/A'
    )
    oia = _trapz_auc(np.array(all_ia_s, np.float64), np.array(all_ia_l, np.int32))
    oib = _trapz_auc(np.array(all_ib_s, np.float64), np.array(all_ib_l, np.int32))
    di  = f"{oib - oia:+.4f}" if (oia and oib) else 'N/A'
    oia_s = 'N/A' if oia is None else f'{oia:.4f}'
    oib_s = 'N/A' if oib is None else f'{oib:.4f}'
    print(
        f"{'OVERALL':<16}"
        f"{overall_pa:>10}{overall_pb:>10}{delta_total:>9}"
        f"{oia_s:>9}{oib_s:>9}{di:>7}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_mil_pixel_auc',
        description='Per-pixel AUC from MIL attention (IMD + TGIF FR/SP).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True,
                   help='Primary checkpoint (.pt).')
    p.add_argument('--checkpoint_b', default=None,
                   help='Second checkpoint for side-by-side comparison (optional).')
    p.add_argument('--label', nargs='*', default=None,
                   help='Display labels: one label per checkpoint (e.g. "baseline" "mil-only").')

    g = p.add_argument_group('datasets')
    g.add_argument('--imd2020_root', default=None, help='IMD2020 dataset root.')
    g.add_argument('--tgif2_root',   default=None, help='TGIF2 dataset root.')
    g.add_argument('--imd_n',      type=int, default=5000,
                   help='Max IMD2020 items (real + splice).')
    g.add_argument('--tgif_fr_n',  type=int, default=2500,
                   help='Max TGIF FR splice items.')
    g.add_argument('--tgif_sp_n',  type=int, default=2500,
                   help='Max TGIF SP splice items.')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp',    action='store_true')
    g.add_argument('--amp_dtype', default='bfloat16', choices=['float16', 'bfloat16'])

    g = p.add_argument_group('output')
    g.add_argument('--out_json', default=None,
                   help='Optional path to save full results as JSON.')
    g.add_argument('--max_items', type=int, default=None,
                   help='Cap per source (smoke-test mode).')
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _build_parser().parse_args()
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    use_amp  = (not args.no_amp) and (device.type == 'cuda')
    amp_dtype = args.amp_dtype

    if not args.imd2020_root and not args.tgif2_root:
        raise SystemExit('eval_mil_pixel_auc: at least one of --imd2020_root or --tgif2_root required.')

    # ── Load checkpoint(s) ────────────────────────────────────────────────────
    log_line(f'[eval] loading checkpoint A: {args.checkpoint}')
    model_a, _, res = load_eval_model(args.checkpoint, device=device, strict=False)
    model_a = unwrap_model(model_a)
    model_a.eval()

    model_b = None
    if args.checkpoint_b:
        log_line(f'[eval] loading checkpoint B: {args.checkpoint_b}')
        model_b, _, res_b = load_eval_model(args.checkpoint_b, device=device, strict=False)
        model_b = unwrap_model(model_b)
        model_b.eval()
        if res_b.image_size != res.image_size or res_b.patch_size != res.patch_size:
            raise SystemExit(
                f'eval_mil_pixel_auc: checkpoints have different resolutions '
                f'({res.image_size}/{res.patch_size} vs {res_b.image_size}/{res_b.patch_size}). '
                'Evaluat them separately.'
            )

    # Warn if model has no pool head (attention will be None → pixel AUC is always None)
    if getattr(model_a, 'pool', None) is None:
        log_line('[eval] WARN: checkpoint A has no AttentionPool — pixel AUC will be N/A')
    if model_b is not None and getattr(model_b, 'pool', None) is None:
        log_line('[eval] WARN: checkpoint B has no AttentionPool — pixel AUC will be N/A')

    # ── Build test set ────────────────────────────────────────────────────────
    sources: List[Tuple[str, List[Item]]] = []

    if args.imd2020_root:
        imd_items = _collect_imd(Path(args.imd2020_root), res, args.imd_n)
        sources.append(('IMD2020', imd_items))

    if args.tgif2_root:
        fr_items, sp_items = _collect_tgif(
            Path(args.tgif2_root), res, args.tgif_fr_n, args.tgif_sp_n
        )
        if fr_items:
            sources.append(('TGIF_FR', fr_items))
        if sp_items:
            sources.append(('TGIF_SP', sp_items))

    log_line(
        f'[eval] test set: '
        + ', '.join(f'{lbl}={len(items)}' for lbl, items in sources)
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = args.label or []
    label_a = labels[0] if len(labels) > 0 else Path(args.checkpoint).stem
    label_b = labels[1] if len(labels) > 1 else (
        Path(args.checkpoint_b).stem if args.checkpoint_b else None
    )

    # ── Eval loop ─────────────────────────────────────────────────────────────
    results_a: List[Dict] = []
    results_b: List[Dict] = []

    for src_label, items in sources:
        log_line(f'[eval] ── {src_label} ({len(items)} items) ──')

        ra = _eval_source(
            model_a, items, res,
            device=device, use_amp=use_amp, amp_dtype=amp_dtype,
            source_label=src_label, max_items=args.max_items,
        )
        results_a.append(ra)

        if model_b is not None:
            rb = _eval_source(
                model_b, items, res,
                device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                source_label=src_label, max_items=args.max_items,
            )
            results_b.append(rb)

    # ── Print results ─────────────────────────────────────────────────────────
    if model_b is not None:
        _print_compare(results_a, results_b, label_a, label_b)
        _print_single(results_a, label_a)
        _print_single(results_b, label_b)
    else:
        _print_single(results_a, label_a)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.out_json:
        out = {
            'label_a': label_a,
            'checkpoint_a': args.checkpoint,
        }
        def _serialise(res_list: List[Dict]) -> List[Dict]:
            return [
                {k: v for k, v in r.items() if k not in ('pixel_aucs', 'img_scores', 'img_labels')}
                for r in res_list
            ]
        out['results_a'] = _serialise(results_a)
        if model_b is not None:
            out['label_b'] = label_b
            out['checkpoint_b'] = args.checkpoint_b
            out['results_b'] = _serialise(results_b)
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        log_line(f'[eval] results saved to {args.out_json}')


if __name__ == '__main__':
    main()
