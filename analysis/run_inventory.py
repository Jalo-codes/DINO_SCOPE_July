"""analysis.run_inventory — index every training run under a runs root.

Walks a runs directory (local, box, or a mounted Drive folder), reads each run's
``run_config.json`` (train.py writes one per run), and emits a single manifest of
the apples-to-apples-relevant identity of every run — so nothing gets lost to
folder drift and any two runs can be compared on a like-for-like basis.

For each run it records: backbone + resolution + LoRA, head config (contrastive
/ pool / patch), the splice_mix, derived flags (pico-in-train, full-fakes-in-train,
contrastive-active), seed/epochs/aug, and which checkpoints are actually on disk.

Run:
    python -m analysis.run_inventory --runs_root /content/drive/MyDrive/DINO_SCOPE_RUNS
    python -m analysis.run_inventory --runs_root /media/ssd/runs --out runs_index.csv
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional


def _fmt_mix(mix: Optional[dict]) -> str:
    if not mix:
        return '-'
    return ','.join(f'{k}={v:g}' for k, v in sorted(mix.items()))


def _row(run_dir: str, runs_root: str) -> Optional[Dict]:
    cfg_path = os.path.join(run_dir, 'run_config.json')
    try:
        cfg = json.load(open(cfg_path))
    except Exception as exc:  # noqa: BLE001 — inventory should never abort on one bad file
        return {'run': os.path.relpath(run_dir, runs_root), 'error': str(exc)[:60]}

    mix = cfg.get('splice_mix') or {}
    cont_dim = cfg.get('contrastive_dim', 0) or 0
    lam_cont = cfg.get('lambda_contrastive', 0.0) or 0.0

    ckpts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, '*.pt')))
    epoch_ckpts = [c for c in ckpts if c.startswith('epoch_')]

    return {
        'run':        os.path.relpath(run_dir, runs_root),
        'model':      str(cfg.get('model_name', '?')).split('/')[-1],
        'res':        f"{cfg.get('image_size','?')}/{cfg.get('patch_size','?')}",
        'lora':       cfg.get('lora_rank', '?'),
        'cont_dim':   cont_dim,
        'patch_bce':  int(bool(cfg.get('patch_bce', False))),
        'cont_active': int(cont_dim > 0 and lam_cont > 0.0),
        'pico_train': int('pico_pseudo' in mix),
        'ff_train':   int('full_fakes' in mix),
        'splice_mix': _fmt_mix(mix),
        'aug':        cfg.get('aug_severity', '?'),
        'seed':       cfg.get('seed', '?'),
        'epochs':     cfg.get('num_epochs', '?'),
        'ckpts':      f"{'best' if 'best.pt' in ckpts else '-'}"
                      f"+{len(epoch_ckpts)}ep",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs_root', required=True, help='directory to scan for runs')
    ap.add_argument('--out', default=None,
                    help='manifest CSV path (default: <runs_root>/run_inventory.csv)')
    args = ap.parse_args()

    run_dirs = sorted(os.path.dirname(p)
                      for p in glob.glob(os.path.join(args.runs_root, '**', 'run_config.json'),
                                         recursive=True))
    if not run_dirs:
        raise SystemExit(f'no run_config.json found under {args.runs_root}')

    rows = [r for r in (_row(d, args.runs_root) for d in run_dirs) if r]

    cols = ['run', 'model', 'res', 'lora', 'cont_dim', 'patch_bce', 'cont_active',
            'pico_train', 'ff_train', 'splice_mix', 'aug', 'seed', 'epochs', 'ckpts']
    widths = {c: max(len(c), *(len(str(r.get(c, ''))) for r in rows)) for c in cols}
    print('  '.join(f'{c:<{widths[c]}}' for c in cols))
    print('  '.join('-' * widths[c] for c in cols))
    for r in rows:
        if 'error' in r:
            print(f"{r['run']:<{widths['run']}}  ERROR: {r['error']}")
            continue
        print('  '.join(f'{str(r.get(c, "")):<{widths[c]}}' for c in cols))

    out = args.out or os.path.join(args.runs_root, 'run_inventory.csv')
    import csv
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ['error'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n# {len(rows)} runs -> {out}')


if __name__ == '__main__':
    main()
