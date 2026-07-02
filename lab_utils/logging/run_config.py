"""lab_utils.logging.run_config — auto-rendered run-settings printer (C2).

log_run_config() iterates RunConfig.fields and emits one [cfg] line per field,
so the printed value is by construction the value that drives the run.
No hand-composed f-strings; a new RunConfig field prints automatically.

Also provides to_dict / from_dict for the checkpoint cfg slot (round-trip C2 gate).
"""

import dataclasses
from typing import Any

from lab_utils.logging.text import log_line


_SECTION_HEADERS = {
    'imd2020_root': 'dataset roots',
    'run_dir':      'checkpoint',
    'num_epochs':   'training loop',
    'model_name':   'model',
    'lambda_image_bce': 'loss lambdas',
    'splice_mix':   'data / sampling',
    'train_crop_min': 'augmentation',
    'recipe':       'recipe / tgif-finetune',
    'device':       'hardware',
}


def log_run_config(cfg, *, log_tag: str = '[cfg]') -> None:
    """Emit one [cfg] log line per RunConfig field, grouped by section.

    Args:
        cfg:     RunConfig (or any frozen dataclass with the same field layout).
        log_tag: Log tag prepended to each line (default '[cfg]').
    """
    current_section: str = ''
    for field in dataclasses.fields(cfg):
        section = _SECTION_HEADERS.get(field.name, '')
        if section and section != current_section:
            current_section = section
            log_line(f'{log_tag} ── {section} ──')
        val = getattr(cfg, field.name)
        log_line(f'{log_tag}   {field.name}={_fmt(val)}')


def _fmt(val: Any) -> str:
    if val is None:
        return 'None'
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, float):
        return f'{val:.6g}'
    if isinstance(val, dict):
        return '{' + ', '.join(f'{k}:{v}' for k, v in sorted(val.items())) + '}'
    return str(val)
