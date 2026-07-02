"""lab_utils.train.hardware — single hardware resolver (C3).

Consolidates the inline device/AMP/distributed wiring that was scattered across
the legacy god script.  Delegates to the existing amp.py and distributed.py
(kept unchanged).

Targets:
    Dual 2080 Ti (CC 7.5): fp16, DDP/nccl, world=2
    Colab A100/L4/T4:      bf16 (CC≥8) or fp16, world=1
    CPU dev box / CI:      amp off, world=1

Returns a HardwareInfo dataclass that is embedded in RunConfig (C2).
"""

import dataclasses
import os
from typing import Optional


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """Resolved hardware and precision state for one run."""
    device:       str            # 'cuda:0', 'cuda:1', 'cpu', 'mps', ...
    use_amp:      bool
    amp_dtype:    Optional[str]  # 'fp16' | 'bf16' | None
    world_size:   int
    rank:         int
    local_rank:   int
    is_main:      bool
    gpu_name:     Optional[str]  # e.g. 'NVIDIA GeForce RTX 2080 Ti'
    compute_cap:  Optional[str]  # e.g. '7.5'


def resolve_hardware(
    *,
    device: str = 'cuda',
    want_amp: bool = True,
    dist_backend: str = 'nccl',
) -> HardwareInfo:
    """Probe the runtime environment; return resolved HardwareInfo.

    Args:
        device:       Requested device type ('cuda', 'cpu', 'mps').
        want_amp:     Whether to enable mixed precision (disabled on CPU/MPS).
        dist_backend: Distributed process group backend ('nccl' or 'gloo').

    Returns:
        HardwareInfo with device, amp settings, distributed state, and GPU info.
    """
    import torch
    from lab_utils.train.distributed import setup as dist_setup
    from lab_utils.train.amp import resolve_amp

    # ── Distributed context ───────────────────────────────────────────────────
    ctx = dist_setup(backend=dist_backend)

    # ── Device resolution ─────────────────────────────────────────────────────
    if device == 'cuda':
        if torch.cuda.is_available():
            resolved_device = f'cuda:{ctx.local_rank}'
        else:
            resolved_device = 'cpu'
    elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        resolved_device = 'mps'
    else:
        resolved_device = device

    torch_device = torch.device(resolved_device)

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp, amp_dtype_obj = resolve_amp(torch_device, want_amp=want_amp)
    if amp_dtype_obj is None:
        amp_dtype_str = None
    elif amp_dtype_obj == torch.bfloat16:
        amp_dtype_str = 'bf16'
    else:
        amp_dtype_str = 'fp16'

    # ── GPU info ──────────────────────────────────────────────────────────────
    gpu_name:    Optional[str] = None
    compute_cap: Optional[str] = None
    if torch_device.type == 'cuda' and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch_device)
        except Exception:
            pass
        try:
            maj, mn  = torch.cuda.get_device_capability(torch_device)
            compute_cap = f'{maj}.{mn}'
        except Exception:
            pass

    # ── cudnn / matmul precision ──────────────────────────────────────────────
    if torch_device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    return HardwareInfo(
        device=resolved_device,
        use_amp=use_amp,
        amp_dtype=amp_dtype_str,
        world_size=ctx.world_size,
        rank=ctx.rank,
        local_rank=ctx.local_rank,
        is_main=ctx.is_main,
        gpu_name=gpu_name,
        compute_cap=compute_cap,
    )
