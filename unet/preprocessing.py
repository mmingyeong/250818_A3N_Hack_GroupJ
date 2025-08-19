# preprocessing.py
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

# hard dependency: torch (당신 코드가 이미 torch 기반)
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

# ---------------------------
# element-wise transform
# ---------------------------

class Log10Transform:
    """Apply log10(clamp_min(x, 0) + eps) to torch tensor with finite guard."""
    def __init__(self, eps: float = 1e-6, post_clip: float | None = None, enabled: bool = True):
        """
        eps: typical noise floor (choose >= min positive you expect)
        post_clip: if given, clamp output to [-post_clip, +post_clip] after log
        enabled: if False, bypass transformation (identity)
        """
        self.eps = float(eps)
        self.post_clip = post_clip
        self.enabled = enabled

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x  # 그냥 통과
        # 1) pre-log guard: forbid negative inputs to log
        x = torch.clamp_min(x, 0.0) + self.eps
        # 2) log
        x = torch.log10(x)
        # 3) finite guard
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # 4) optional post-log clipping to avoid heavy tails
        if self.post_clip is not None:
            x = torch.clamp(x, -self.post_clip, +self.post_clip)
        return x

# ---------------------------
# 2D random aug (per-sample)
# ---------------------------

class RandomFlipRotate2D:
    """
    Per-sample random 2D augmentation on (C,H,W) or (H,W).
      - Rotate by k*90°, k∈{0,1,2,3}
      - Horizontal/Vertical flips with prob p_flip each
    """
    def __init__(self, p_flip: float = 0.5, generator: torch.Generator | None = None):
        self.p_flip = float(p_flip)
        self.gen = generator  # for reproducibility if desired

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        # ensure shape (C,H,W)
        added_c = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            added_c = True
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

        # choose params
        k = int(torch.randint(0, 4, (), generator=self.gen))
        do_h = bool(torch.rand((), generator=self.gen) < self.p_flip)
        do_v = bool(torch.rand((), generator=self.gen) < self.p_flip)

        # rotate by k*90 along H,W (last two dims)
        if k > 0:
            x = torch.rot90(x, k=k, dims=(-2, -1))

        # flips
        if do_h:
            x = torch.flip(x, dims=(-1,))   # horizontal (W)
        if do_v:
            x = torch.flip(x, dims=(-2,))   # vertical (H)

        return x[0] if added_c else x


# ---------------------------
# batched collate helpers
# ---------------------------

def _apply_per_sample(x: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
    """
    Apply fn to each sample in a batch tensor.
    x: (B,C,H,W) or (B,H,W)
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected 3D/4D batch tensor, got {tuple(x.shape)}")
    out = []
    for i in range(x.shape[0]):
        out.append(fn(x[i]))
    return torch.stack(out, dim=0)


# ---------------------------
# batched collate helpers
# ---------------------------

def make_train_collate(eps: float = 1e-6, p_flip: float = 0.5,
                       seed: int | None = None, use_log: bool = True) -> Callable:
    """
    Build a collate_fn for TRAIN loader:
      - x: optional log10(+eps) then random 90° rotations + flips
      - y: unchanged
    """
    log10 = Log10Transform(eps=eps, enabled=use_log)
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    aug = RandomFlipRotate2D(p_flip=p_flip, generator=gen)

    def collate(batch) -> Tuple[Tensor, Tensor]:
        x, y = default_collate(batch)
        # transforms
        x = log10(x)
        x = _apply_per_sample(x, aug)
        return x, y

    return collate


def make_eval_collate(eps: float = 1e-6, use_log: bool = True) -> Callable:
    """
    Build a collate_fn for VAL/TEST/PRED:
      - x: optional log10(+eps)
      - y: unchanged
    """
    log10 = Log10Transform(eps=eps, enabled=use_log)

    def collate(batch) -> Tuple[Tensor, Tensor]:
        x, y = default_collate(batch)
        x = log10(x)
        return x, y

    return collate