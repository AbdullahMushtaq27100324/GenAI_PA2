"""Pieces shared by the from-scratch SFT and DPO loops (no HF Trainer)."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
from typing import Any

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def config_from_dict(cls, values: dict[str, Any]):
    """Build a dataclass config, rejecting unknown keys (catches typos in YAML)."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def autocast(precision: str):
    if precision == "fp32" or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=DTYPES[precision])


def make_scaler(precision: str):
    # Loss scaling is only needed for fp16; trainable weights stay fp32 in every mode.
    return torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))


def make_optimizer(model, name: str, lr: float, weight_decay: float, betas=(0.9, 0.999), eps: float = 1e-8):
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, alpha=0.99, eps=eps)
    raise ValueError(f"unknown optimizer {name}")


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, kind: str):
    """``cosine``: linear warmup then cosine decay to 0. ``constant``: warmup then flat."""

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def micro_batch_order(
    lengths: list[int], micro_batch_size: int, seed: int, epoch: int, group_by_length: bool, megabatch_mult: int = 50
) -> list[list[int]]:
    """Deterministic micro-batches of example indices for one epoch.

    With ``group_by_length`` the shuffled order is cut into megabatches of
    ``micro_batch_size * megabatch_mult`` examples, each sorted by length, which
    cuts padding while keeping the order random at the megabatch level. The same
    (seed, epoch) always gives the same order, so resume can skip exactly.
    """
    rng = random.Random(seed + 1000 * epoch)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    if group_by_length:
        size = micro_batch_size * megabatch_mult
        megabatches = [order[i : i + size] for i in range(0, len(order), size)]
        order = [i for mb in megabatches for i in sorted(mb, key=lambda j: -lengths[j])]
    return [order[i : i + micro_batch_size] for i in range(0, len(order), micro_batch_size)]


class SimulatedCrash(RuntimeError):
    """Raised by ``stop_after_steps`` to test kill-and-resume in the dry run."""
