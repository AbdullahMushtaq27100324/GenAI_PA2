"""Response-only sequence log-probabilities for causal LMs.

For a batch from ``alignment.batching`` (labels aligned with input_ids, -100 on
prompt and padding), the score of row ``b`` is

    sum_{t : labels[b, t+1] != -100} log p(input_ids[b, t+1] | input_ids[b, :t+1])

i.e. logits at position t predict the token at t+1. Token log-probs are always
computed in fp32 (``logsumexp`` via ``cross_entropy``), even under fp16 autocast.

Two numerically equivalent paths:
* ``generic``: model(...).logits on every position (works for any causal LM,
  including the toy test model).
* ``hidden``: for known HF decoder architectures, run the backbone, select only
  response positions and apply the LM head on those rows in checkpointed chunks.
  This avoids materializing the full [B, T, 151936] logits tensor and is what
  makes full fine-tuning fit in 8 GB of VRAM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from alignment import IGNORE_INDEX

_HIDDEN_PATH_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral"}


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _shifted_mask(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_labels = labels[:, 1:]
    return shift_labels, shift_labels.ne(IGNORE_INDEX)


def _rows_sum(token_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = mask.nonzero(as_tuple=True)[0]
    out = torch.zeros(mask.shape[0], dtype=token_logps.dtype, device=token_logps.device)
    return out.index_add(0, rows, token_logps)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed response log-probs ``[B]`` and response token counts ``[B]`` from full logits."""
    shift_labels, mask = _shifted_mask(labels)
    if not bool(mask.any()):
        zero = logits.float().sum() * 0.0
        return zero.expand(labels.shape[0]).clone(), mask.sum(dim=1)
    selected = logits[:, :-1, :][mask].float()
    token_logps = -F.cross_entropy(selected, shift_labels[mask], reduction="none")
    return _rows_sum(token_logps, mask), mask.sum(dim=1)


def _chunk_token_logps(hidden: torch.Tensor, weight: torch.Tensor, bias, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight, bias).float()
    return -F.cross_entropy(logits, targets, reduction="none")


def _supports_hidden_path(model) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) not in _HIDDEN_PATH_MODEL_TYPES:
        return False
    get_head = getattr(model, "get_output_embeddings", None)
    base = getattr(model, "base_model", None)
    return callable(get_head) and isinstance(get_head(), nn.Linear) and base is not None and base is not model


def sequence_logprobs(
    model,
    batch: dict[str, torch.Tensor],
    fast: bool = True,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(summed response log-probs [B], response token counts [B])``.

    Gradients flow to ``model`` unless called under ``torch.no_grad()``.
    """
    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if not (fast and _supports_hidden_path(model)):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logprobs_from_logits(logits, labels)

    outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = outputs[0]
    shift_labels, mask = _shifted_mask(labels)
    counts = mask.sum(dim=1)
    if not bool(mask.any()):
        zero = hidden.float().sum() * 0.0
        return zero.expand(labels.shape[0]).clone(), counts

    selected_hidden = hidden[:, :-1, :][mask]
    targets = shift_labels[mask]
    head = model.get_output_embeddings()
    pieces = []
    for start in range(0, selected_hidden.shape[0], chunk_size):
        h = selected_hidden[start : start + chunk_size]
        t = targets[start : start + chunk_size]
        if torch.is_grad_enabled():
            pieces.append(checkpoint(_chunk_token_logps, h, head.weight, head.bias, t, use_reentrant=False))
        else:
            pieces.append(_chunk_token_logps(h, head.weight, head.bias, t))
    token_logps = torch.cat(pieces)
    return _rows_sum(token_logps, mask), counts
