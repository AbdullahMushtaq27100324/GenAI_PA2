"""Tokenization, response-only label masking and padding for SFT and DPO.

Conventions (shared by training, evaluation and ``submission_adapter.py``):

* Prompt and response are tokenized *separately* and concatenated, so the
  prompt/response boundary is exact and independent of BPE merges.
* EOS is appended to the response *before* truncation; a truncated response may
  therefore lose its EOS.
* Optional ``max_prompt_length`` keeps the *last* prompt tokens (closest to the
  response, i.e. the assistant header survives) before total truncation.
* ``labels`` are aligned with ``input_ids`` (no shift); prompt and padding
  positions are ``-100``. The causal shift happens in ``alignment.logprobs``.
* Right padding. Masks come from sequence lengths, never from ``id == pad_id``,
  because for Qwen2.5 the pad token *is* the EOS token.
* Row order is always preserved.
"""

from __future__ import annotations

from typing import Sequence

import torch

from alignment import IGNORE_INDEX


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(i) for i in ids]


def pad_token_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad_id is None else int(pad_id)


def encode_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return unpadded ``(input_ids, labels)`` for one prompt/response pair."""
    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :] if max_prompt_length > 0 else []
    input_ids = (prompt_ids + response_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]
    return input_ids, labels


def pad_examples(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_id: int,
    pad_to_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad a list of ``(input_ids, labels)`` into a batch of tensors."""
    longest = max((len(ids) for ids, _ in examples), default=0)
    width = max(longest, pad_to_length or 0, 1)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, (ids, labs) in enumerate(examples):
        n = len(ids)
        if n == 0:
            continue
        input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(labs, dtype=torch.long)
        attention_mask[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_sft_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
    padding: str = "max_length",
) -> dict[str, torch.Tensor]:
    """Batch with ``input_ids``, ``attention_mask`` and response-only ``labels``.

    ``padding="max_length"`` pads every row to ``max_length`` (adapter contract);
    ``padding="longest"`` pads to the longest row (used in training for speed;
    gives identical log-probs because padding is on the right).
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    examples = [
        encode_example(tokenizer, p, r, max_length, max_prompt_length) for p, r in zip(prompts, responses)
    ]
    pad_to = max_length if padding == "max_length" else None
    return pad_examples(examples, pad_token_id(tokenizer), pad_to)


def build_dpo_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
    padding: str = "max_length",
) -> dict[str, dict[str, torch.Tensor]]:
    """Separate chosen/rejected LM batches built with the same prompt formatting."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must have the same length")
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length, padding),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length, padding),
    }


def concat_chosen_rejected(dpo_batch: dict[str, dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Stack chosen rows then rejected rows into one batch (one forward pass)."""
    chosen, rejected = dpo_batch["chosen"], dpo_batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])

    def _widen(t: torch.Tensor, fill: int) -> torch.Tensor:
        if t.shape[1] == width:
            return t
        pad = torch.full((t.shape[0], width - t.shape[1]), fill, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    return {
        "input_ids": torch.cat([_widen(chosen["input_ids"], pad_id), _widen(rejected["input_ids"], pad_id)]),
        "attention_mask": torch.cat([_widen(chosen["attention_mask"], 0), _widen(rejected["attention_mask"], 0)]),
        "labels": torch.cat([_widen(chosen["labels"], IGNORE_INDEX), _widen(rejected["labels"], IGNORE_INDEX)]),
    }


def trim_padding(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop all-padding columns on the right (pure speed optimization)."""
    used = int(batch["attention_mask"].sum(dim=1).max().item()) if batch["attention_mask"].numel() else 0
    used = max(used, 1)
    return {k: v[:, :used] for k, v in batch.items()}
