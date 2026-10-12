"""SFT and DPO objectives.

SFT reduction (documented in the report): **mean negative log-probability over
all unmasked response tokens in the batch** (token-level mean). In the training
loop the denominator is the response-token count of the whole accumulation
group, so gradient accumulation reproduces the large-batch token mean exactly.

DPO (Rafailov et al., 2023), per example:
    L = -log sigmoid(beta * [(log pi(y_w|x) - log pi(y_l|x)) - (log ref(y_w|x) - log ref(y_l|x))])
Implicit rewards r = beta * (log pi - log ref); reward accuracy uses r_w > r_l
strictly, so identical policy/reference gives accuracy 0, not 0.5.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from alignment.batching import build_dpo_batch, build_sft_batch, concat_chosen_rejected, pad_token_id
from alignment.logprobs import sequence_logprobs


# ----------------------------------------------------------------------------- SFT
def sft_loss_from_batch(model, batch, token_normalizer=None, fast: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, n_response_tokens)``; loss = -sum(response logp) / normalizer."""
    logps, counts = sequence_logprobs(model, batch, fast=fast)
    n_tokens = counts.sum()
    denom = token_normalizer if token_normalizer is not None else n_tokens.clamp_min(1)
    return -logps.sum() / denom, n_tokens


def compute_sft_loss(model, tokenizer, prompts: Sequence[str], responses: Sequence[str], max_length: int) -> torch.Tensor:
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    loss, _ = sft_loss_from_batch(model, batch)
    return loss


# ----------------------------------------------------------------------------- DPO
def dpo_losses(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Per-example DPO losses plus implicit rewards (rewards are detached)."""
    reference_chosen_logps = reference_chosen_logps.to(policy_chosen_logps.device).detach()
    reference_rejected_logps = reference_rejected_logps.to(policy_chosen_logps.device).detach()
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    losses = -F.logsigmoid(beta * (policy_gap - reference_gap))
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    losses, _, _ = dpo_losses(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )
    return losses.mean()


def reward_stats(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "reward_accuracy": (chosen_rewards > rejected_rewards).float(),
        "reward_margin": chosen_rewards - rejected_rewards,
        "chosen_reward": chosen_rewards,
        "rejected_reward": rejected_rewards,
    }


def split_chosen_rejected(logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    half = logps.shape[0] // 2
    return logps[:half], logps[half:]


def concat_logps(model, concat_batch, fast: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Chosen and rejected summed log-probs from one forward over the stacked batch."""
    logps, _ = sequence_logprobs(model, concat_batch, fast=fast)
    return split_chosen_rejected(logps)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
) -> torch.Tensor:
    dpo_batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    stacked = concat_chosen_rejected(dpo_batch, pad_token_id(tokenizer))
    policy_chosen, policy_rejected = concat_logps(policy_model, stacked)
    with torch.no_grad():
        reference_chosen, reference_rejected = concat_logps(reference_model, stacked)
    return dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
