"""Small-model alignment: from-scratch SFT and DPO on Qwen2.5-0.5B, plus TRL LoRA-DPO.

Core math lives in:
    alignment.batching  - prompt/response tokenization, -100 masking, padding
    alignment.logprobs  - response-only summed log-probabilities
    alignment.losses    - SFT loss and DPO loss
The root-level ``submission_adapter.py`` is a thin wrapper around these modules.
"""

IGNORE_INDEX = -100
