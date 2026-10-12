"""Prompt template shared by SFT, DPO and evaluation (Qwen ChatML).

The same ``format_prompt`` is used for every Qwen model (base, SFT, both DPO
runs) so that training and generation see exactly the same string. The response
is appended raw and terminated with the tokenizer's EOS token by the batch
builders in ``alignment.batching``.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Generation stops on either terminator: the tokenizer EOS that SFT appends
# (<|endoftext|> for Qwen2.5 base) or the ChatML end-of-turn marker.
STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")


def format_prompt(instruction: str, system: str = SYSTEM_PROMPT) -> str:
    return CHATML_TEMPLATE.format(system=system, instruction=instruction.strip())


def clean_response(text: str) -> str:
    return text.strip()


def stop_token_ids(tokenizer) -> list[int]:
    ids = []
    for token in STOP_STRINGS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            ids.append(token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids
