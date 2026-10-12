"""Batched, resumable generation on fixed evaluation sets.

Output JSON (one file per model x eval set) is a list in eval-set order with
entries ``{"id", "instruction", "output", "generator", "dataset", "category",
"new_tokens", "hit_max_new_tokens"}``.

Completed rows are appended to ``<out>.partial.jsonl`` after every batch, so an
interrupted run resumes without regenerating finished prompts. The final JSON
is written once every prompt is done.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.formatting import format_prompt, stop_token_ids
from alignment.train_common import DTYPES
from alignment.utils import append_jsonl, read_json, read_jsonl, write_json


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    batch_size: int = 16
    do_sample: bool = False  # greedy decoding for every comparison
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    dtype: str = "fp16"
    prompt_style: str = "chatml"  # "chatml" (Qwen runs) or "chat_template" (SmolLM2 persona runs)


def load_for_generation(model_path: str, dtype: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPES[dtype], attn_implementation="sdpa")
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.config.use_cache = True
    return model.cuda().eval(), tokenizer


def build_prompt(tokenizer, instruction: str, style: str) -> str:
    if style == "chatml":
        return format_prompt(instruction)
    if style == "chat_template":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
    raise ValueError(style)


@torch.no_grad()
def generate_batch(model, tokenizer, instructions: list[str], cfg: GenerationConfig) -> list[dict]:
    prompts = [build_prompt(tokenizer, x, cfg.prompt_style) for x in instructions]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    stops = stop_token_ids(tokenizer) if cfg.prompt_style == "chatml" else [tokenizer.eos_token_id]
    kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        eos_token_id=stops,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=cfg.repetition_penalty,
    )
    if not cfg.do_sample:
        kwargs.update(temperature=None, top_p=None, top_k=None)
    if cfg.no_repeat_ngram_size:
        kwargs["no_repeat_ngram_size"] = cfg.no_repeat_ngram_size
    out = model.generate(**enc, **kwargs)
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    results = []
    for row in new_tokens:
        stop_mask = torch.isin(row, torch.tensor(stops + [tokenizer.pad_token_id], device=row.device))
        n_new = int(stop_mask.float().argmax().item()) if bool(stop_mask.any()) else row.shape[0]
        text = tokenizer.decode(row[:n_new], skip_special_tokens=True).strip()
        results.append({"output": text, "new_tokens": n_new, "hit_max_new_tokens": not bool(stop_mask.any())})
    return results


def generate_eval_set(
    model_path: str,
    eval_rows: list[dict],
    out_path: str | Path,
    generator: str,
    cfg: GenerationConfig,
    adapter_path: str | None = None,
    model=None,
    tokenizer=None,
) -> list[dict]:
    out_path = Path(out_path)
    if out_path.exists():
        print(f"[gen] {out_path.name} exists - skipping")
        return read_json(out_path)
    partial_path = out_path.with_suffix(".partial.jsonl")
    done = {r["id"]: r for r in read_jsonl(partial_path)}
    todo = [r for r in eval_rows if r["id"] not in done]
    own_model = model is None
    if todo and own_model:
        model, tokenizer = load_for_generation(model_path, cfg.dtype, adapter_path)
    # Longest prompts first: similar lengths per batch, and an OOM shows up immediately.
    todo.sort(key=lambda r: -len(r["instruction"]))
    t0 = time.time()
    for start in range(0, len(todo), cfg.batch_size):
        chunk = todo[start : start + cfg.batch_size]
        outputs = generate_batch(model, tokenizer, [r["instruction"] for r in chunk], cfg)
        for r, o in zip(chunk, outputs):
            row = {
                "id": r["id"],
                "instruction": r["instruction"],
                "output": o["output"],
                "generator": generator,
                "dataset": r.get("dataset", ""),
                "category": r.get("category", ""),
                "new_tokens": o["new_tokens"],
                "hit_max_new_tokens": o["hit_max_new_tokens"],
            }
            append_jsonl(partial_path, row)
            done[r["id"]] = row
        print(f"[gen] {generator}: {min(start + cfg.batch_size, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)")
    final = [done[r["id"]] for r in eval_rows]
    write_json(out_path, final)
    write_json(out_path.with_suffix(".config.json"), {"model_path": model_path, "adapter_path": adapter_path,
                                                      "generator": generator, **asdict(cfg)})
    partial_path.unlink(missing_ok=True)
    if own_model and model is not None:
        del model
        torch.cuda.empty_cache()
    return final


def sanity_check(model_path: str, instructions: list[str], cfg: GenerationConfig, adapter_path: str | None = None):
    """Generate for a handful of prompts and print them for manual inspection."""
    model, tokenizer = load_for_generation(model_path, cfg.dtype, adapter_path)
    outputs = generate_batch(model, tokenizer, instructions, cfg)
    print(f"=== sanity check: {model_path}{' + ' + adapter_path if adapter_path else ''}")
    print(f"    prompt template example:\n{build_prompt(tokenizer, instructions[0], cfg.prompt_style)!r}")
    for x, o in zip(instructions, outputs):
        print(f"\n--- PROMPT: {x}\n--- OUTPUT ({o['new_tokens']} tok, hit_max={o['hit_max_new_tokens']}):\n{o['output']}")
    del model
    torch.cuda.empty_cache()
    return outputs
