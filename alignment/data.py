"""Dataset download, filtering and fixed-seed subsetting.

Every function is idempotent: it writes ``data/processed/<name>.jsonl`` plus a
``<name>.stats.json`` and skips work when the output already exists. All
sampling uses seed 42 so the exact subsets are reproducible; the subsets
themselves are saved so they never need to be regenerated.

Processed record formats:
    SFT:        {"id", "prompt", "response"}
    preference: {"id", "prompt", "chosen", "rejected", "source"}
    eval:       {"id", "instruction", "dataset", ...}
Prompts are raw user instructions; the ChatML template is applied at
tokenization time (``alignment.formatting.format_prompt``).
"""

from __future__ import annotations

import random
from pathlib import Path

from alignment.utils import read_json, read_jsonl, write_json, write_jsonl

SEED = 42


def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _save_ids(out_dir: Path, name: str, train: list, val: list) -> None:
    """Exact subset membership (small enough to commit; the jsonl files are not)."""
    write_json(out_dir / f"{name}.ids.json", {"train": [r["id"] for r in train], "val": [r["id"] for r in val]}, indent=None)


def _sample(rows: list, n: int, seed: int = SEED) -> list:
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} rows available, need {n}")
    idx = random.Random(seed).sample(range(len(rows)), n)
    return [rows[i] for i in idx]


# ------------------------------------------------------------------ SFT: UltraChat
def _ultrachat_rows(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    rows = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        prompt, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if prompt and response:
            rows.append({"id": ex["prompt_id"], "prompt": prompt, "response": response})
    return rows


def prepare_sft(out_dir: Path, n_train: int = 30_000, n_val: int = 1_000) -> dict:
    train_path, val_path = out_dir / "sft_train.jsonl", out_dir / "sft_val.jsonl"
    if _done(train_path) and _done(val_path):
        return read_json(out_dir / "sft.stats.json")
    train_pool = _ultrachat_rows("train_sft")
    val_pool = _ultrachat_rows("test_sft")
    train, val = _sample(train_pool, n_train), _sample(val_pool, n_val)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    stats = {
        "source": "HuggingFaceH4/ultrachat_200k (train_sft / test_sft), first user->assistant turn",
        "pool_train": len(train_pool), "pool_val": len(val_pool), "train": len(train), "val": len(val), "seed": SEED,
    }
    write_json(out_dir / "sft.stats.json", stats)
    _save_ids(out_dir, "sft", train, val)
    return stats


# ------------------------------------------------------------------ HH-RLHF
_H, _A = "\n\nHuman:", "\n\nAssistant:"


def parse_hh_single_turn(chosen: str, rejected: str) -> dict | None:
    """Return {prompt, chosen, rejected} for single-turn HH dialogues, else None."""
    for text in (chosen, rejected):
        if text.count(_H) != 1 or text.count(_A) != 1 or not text.startswith(_H):
            return None
    c_cut, r_cut = chosen.index(_A), rejected.index(_A)
    prompt = chosen[len(_H) : c_cut].strip()
    if prompt != rejected[len(_H) : r_cut].strip():
        return None
    c_resp, r_resp = chosen[c_cut + len(_A) :].strip(), rejected[r_cut + len(_A) :].strip()
    if not prompt or not c_resp or not r_resp or c_resp == r_resp:
        return None
    return {"prompt": prompt, "chosen": c_resp, "rejected": r_resp}


def _hh_rows(subset: str, split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    rows = []
    for i, ex in enumerate(ds):
        parsed = parse_hh_single_turn(ex["chosen"], ex["rejected"])
        if parsed:
            rows.append({"id": f"{subset}-{split}-{i}", **parsed, "source": subset})
    return rows


def _balanced(pools: dict[str, list], n: int) -> list:
    """Equal share per pool where possible; any shortfall is filled from the others."""
    rng = random.Random(SEED)
    shuffled = {k: rng.sample(v, len(v)) for k, v in sorted(pools.items())}
    share = n // len(pools)
    picked, leftovers = [], []
    for rows in shuffled.values():
        picked += rows[:share]
        leftovers += rows[share:]
    picked += rng.sample(leftovers, n - len(picked))
    rng.shuffle(picked)
    return picked


def prepare_hh(out_dir: Path, n_train: int = 15_000, n_val: int = 1_000) -> dict:
    train_path, val_path = out_dir / "hh_train.jsonl", out_dir / "hh_val.jsonl"
    if _done(train_path) and _done(val_path):
        return read_json(out_dir / "hh.stats.json")
    subsets = ("harmless-base", "helpful-base")
    train_pools = {s: _hh_rows(s, "train") for s in subsets}
    val_pools = {s: _hh_rows(s, "test") for s in subsets}
    train, val = _balanced(train_pools, n_train), _balanced(val_pools, n_val)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    stats = {
        "source": "Anthropic/hh-rlhf harmless-base + helpful-base, single-turn only, balanced",
        "pool_train": {k: len(v) for k, v in train_pools.items()},
        "pool_val": {k: len(v) for k, v in val_pools.items()},
        "train": len(train), "val": len(val),
        "train_mix": {s: sum(r["source"] == s for r in train) for s in subsets},
        "val_mix": {s: sum(r["source"] == s for r in val) for s in subsets},
        "seed": SEED,
    }
    write_json(out_dir / "hh.stats.json", stats)
    _save_ids(out_dir, "hh", train, val)
    return stats


# ------------------------------------------------------------------ UltraFeedback
def _uf_rows(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    rows = []
    for ex in ds:
        if ex["score_chosen"] <= ex["score_rejected"]:
            continue
        chosen, rejected = ex["chosen"], ex["rejected"]
        if len(chosen) != 2 or len(rejected) != 2:  # single-turn only
            continue
        prompt = ex["prompt"].strip()
        c_resp, r_resp = chosen[-1]["content"].strip(), rejected[-1]["content"].strip()
        if prompt and c_resp and r_resp and c_resp != r_resp:
            rows.append({"id": ex["prompt_id"], "prompt": prompt, "chosen": c_resp, "rejected": r_resp,
                         "source": "ultrafeedback", "score_chosen": ex["score_chosen"],
                         "score_rejected": ex["score_rejected"]})
    return rows


def prepare_uf(out_dir: Path, n_train: int = 15_000, n_val: int = 1_000) -> dict:
    train_path, val_path = out_dir / "uf_train.jsonl", out_dir / "uf_val.jsonl"
    if _done(train_path) and _done(val_path):
        return read_json(out_dir / "uf.stats.json")
    train_pool, val_pool = _uf_rows("train_prefs"), _uf_rows("test_prefs")
    train, val = _sample(train_pool, n_train), _sample(val_pool, n_val)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    stats = {
        "source": "HuggingFaceH4/ultrafeedback_binarized train_prefs / test_prefs, score_chosen > score_rejected",
        "pool_train": len(train_pool), "pool_val": len(val_pool), "train": len(train), "val": len(val), "seed": SEED,
    }
    write_json(out_dir / "uf.stats.json", stats)
    _save_ids(out_dir, "uf", train, val)
    return stats


# ------------------------------------------------------------------ AlpacaEval subset
def prepare_alpaca_eval(out_path: Path, n: int = 300, seed: int = SEED) -> dict:
    """300 prompts sampled with ``random.Random(42).sample`` from the 805 AlpacaEval prompts."""
    if _done(out_path):
        rows = read_json(out_path)
        return {"n": len(rows), "path": str(out_path)}
    from huggingface_hub import hf_hub_download

    src = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    full = read_json(src)
    idx = random.Random(seed).sample(range(len(full)), n)
    rows = [
        {
            "id": f"alpaca-eval-{seed}-{i:04d}",
            "instruction": full[i]["instruction"],
            "dataset": "alpaca_eval",
            "category": full[i].get("dataset", "unknown"),
            "source_index": i,
        }
        for i in idx
    ]
    write_json(out_path, rows)
    return {"n": len(rows), "pool": len(full), "seed": seed, "path": str(out_path)}


# ------------------------------------------------------------------ Persona LoRA
def prepare_persona(src: Path, out_dir: Path, n_val: int = 20) -> dict:
    train_path, val_path = out_dir / "persona_train.jsonl", out_dir / "persona_val.jsonl"
    if _done(train_path) and _done(val_path):
        return read_json(out_dir / "persona.stats.json")
    rows = read_json(src)
    rows = [{"id": f"persona-{i:03d}", **r} for i, r in enumerate(rows)]
    shuffled = random.Random(SEED).sample(rows, len(rows))
    val, train = shuffled[:n_val], shuffled[n_val:]
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    stats = {"source": str(src), "total": len(rows), "train": len(train), "val": len(val), "seed": SEED}
    write_json(out_dir / "persona.stats.json", stats)
    _save_ids(out_dir, "persona", train, val)
    return stats


def check_persona_eval_disjoint(eval_path: Path, persona_src: Path) -> int:
    """Number of persona eval prompts that also appear in the persona training data (should be 0)."""
    train_prompts = {r["prompt"].strip().lower() for r in read_json(persona_src)}
    return sum(r["instruction"].strip().lower() in train_prompts for r in read_json(eval_path))


def prepare_all(root: Path, sizes: dict) -> dict:
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "sft": prepare_sft(out_dir, sizes["sft_train"], sizes["sft_val"]),
        "hh": prepare_hh(out_dir, sizes["dpo_train"], sizes["dpo_val"]),
        "uf": prepare_uf(out_dir, sizes["dpo_train"], sizes["dpo_val"]),
        "alpaca_eval": prepare_alpaca_eval(root / "eval_sets" / "alpaca_eval_300_seed42.json"),
        "persona": prepare_persona(root / "data" / "persona_lora" / "ramsay_style_preferences.json", out_dir),
    }
    stats["persona_eval_overlap_with_train"] = check_persona_eval_disjoint(
        root / "eval_sets" / "persona_style_eval.json", root / "data" / "persona_lora" / "ramsay_style_preferences.json"
    )
    write_json(out_dir / "all.stats.json", stats)
    return stats
