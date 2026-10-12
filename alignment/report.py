"""Tables for the final report, built only from saved run artifacts (configs, summaries, judgments)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alignment.pairwise import length_stats, summarize_judgments
from alignment.utils import read_json, read_jsonl


def _maybe(path: Path) -> dict:
    return read_json(path) if Path(path).exists() else {}


def experiment_config_table(out: Path, lora_dir: Path, gen_configs: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per submitted training run with the values actually used (manual Section 14.2)."""
    rows = []
    sft = _maybe(out / "sft" / "config.json")
    sft_sum = _maybe(out / "sft" / "summary.json")
    if sft:
        rows.append({
            "run": "SFT", "init": sft["model_name"], "trainable": "full fine-tuning",
            "optimizer": sft["optimizer"], "peak_lr": sft["lr"],
            "schedule": f"{sft['scheduler']} (warmup {sft_sum.get('warmup_steps', '?')} steps)",
            "weight_decay": sft["weight_decay"], "grad_clip": sft["max_grad_norm"],
            "micro_batch": sft["micro_batch_size"], "grad_accum": sft["grad_accum"],
            "effective_batch": sft["micro_batch_size"] * sft["grad_accum"],
            "precision": f"{sft['precision']} autocast, fp32 weights" + (", GradScaler" if sft["precision"] == "fp16" else ""),
            "max_len": sft["max_length"], "epochs/steps": f"{sft['epochs']} ep / {sft_sum.get('total_steps', '?')} steps",
            "beta": "-", "reference": "-", "seed": sft["seed"],
            "selected_ckpt": f"best val loss @ step {sft_sum.get('best_step')}",
        })
    for track, name in (("hh", "HH-RLHF DPO"), ("uf", "UltraFeedback DPO")):
        cfg = _maybe(out / f"dpo_{track}" / "config.json")
        summ = _maybe(out / f"dpo_{track}" / "summary.json")
        if not cfg:
            continue
        rows.append({
            "run": name, "init": "SFT best", "trainable": "full fine-tuning",
            "optimizer": cfg["optimizer"], "peak_lr": cfg["lr"],
            "schedule": f"{cfg['scheduler']} (warmup {cfg['warmup_steps']} steps)",
            "weight_decay": cfg["weight_decay"], "grad_clip": cfg["max_grad_norm"],
            "micro_batch": f"{cfg['micro_batch_size']} pair", "grad_accum": cfg["grad_accum"],
            "effective_batch": cfg["micro_batch_size"] * cfg["grad_accum"],
            "precision": f"{cfg['precision']} autocast, fp32 weights" + (", GradScaler" if cfg["precision"] == "fp16" else ""),
            "max_len": f"{cfg['max_length']} (prompt <= {cfg['max_prompt_length']})",
            "epochs/steps": f"{cfg['epochs']} ep / {summ.get('total_steps', '?')} steps",
            "beta": cfg["beta"],
            "reference": f"frozen SFT best ({cfg['ref_mode']}, {cfg['ref_dtype'] if cfg['ref_mode'] == 'online' else 'fp32'})",
            "seed": cfg["seed"],
            "selected_ckpt": f"best val reward acc @ step {summ.get('best_step')}",
        })
    lcfg = _maybe(lora_dir / "config.json")
    lsum = _maybe(lora_dir / "summary.json")
    if lcfg:
        rows.append({
            "run": "LoRA-DPO (persona)", "init": lcfg["model_name"],
            "trainable": f"LoRA r={lcfg['lora_r']} a={lcfg['lora_alpha']} drop={lcfg['lora_dropout']} on "
                         f"{','.join(lcfg['target_modules'])}",
            "optimizer": "adamw (TRL default)", "peak_lr": lcfg["lr"],
            "schedule": f"{lcfg['scheduler']} (warmup ratio {lcfg['warmup_ratio']})",
            "weight_decay": lcfg["weight_decay"], "grad_clip": lcfg["max_grad_norm"],
            "micro_batch": lcfg["per_device_batch_size"], "grad_accum": lcfg["grad_accum"],
            "effective_batch": lcfg["per_device_batch_size"] * lcfg["grad_accum"],
            "precision": "bf16 autocast, fp32 weights" if lcfg["bf16"] else "fp32",
            "max_len": lcfg["max_length"], "epochs/steps": f"{lcfg['epochs']} ep / {lsum.get('global_steps', '?')} steps",
            "beta": lcfg["beta"], "reference": "base model, adapter disabled", "seed": lcfg["seed"],
            "selected_ckpt": f"min val DPO loss ({lsum.get('best_checkpoint', '?')})",
        })
    df = pd.DataFrame(rows)
    gen = pd.DataFrame([{"generation_set": k, **v} for k, v in gen_configs.items()])
    return df, gen


def results_table(eval_dir: Path, comparisons: dict[str, dict]) -> pd.DataFrame:
    """``comparisons`` maps a label to {"pairs", "judged", "failures", "criterion"}."""
    rows = []
    for label, c in comparisons.items():
        if not Path(c["judged"]).exists():
            continue
        s = summarize_judgments(c["pairs"], c["judged"], c.get("failures"), c.get("criterion", "winner"))
        o = s["overall"]
        lo, hi = o["win_rate_excl_ties_ci95"]
        rows.append({
            "comparison": label, "judge": s["judge_model"], "criterion": s["criterion"], "n": o["n_pairs"],
            "wins": o["wins"], "ties": o["ties"], "losses": o["losses"], "unknown": o["unknown"],
            "failures": o["failures"], "win_rate_excl_ties": round(o["win_rate_excl_ties"], 3),
            "ci95": f"[{lo:.2f}, {hi:.2f}]", "win_rate_ties_half": round(o["win_rate_ties_half"], 3),
            "retried_batches_pairs": sum(v for k, v in s["attempts_histogram"].items() if int(k) > 1),
            **{f"len_{k}": round(v, 2) for k, v in length_stats(c["pairs"]).items()},
        })
    return pd.DataFrame(rows)


def per_category_table(pairs, judged, failures=None, criterion="winner") -> pd.DataFrame:
    s = summarize_judgments(pairs, judged, failures, criterion)
    return pd.DataFrame([
        {"category": cat, "n": r["n_pairs"], "wins": r["wins"], "ties": r["ties"], "losses": r["losses"],
         "unknown": r["unknown"], "win_rate_excl_ties": round(r["win_rate_excl_ties"], 3),
         "win_rate_ties_half": round(r["win_rate_ties_half"], 3)}
        for cat, r in s["per_category"].items()
    ])


def pick_examples(pairs_path, judged_path, criterion="winner", n_each=2, max_chars=700) -> str:
    """Markdown with a few wins, losses and ties for the treatment model (shortest responses first)."""
    pairs = {p["pair_id"]: p for p in read_jsonl(pairs_path)}
    buckets = {"win": [], "loss": [], "tie": []}
    for j in read_jsonl(judged_path):
        p = pairs.get(j["pair_id"])
        if p is None:
            continue
        pos = p["hidden_metadata"]["treatment_position"]
        verdict = j.get(criterion, "unknown")
        outcome = "tie" if verdict == "tie" else ("win" if verdict == pos else "loss" if verdict in "AB" else None)
        if outcome:
            t, b = (p["response_a"], p["response_b"]) if pos == "A" else (p["response_b"], p["response_a"])
            buckets[outcome].append((len(t) + len(b), p, t, b, j.get("reason", "")))
    meta = next(iter(pairs.values()))["hidden_metadata"] if pairs else {}
    lines = []
    for outcome, items in buckets.items():
        for _, p, t, b, reason in sorted(items, key=lambda x: x[0])[:n_each]:
            lines += [f"### {outcome.upper()} - {p['id']} ({p.get('category', '')})",
                      f"**Prompt:** {p['instruction'][:max_chars]}",
                      f"**{meta.get('treatment')}:** {t[:max_chars]}",
                      f"**{meta.get('baseline')}:** {b[:max_chars]}",
                      f"**Judge reason:** {reason}", ""]
    return "\n".join(lines)


_REPO_ROOT = str(Path(__file__).resolve().parents[1]) + "/"


def df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown table; absolute repo paths are shown relative to the repo root, and missing values as '-'."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]

    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return "-"
        return str(v).replace(_REPO_ROOT, "").replace("|", "\\|")

    for _, row in df.iterrows():
        out.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(out)
