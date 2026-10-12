"""From-scratch supervised fine-tuning loop (no HF Trainer).

Implements every manual requirement (Task 3 / Section 9.1):
prompt formatting, response-only loss masking, padding + attention masks,
gradient accumulation, fp16 autocast with fp32 trainable weights and GradScaler,
gradient clipping, validation loss, best + final checkpoints. Adds crash-safe
resume from ``<run>/last``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import encode_example, pad_examples, pad_token_id
from alignment.ckpt import export_model, load_resume, remove_resume_state, save_resume, truncate_metrics
from alignment.formatting import format_prompt
from alignment.losses import sft_loss_from_batch
from alignment.train_common import (
    SimulatedCrash,
    autocast,
    make_optimizer,
    make_scaler,
    make_scheduler,
    micro_batch_order,
)
from alignment.utils import append_jsonl, ensure_dir, gpu_memory_gb, read_jsonl, set_seed, write_json

MAX_CONSECUTIVE_SKIPS = 25  # consecutive GradScaler-skipped steps before aborting (a few early skips are normal)


@dataclass
class SFTConfig:
    run_dir: str
    train_path: str
    val_path: str
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_train: int | None = None
    max_val: int | None = None
    max_length: int = 512
    micro_batch_size: int = 2
    grad_accum: int = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    epochs: int = 1
    optimizer: str = "adamw8bit"
    precision: str = "fp16"
    grad_checkpointing: bool = True
    group_by_length: bool = True
    eval_every: int = 100
    eval_at_start: bool = True
    eval_batch_size: int = 8
    save_every: int = 25
    seed: int = 42
    max_steps: int | None = None
    stop_after_steps: int | None = None
    keep_resume_state: bool = False


def _encode_rows(tokenizer, rows, max_length):
    return [encode_example(tokenizer, format_prompt(r["prompt"]), r["response"], max_length) for r in rows]


def _response_tokens(labels: torch.Tensor) -> int:
    return int(labels[:, 1:].ne(-100).sum().item())


@torch.no_grad()
def evaluate_sft(model, encoded, pad_id, batch_size, precision) -> dict:
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i][0]))
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(order), batch_size):
        batch = pad_examples([encoded[i] for i in order[start : start + batch_size]], pad_id)
        with autocast(precision):
            loss, n_tok = sft_loss_from_batch(model, batch, token_normalizer=1.0)
        total_nll += float(loss.item())
        total_tokens += int(n_tok.item())
    model.train()
    val_loss = total_nll / max(1, total_tokens)
    return {"val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 50)), "val_tokens": total_tokens}


def train_sft(cfg: SFTConfig) -> dict:
    run_dir = ensure_dir(cfg.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if (run_dir / "DONE").exists():
        print(f"[sft] {run_dir} already DONE - skipping")
        return {"status": "skipped"}
    write_json(run_dir / "config.json", asdict(cfg))
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_id = pad_token_id(tokenizer)
    train_rows = read_jsonl(cfg.train_path)[: cfg.max_train]
    val_rows = read_jsonl(cfg.val_path)[: cfg.max_val]
    train_enc = _encode_rows(tokenizer, train_rows, cfg.max_length)
    val_enc = _encode_rows(tokenizer, val_rows, cfg.max_length)
    lengths = [len(ids) for ids, _ in train_enc]

    micro_per_epoch = math.ceil(len(train_enc) / cfg.micro_batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    warmup_steps = math.ceil(cfg.warmup_ratio * total_steps)
    print(f"[sft] train={len(train_enc)} val={len(val_enc)} steps={total_steps} warmup={warmup_steps}")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda().train()

    optimizer = make_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps, cfg.scheduler)
    scaler = make_scaler(cfg.precision)

    progress = load_resume(run_dir, model, optimizer, scheduler, scaler)
    if progress is None:
        progress = {"step": 0, "best_val_loss": None, "best_step": None, "train_seconds": 0.0, "evaluated_start": False}
        if metrics_path.exists():
            metrics_path.unlink()
    else:
        truncate_metrics(metrics_path, progress["step"])
        print(f"[sft] resumed from step {progress['step']}")

    def log(row):
        append_jsonl(metrics_path, row)

    def run_eval(step):
        result = evaluate_sft(model, val_enc, pad_id, cfg.eval_batch_size, cfg.precision)
        log({"type": "eval", "step": step, **result})
        print(f"[sft] step {step} val_loss={result['val_loss']:.4f}")
        best = progress["best_val_loss"]
        if best is None or result["val_loss"] < best:
            progress["best_val_loss"] = result["val_loss"]
            progress["best_step"] = step
            export_model(model, tokenizer, run_dir / "best", {"step": step, **result, "selection": "min val_loss"})
        return result

    if cfg.eval_at_start and not progress["evaluated_start"]:
        run_eval(0)
        progress["evaluated_start"] = True

    session_start_step = progress["step"]
    torch.cuda.reset_peak_memory_stats()
    while progress["step"] < total_steps:
        step = progress["step"]
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        order = micro_batch_order(lengths, cfg.micro_batch_size, cfg.seed, epoch, cfg.group_by_length)
        group = order[step_in_epoch * cfg.grad_accum : (step_in_epoch + 1) * cfg.grad_accum]
        batches = [pad_examples([train_enc[i] for i in idx], pad_id) for idx in group]
        normalizer = max(1, sum(_response_tokens(b["labels"]) for b in batches))

        t0 = time.time()
        step_loss = 0.0
        for batch in batches:
            with autocast(cfg.precision):
                loss, _ = sft_loss_from_batch(model, batch, token_normalizer=float(normalizer))
            scaler.scale(loss).backward()
            step_loss += float(loss.item())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        progress["consecutive_skips"] = progress.get("consecutive_skips", 0) + 1 if skipped else 0
        if progress["consecutive_skips"] >= MAX_CONSECUTIVE_SKIPS:
            raise FloatingPointError(
                f"{MAX_CONSECUTIVE_SKIPS} consecutive fp16 overflow steps - set precision='bf16' and rerun"
            )

        progress["step"] = step + 1
        progress["train_seconds"] += time.time() - t0
        log(
            {
                "type": "train",
                "step": progress["step"],
                "epoch": epoch,
                "loss": step_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "loss_scale": scale_before,
                "skipped_step": bool(skipped),
                "response_tokens": normalizer,
                "step_seconds": round(time.time() - t0, 3),
                "train_seconds": round(progress["train_seconds"], 1),
                **gpu_memory_gb(),
            }
        )
        done = progress["step"] >= total_steps
        if progress["step"] % cfg.eval_every == 0 or done:
            run_eval(progress["step"])
        if progress["step"] % cfg.save_every == 0 or done:
            save_resume(run_dir, model, optimizer, scheduler, scaler, progress)
        if cfg.stop_after_steps is not None and progress["step"] - session_start_step >= cfg.stop_after_steps and not done:
            save_resume(run_dir, model, optimizer, scheduler, scaler, progress)
            raise SimulatedCrash(f"simulated crash after step {progress['step']}")

    export_model(model, tokenizer, run_dir / "final", {"step": progress["step"], "selection": "final"})
    summary = {
        "status": "done",
        "total_steps": total_steps,
        "best_step": progress["best_step"],
        "best_val_loss": progress["best_val_loss"],
        "train_hours": round(progress["train_seconds"] / 3600, 3),
        "train_examples": len(train_enc),
        "val_examples": len(val_enc),
        "warmup_steps": warmup_steps,
        "effective_batch": cfg.micro_batch_size * cfg.grad_accum,
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    if not cfg.keep_resume_state:
        remove_resume_state(run_dir)
    del model, optimizer
    torch.cuda.empty_cache()
    return summary
