"""From-scratch DPO loop (no TRL / HF Trainer), shared by the HH-RLHF and UltraFeedback runs.

Manual Task 4 / Section 10.1 requirements:
1. policy = SFT best checkpoint (trainable, fp32 weights, fp16 autocast + GradScaler)
2. reference = a second, frozen copy of the SFT best checkpoint (no_grad)
3. response-only log-probs for chosen and rejected responses
4. DPO loss  -log sigmoid(beta * (policy gap - reference gap))
5. validation DPO loss, reward accuracy (strict r_w > r_l) and reward margin
6. best checkpoint by validation reward accuracy (tie-break: margin) + final

``ref_mode``:
* ``online``      - the frozen reference runs a no_grad forward on every
                    micro-batch (literal recipe; needs memory for 2 models).
* ``precomputed`` - the frozen reference scores every training pair once before
                    training and is then freed. Mathematically identical because
                    the reference never changes; used if memory is too tight.
Validation reference log-probs are always cached once (the reference is frozen).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import concat_chosen_rejected, encode_example, pad_examples, pad_token_id
from alignment.ckpt import export_model, load_resume, remove_resume_state, save_resume, truncate_metrics
from alignment.formatting import format_prompt
from alignment.losses import concat_logps, dpo_losses, reward_stats
from alignment.train_common import (
    DTYPES,
    SimulatedCrash,
    autocast,
    make_optimizer,
    make_scaler,
    make_scheduler,
    micro_batch_order,
)
from alignment.utils import append_jsonl, ensure_dir, free_cuda, gpu_memory_gb, read_jsonl, set_seed, write_json

MAX_CONSECUTIVE_SKIPS = 25  # consecutive GradScaler-skipped steps before aborting (a few early skips are normal)


@dataclass
class DPOConfig:
    run_dir: str
    sft_checkpoint: str
    train_path: str
    val_path: str
    max_train: int | None = None
    max_val: int | None = None
    max_length: int = 512
    max_prompt_length: int = 256
    micro_batch_size: int = 1
    grad_accum: int = 32
    lr: float = 1e-6
    warmup_steps: int = 0
    scheduler: str = "constant"
    beta: float = 0.1
    optimizer: str = "rmsprop"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    epochs: int = 1
    precision: str = "fp16"
    grad_checkpointing: bool = True
    ref_mode: str = "online"
    ref_dtype: str = "fp16"
    eval_every: int = 50
    eval_batch_size: int = 4
    save_every: int = 25
    seed: int = 42
    max_steps: int | None = None
    stop_after_steps: int | None = None
    keep_resume_state: bool = False


def _encode_pairs(tokenizer, rows, cfg):
    encoded = []
    for r in rows:
        prompt = format_prompt(r["prompt"])
        chosen = encode_example(tokenizer, prompt, r["chosen"], cfg.max_length, cfg.max_prompt_length)
        rejected = encode_example(tokenizer, prompt, r["rejected"], cfg.max_length, cfg.max_prompt_length)
        encoded.append((chosen, rejected))
    return encoded


def _stacked_batch(pairs, pad_id):
    chosen = pad_examples([p[0] for p in pairs], pad_id)
    rejected = pad_examples([p[1] for p in pairs], pad_id)
    return concat_chosen_rejected({"chosen": chosen, "rejected": rejected}, pad_id)


def _val_batches(n, batch_size):
    return [list(range(i, min(n, i + batch_size))) for i in range(0, n, batch_size)]


def _load_model(path, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    return model.to(device)


@torch.no_grad()
def score_reference(reference, encoded, batches, pad_id, precision, desc="ref") -> torch.Tensor:
    """Return ``[N, 2]`` (chosen, rejected) reference log-probs, in row order."""
    reference.eval()
    out = torch.zeros(len(encoded), 2, dtype=torch.float32)
    t0 = time.time()
    for k, idx in enumerate(batches):
        batch = _stacked_batch([encoded[i] for i in idx], pad_id)
        with autocast(precision):
            chosen, rejected = concat_logps(reference, batch)
        out[idx, 0] = chosen.float().cpu()
        out[idx, 1] = rejected.float().cpu()
        if (k + 1) % 500 == 0:
            print(f"[{desc}] {k + 1}/{len(batches)} batches ({time.time() - t0:.0f}s)")
    return out


@torch.no_grad()
def evaluate_dpo(policy, val_enc, val_ref, val_batches, pad_id, beta, precision) -> dict:
    policy.eval()
    sums = {"val_loss": 0.0, "val_reward_accuracy": 0.0, "val_reward_margin": 0.0,
            "val_chosen_reward": 0.0, "val_rejected_reward": 0.0,
            "val_policy_chosen_logp": 0.0, "val_policy_rejected_logp": 0.0}
    for idx in val_batches:
        batch = _stacked_batch([val_enc[i] for i in idx], pad_id)
        with autocast(precision):
            pc, pr = concat_logps(policy, batch)
        pc, pr = pc.float(), pr.float()
        rc, rr = val_ref[idx, 0].to(pc.device), val_ref[idx, 1].to(pc.device)
        losses, cr, rj = dpo_losses(pc, pr, rc, rr, beta)
        stats = reward_stats(cr, rj)
        sums["val_loss"] += losses.sum().item()
        sums["val_reward_accuracy"] += stats["reward_accuracy"].sum().item()
        sums["val_reward_margin"] += stats["reward_margin"].sum().item()
        sums["val_chosen_reward"] += cr.sum().item()
        sums["val_rejected_reward"] += rj.sum().item()
        sums["val_policy_chosen_logp"] += pc.sum().item()
        sums["val_policy_rejected_logp"] += pr.sum().item()
    policy.train()
    n = max(1, len(val_enc))
    return {k: v / n for k, v in sums.items()}


def train_dpo(cfg: DPOConfig) -> dict:
    run_dir = ensure_dir(cfg.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if (run_dir / "DONE").exists():
        print(f"[dpo] {run_dir} already DONE - skipping")
        return {"status": "skipped"}
    write_json(run_dir / "config.json", asdict(cfg))
    set_seed(cfg.seed)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    pad_id = pad_token_id(tokenizer)
    train_rows = read_jsonl(cfg.train_path)[: cfg.max_train]
    val_rows = read_jsonl(cfg.val_path)[: cfg.max_val]
    train_enc = _encode_pairs(tokenizer, train_rows, cfg)
    val_enc = _encode_pairs(tokenizer, val_rows, cfg)
    lengths = [max(len(c[0]), len(r[0])) for c, r in train_enc]
    val_batches = _val_batches(len(val_enc), cfg.eval_batch_size)

    micro_per_epoch = math.ceil(len(train_enc) / cfg.micro_batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    print(f"[dpo] train={len(train_enc)} val={len(val_enc)} steps={total_steps} ref_mode={cfg.ref_mode}")

    # ---- frozen reference: a second copy of the SFT best checkpoint ------------
    ref_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    ref_dtype = DTYPES[cfg.ref_dtype] if cfg.ref_mode == "online" else torch.float32
    val_ref_path = run_dir / "ref_val_logps.pt"
    train_ref_path = run_dir / "ref_train_logps.pt"
    need_val = not val_ref_path.exists()
    need_train = cfg.ref_mode == "precomputed" and not train_ref_path.exists()
    reference = None
    if cfg.ref_mode == "online" or need_val or need_train:
        reference = _load_model(cfg.sft_checkpoint, ref_dtype, ref_device)
        reference.eval().requires_grad_(False)
    if need_val:
        torch.save(score_reference(reference, val_enc, val_batches, pad_id, cfg.precision, "ref-val"), val_ref_path)
    val_ref = torch.load(val_ref_path)
    train_ref = None
    if cfg.ref_mode == "precomputed":
        if need_train:
            batches = [[i] for i in range(len(train_enc))] if cfg.micro_batch_size == 1 else None
            if batches is None:
                raise ValueError("precomputed reference requires micro_batch_size == 1 (batch-invariant padding)")
            torch.save(score_reference(reference, train_enc, batches, pad_id, cfg.precision, "ref-train"), train_ref_path)
        train_ref = torch.load(train_ref_path)
        del reference
        reference = None
        free_cuda()

    # ---- trainable policy: the SFT best checkpoint in fp32 ---------------------
    policy = _load_model(cfg.sft_checkpoint, torch.float32, device)
    if cfg.grad_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()

    optimizer = make_optimizer(policy, cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = make_scheduler(optimizer, total_steps, cfg.warmup_steps, cfg.scheduler)
    scaler = make_scaler(cfg.precision)

    progress = load_resume(run_dir, policy, optimizer, scheduler, scaler)
    if progress is None:
        progress = {"step": 0, "best_acc": None, "best_margin": None, "best_step": None,
                    "train_seconds": 0.0, "evaluated_start": False}
        if metrics_path.exists():
            metrics_path.unlink()
    else:
        truncate_metrics(metrics_path, progress["step"])
        print(f"[dpo] resumed from step {progress['step']}")

    def log(row):
        append_jsonl(metrics_path, row)

    def run_eval(step):
        result = evaluate_dpo(policy, val_enc, val_ref, val_batches, pad_id, cfg.beta, cfg.precision)
        log({"type": "eval", "step": step, **result})
        print(f"[dpo] step {step} val_loss={result['val_loss']:.4f} "
              f"acc={result['val_reward_accuracy']:.3f} margin={result['val_reward_margin']:.4f}")
        if step > 0:  # step 0 is the reference itself (accuracy 0 by definition)
            acc, margin = result["val_reward_accuracy"], result["val_reward_margin"]
            best = (progress["best_acc"], progress["best_margin"])
            if best[0] is None or (acc, margin) > best:
                progress["best_acc"], progress["best_margin"], progress["best_step"] = acc, margin, step
                export_model(policy, tokenizer, run_dir / "best",
                             {"step": step, **result, "selection": "max val_reward_accuracy, tie-break margin"})
        return result

    if not progress["evaluated_start"]:
        run_eval(0)
        progress["evaluated_start"] = True

    session_start_step = progress["step"]
    torch.cuda.reset_peak_memory_stats()
    while progress["step"] < total_steps:
        step = progress["step"]
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        order = micro_batch_order(lengths, cfg.micro_batch_size, cfg.seed, epoch, group_by_length=False)
        group = order[step_in_epoch * cfg.grad_accum : (step_in_epoch + 1) * cfg.grad_accum]
        n_pairs = sum(len(idx) for idx in group)

        t0 = time.time()
        agg = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0}
        for idx in group:
            batch = _stacked_batch([train_enc[i] for i in idx], pad_id)
            with autocast(cfg.precision):
                pc, pr = concat_logps(policy, batch)
                if train_ref is not None:
                    rc, rr = train_ref[idx, 0].to(device), train_ref[idx, 1].to(device)
                else:
                    with torch.no_grad():
                        rc, rr = concat_logps(reference, batch)
            losses, cr, rj = dpo_losses(pc.float(), pr.float(), rc.float(), rr.float(), cfg.beta)
            scaler.scale(losses.sum() / n_pairs).backward()
            stats = reward_stats(cr, rj)
            agg["loss"] += losses.detach().sum().item() / n_pairs
            for key in ("reward_accuracy", "reward_margin", "chosen_reward", "rejected_reward"):
                agg[key] += stats[key].sum().item() / n_pairs
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
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
        log({"type": "train", "step": progress["step"], "epoch": epoch, **agg,
             "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm), "loss_scale": scale_before,
             "skipped_step": bool(skipped), "step_seconds": round(time.time() - t0, 3),
             "train_seconds": round(progress["train_seconds"], 1), **gpu_memory_gb()})
        done = progress["step"] >= total_steps
        if progress["step"] % cfg.eval_every == 0 or done:
            run_eval(progress["step"])
        if progress["step"] % cfg.save_every == 0 or done:
            save_resume(run_dir, policy, optimizer, scheduler, scaler, progress)
        if cfg.stop_after_steps is not None and progress["step"] - session_start_step >= cfg.stop_after_steps and not done:
            save_resume(run_dir, policy, optimizer, scheduler, scaler, progress)
            raise SimulatedCrash(f"simulated crash after step {progress['step']}")

    if progress["best_step"] is None:  # zero-step edge case: fall back to final weights
        export_model(policy, tokenizer, run_dir / "best", {"step": progress["step"], "selection": "final (no eval)"})
    export_model(policy, tokenizer, run_dir / "final", {"step": progress["step"], "selection": "final"})
    summary = {
        "status": "done",
        "total_steps": total_steps,
        "best_step": progress["best_step"],
        "best_val_reward_accuracy": progress["best_acc"],
        "best_val_reward_margin": progress["best_margin"],
        "train_hours": round(progress["train_seconds"] / 3600, 3),
        "train_pairs": len(train_enc),
        "val_pairs": len(val_enc),
        "effective_batch": cfg.micro_batch_size * cfg.grad_accum,
        "reference_checkpoint": str(cfg.sft_checkpoint),
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    if not cfg.keep_resume_state:
        remove_resume_state(run_dir)
    del policy, optimizer, reference
    free_cuda()
    return summary
