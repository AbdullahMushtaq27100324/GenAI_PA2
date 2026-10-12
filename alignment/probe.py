"""Worst-case memory/speed probe, run in a subprocess so an OOM cannot poison the notebook kernel.

Uses full-length random sequences (the worst case for memory) and exactly the
same model loading, log-prob path, optimizer and precision as the real loops.

    python -m alignment.probe --kind sft --micro_batch 4 --optimizer adamw8bit
    python -m alignment.probe --kind dpo --optimizer rmsprop --ref_mode online --ref_dtype fp16

Prints one JSON line: {"ok": bool, "peak_gb": float, "sec_per_micro": float, ...}.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM

from alignment.losses import concat_logps, dpo_losses, sft_loss_from_batch
from alignment.train_common import DTYPES, autocast, make_optimizer


def _random_batch(n_rows, prompt_len, total_len, vocab):
    ids = torch.randint(0, vocab, (n_rows, total_len))
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sft", "dpo"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--optimizer", default="adamw8bit")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--prompt_length", type=int, default=256)
    ap.add_argument("--ref_mode", default="online")
    ap.add_argument("--ref_dtype", default="fp16")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--grad_checkpointing", type=int, default=1)
    args = ap.parse_args()
    result = {**vars(args), "ok": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
        model.config.use_cache = False
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.cuda().train()
        vocab = model.config.vocab_size
        reference = None
        if args.kind == "dpo" and args.ref_mode == "online":
            reference = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPES[args.ref_dtype]).cuda().eval()
            reference.requires_grad_(False)
        optimizer = make_optimizer(model, args.optimizer, 1e-6, 0.0)
        # init_scale=1: random-token losses would overflow the default 65536 scale, every step would be
        # skipped, and the optimizer state (the biggest memory item) would never be allocated.
        scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"), init_scale=1.0)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(args.steps):
            t0 = time.time()
            if args.kind == "sft":
                batch = _random_batch(args.micro_batch, 32, args.max_length, vocab)
                with autocast(args.precision):
                    loss, _ = sft_loss_from_batch(model, batch)
            else:
                batch = _random_batch(2 * args.micro_batch, args.prompt_length, args.max_length, vocab)
                with autocast(args.precision):
                    pc, pr = concat_logps(model, batch)
                    if reference is not None:
                        with torch.no_grad():
                            rc, rr = concat_logps(reference, batch)
                    else:
                        rc, rr = pc.detach(), pr.detach()
                losses, _, _ = dpo_losses(pc.float(), pr.float(), rc.float(), rr.float(), 0.1)
                loss = losses.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        n_state = sum(len(v) for v in optimizer.state.values())
        assert n_state > 0, "optimizer state was never materialized - probe would underestimate memory"
        result.update(
            ok=True,
            optimizer_state_tensors=n_state,
            peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
            reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3),
            sec_per_micro=round(sum(times[1:]) / max(1, len(times) - 1), 3),
            total_gpu_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
        )
    except torch.OutOfMemoryError as exc:
        result.update(error=f"OOM: {str(exc)[:200]}")
    except Exception as exc:  # report, don't crash the caller
        result.update(error=f"{type(exc).__name__}: {str(exc)[:300]}")
    print("PROBE_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
