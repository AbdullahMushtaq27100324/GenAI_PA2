"""Task 5 ("Idiot Sandwich"): TRL DPOTrainer + PEFT LoRA persona tuning.

This is the only track where TRL / HF Trainer are allowed. The data rows
(prompt / chosen / rejected strings) are converted to TRL's conversational
format so the model's own chat template is applied consistently in training
and in generation:
    prompt   = [{"role": "user", "content": ...}]
    chosen   = [{"role": "assistant", "content": ...}]
    rejected = [{"role": "assistant", "content": ...}]
With ``peft_config`` and ``ref_model=None``, TRL uses the base model with the
adapter disabled as the frozen reference, so no second model copy is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from alignment.utils import ensure_dir, read_json, read_jsonl, write_json

LLAMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoraDPOConfig:
    run_dir: str
    train_path: str
    val_path: str
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    max_train: int | None = None
    max_val: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(LLAMA_TARGETS))
    lr: float = 5e-5
    beta: float = 0.1
    epochs: int = 5
    per_device_batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 512
    bf16: bool = True
    eval_steps: int = 5
    seed: int = 42


def to_conversational(rows: list[dict]) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ]


def train_lora_dpo(cfg: LoraDPOConfig) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import DPOConfig, DPOTrainer

    run_dir = ensure_dir(cfg.run_dir)
    if (run_dir / "DONE").exists():
        print(f"[lora] {run_dir} already DONE - skipping")
        return read_json(run_dir / "summary.json")
    write_json(run_dir / "config.json", asdict(cfg))

    train_ds = Dataset.from_list(to_conversational(read_jsonl(cfg.train_path)[: cfg.max_train]))
    val_ds = Dataset.from_list(to_conversational(read_jsonl(cfg.val_path)[: cfg.max_val]))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 base weights + bf16 autocast; the frozen base is tiny (360M) so memory is not a concern.
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=str(run_dir / "trainer"),
        beta=cfg.beta,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        lr_scheduler_type=cfg.scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        max_length=cfg.max_length,
        truncation_mode="keep_start",
        bf16=cfg.bf16,
        gradient_checkpointing=False,  # TRL defaults to True; unnecessary for a 360M model and noisy with PEFT
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        # Validation reward accuracy saturates at 1.0 within the first epoch on 20 held-out pairs, and ties never
        # replace the best checkpoint, so it cannot rank checkpoints. Validation DPO loss keeps moving.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=cfg.seed,
        dataloader_num_workers=0,
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
    trainer.train(resume_from_checkpoint=last)
    final_eval = trainer.evaluate()  # model is the best checkpoint after load_best_model_at_end

    trainer.model.save_pretrained(run_dir / "best_adapter")
    tokenizer.save_pretrained(run_dir / "best_adapter")
    write_json(run_dir / "log_history.json", trainer.state.log_history)
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    summary = {
        "status": "done",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_eval_loss": trainer.state.best_metric,
        "final_eval_of_best": final_eval,
        "global_steps": trainer.state.global_step,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_fraction": n_trainable / n_total,
        "effective_batch": cfg.per_device_batch_size * cfg.grad_accum,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    del trainer, model
    torch.cuda.empty_cache()
    return summary
