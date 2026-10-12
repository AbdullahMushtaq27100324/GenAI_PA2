"""Training-curve figures for the report (PNG + PDF), read from metrics.jsonl files.

Smoothing: training curves show raw per-step values (faint) plus an exponential
moving average, EMA_t = a * EMA_{t-1} + (1 - a) * x_t with a = 0.9 (stated in
every legend). The selected best checkpoint is marked with a dashed vertical line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.utils import read_json, read_jsonl  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EMA_ALPHA = 0.9

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def ema(values, alpha=EMA_ALPHA):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * acc + (1 - alpha) * v
        out.append(acc)
    return out


def _split(run_dir):
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
    return [r for r in rows if r["type"] == "train"], [r for r in rows if r["type"] == "eval"]


def _best_step(run_dir):
    path = Path(run_dir) / "summary.json"
    return read_json(path).get("best_step") if path.exists() else None


def _mark_best(ax, step, label=True):
    if step is not None:
        ax.axvline(step, color=INK2, linestyle="--", linewidth=1, label=f"best ckpt (step {step})" if label else None)


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path.with_suffix(".png")


def plot_sft(run_dir, out_path, title="SFT (Qwen2.5-0.5B, UltraChat 30k)"):
    train, evals = _split(run_dir)
    best = _best_step(run_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    steps = [r["step"] for r in train]
    loss = [r["loss"] for r in train]
    ax1.plot(steps, loss, color=BLUE, alpha=0.25, linewidth=0.8, label="train loss (per step)")
    ax1.plot(steps, ema(loss), color=BLUE, label=f"train loss (EMA a={EMA_ALPHA})")
    ax1.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], color=ORANGE, marker="o",
             markersize=4, label="validation loss (1k held-out)")
    _mark_best(ax1, best)
    ax1.set(xlabel="optimizer step (effective batch 32)", ylabel="response-token NLL", title=title)
    ax1.legend()
    ax2.plot(steps, [r["lr"] for r in train], color=BLUE)
    ax2.set(xlabel="optimizer step", ylabel="learning rate", title="LR schedule (warmup + cosine)")
    return _save(fig, out_path)


def plot_dpo(runs: dict[str, str], out_path, beta=0.1):
    """``runs`` maps a display name (e.g. 'HH-RLHF DPO') to its run directory."""
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    (ax_loss, ax_acc), (ax_margin, ax_rew) = axes
    for k, (name, run_dir) in enumerate(runs.items()):
        c = colors[k % len(colors)]
        train, evals = _split(run_dir)
        best = _best_step(run_dir)
        ts = [r["step"] for r in train]
        es = [r["step"] for r in evals]
        ax_loss.plot(ts, ema([r["loss"] for r in train]), color=c, alpha=0.55, linewidth=1,
                     label=f"{name}: train (EMA a={EMA_ALPHA})")
        ax_loss.plot(es, [r["val_loss"] for r in evals], color=c, marker="o", markersize=4, label=f"{name}: validation")
        ax_acc.plot(es, [r["val_reward_accuracy"] for r in evals], color=c, marker="o", markersize=4, label=name)
        ax_margin.plot(es, [r["val_reward_margin"] for r in evals], color=c, marker="o", markersize=4, label=name)
        ax_rew.plot(es, [r["val_chosen_reward"] for r in evals], color=c, marker="o", markersize=4,
                    label=f"{name}: chosen")
        ax_rew.plot(es, [r["val_rejected_reward"] for r in evals], color=c, marker="s", markersize=4,
                    linestyle=":", label=f"{name}: rejected")
        for ax in (ax_loss, ax_acc, ax_margin):
            if best is not None:
                ax.axvline(best, color=c, linestyle="--", linewidth=1)
    ax_loss.axhline(0.6931, color=INK2, linewidth=0.8, linestyle=":", label="log 2 (no preference)")
    ax_loss.set(xlabel="optimizer step", ylabel="DPO loss", title="DPO loss")
    ax_acc.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax_acc.set(xlabel="optimizer step", ylabel="fraction r_w > r_l",
               title="Validation reward accuracy (dashed = best ckpt)")
    ax_margin.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
    ax_margin.set(xlabel="optimizer step", ylabel="mean r_w - r_l", title=f"Validation reward margin (beta={beta})")
    ax_rew.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
    ax_rew.set(xlabel="optimizer step", ylabel="beta * (log pi - log ref)", title="Validation implicit rewards")
    for ax in axes.flat:
        ax.legend()
    return _save(fig, out_path)


def plot_lora(log_history: list[dict], out_path, title="LoRA-DPO (SmolLM2-360M-Instruct, persona)"):
    train = [r for r in log_history if "loss" in r and "eval_loss" not in r]
    evals = [r for r in log_history if "eval_loss" in r]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.2))
    ax1.plot([r["step"] for r in train], [r["loss"] for r in train], color=BLUE, marker="o", markersize=3,
             label="train DPO loss (per logging step)")
    ax1.plot([r["step"] for r in evals], [r["eval_loss"] for r in evals], color=ORANGE, marker="o", markersize=4,
             label="validation DPO loss")
    ax1.set(xlabel="optimizer step", ylabel="DPO loss", title=title)
    ax1.legend()
    es = [r["step"] for r in evals]
    if evals and "eval_rewards/accuracies" in evals[0]:
        ax2.plot(es, [r["eval_rewards/accuracies"] for r in evals], color=BLUE, marker="o", markersize=4)
    ax2.set(xlabel="optimizer step", ylabel="fraction r_w > r_l", title="Validation reward accuracy")
    if evals and "eval_rewards/margins" in evals[0]:
        ax3.plot(es, [r["eval_rewards/margins"] for r in evals], color=BLUE, marker="o", markersize=4)
    ax3.set(xlabel="optimizer step", ylabel="mean r_w - r_l", title="Validation reward margin")
    return _save(fig, out_path)
