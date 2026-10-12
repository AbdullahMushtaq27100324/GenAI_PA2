"""Crash-safe checkpointing.

Layout of a run directory::

    <run>/config.json      exact config used
    <run>/metrics.jsonl    one row per train log / eval (appended + fsynced)
    <run>/last/            resume state: model, optimizer, scheduler, scaler, RNG, progress
    <run>/best/            HF model export selected by the validation metric
    <run>/final/           HF model export at the end of training
    <run>/DONE             written only after final/ exists

Every directory is written to ``<name>.tmp`` and swapped in with renames, so a
crash (or power loss) at any moment leaves either the old or the new version,
never a half-written one.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import torch

from alignment.utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def _swap_in(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


def resolve_dir(final: Path) -> Path | None:
    """Return the complete version of ``final`` (recovering from a crash mid-swap)."""
    final = Path(final)
    tmp = final.with_name(final.name + ".tmp")
    old = final.with_name(final.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists():
        if old.exists():
            shutil.rmtree(old)
        return final
    if old.exists():
        old.rename(final)
        return final
    return None


def export_model(model, tokenizer, out_dir: str | Path, extra: dict | None = None) -> None:
    """Save an HF-loadable model (+ tokenizer) atomically."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_dir(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(tmp)
    if extra is not None:
        write_json(tmp / "checkpoint_info.json", extra)
    _swap_in(tmp, out_dir)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_resume(run_dir: str | Path, model, optimizer, scheduler, scaler, progress: dict) -> None:
    run_dir = Path(run_dir)
    final = run_dir / "last"
    tmp = run_dir / "last.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_dir(tmp)
    torch.save(model.state_dict(), tmp / "model.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng_state(),
        },
        tmp / "trainer_state.pt",
    )
    write_json(tmp / "progress.json", progress)
    _swap_in(tmp, final)


def load_resume(run_dir: str | Path, model, optimizer, scheduler, scaler) -> dict | None:
    """Restore everything from ``<run>/last``; return the saved progress dict or None."""
    last = resolve_dir(Path(run_dir) / "last")
    if last is None or not (last / "progress.json").exists():
        return None
    model.load_state_dict(torch.load(last / "model.pt", map_location="cpu", weights_only=True))
    state = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    set_rng_state(state["rng"])
    return read_json(last / "progress.json")


def truncate_metrics(metrics_path: str | Path, max_step: int) -> None:
    """Drop metric rows logged after the resume point (they will be re-logged)."""
    rows = read_jsonl(metrics_path)
    kept = [r for r in rows if r.get("step", 0) <= max_step]
    if len(kept) != len(rows):
        write_jsonl(metrics_path, kept)


def remove_resume_state(run_dir: str | Path) -> None:
    """Free disk once a run is DONE (best/ and final/ are kept)."""
    for name in ("last", "last.tmp", "last.old"):
        path = Path(run_dir) / name
        if path.exists():
            shutil.rmtree(path)
