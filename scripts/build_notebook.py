"""Generate notebooks/PA2_local_pipeline.ipynb (the single driver notebook for every experiment).

Run:  python scripts/build_notebook.py
The notebook itself is the deliverable; this builder only keeps its source reviewable as code.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str, list[str]]] = []  # (cell_type, source, tags)


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n"), []))


def code(src: str, tags: list[str] | None = None) -> None:
    CELLS.append(("code", src.strip("\n"), tags or []))


# =============================================================================
md(r"""
# PA2 — Instruction Tuning and DPO on Qwen2.5-0.5B (local pipeline)

The single driver notebook for every experiment in the assignment. Core code lives in `alignment/`, and `submission_adapter.py` wraps it.

| Section | Manual task |
|---|---|
| 0. Setup + tests | Task 1 (local part; the Kaggle T4×2 check is in `notebooks/kaggle_setup_and_judge.ipynb`) |
| 1. Data | §6 Model and data (fixed seed-42 subsets, saved) |
| 2. Memory/speed pilot | resource decisions for the 8 GB RTX 3050 (documented deviations) |
| 3. SFT | Task 3 |
| 4. DPO ×2 | Task 4 (HH-RLHF and UltraFeedback, from-scratch) |
| 5. LoRA-DPO | Task 5 (TRL DPOTrainer + PEFT, persona) |
| 6. Generation | Task 2 (baseline) + Task 6 (sanity check, SFT/DPO outputs) |
| 7. Pairwise files | Task 6 (blinded, randomized A/B) |
| 8. Judging | Task 7 (Kaggle model proxy; `mock` judge in the dry run) |
| 9. Results tables | Task 8 report material |

**How to run**
1. **Dry run first** with `DRY_RUN = True`. It runs every stage end to end with tiny sizes into `outputs_dryrun/`, including a forced kill-and-resume test.
2. **Full run**: `bash scripts/run_pipeline.sh full`. This runs headless through papermill, so closing the editor or terminal does not kill it.
3. **After a crash or power loss**, run the same command again. Finished stages are skipped (`DONE` markers), training resumes from `<run>/last/`, and generation and judging resume from their partial files.

**Checkpoints:** every run keeps `best/` (selected by validation metric), `final/`, and `last/` (resume state, deleted once the run is DONE). Downstream stages always use `best/`.
""")

code(r"""
# ---- run mode -------------------------------------------------------------------------------
DRY_RUN = True              # True: tiny end-to-end test of every stage. False: full experiments.
RUN_JUDGE_LOCALLY = False   # full run only: True if `kaggle b init -y` wrote Kaggle model-proxy creds to .env
JUDGE_MODEL = "google/gemini-3.6-flash"
""", tags=["parameters"])

code(r"""
import os, sys, json, time, subprocess, shutil, zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch, transformers, datasets, trl, peft, bitsandbytes
import pandas as pd
assert int(transformers.__version__.split(".")[0]) < 5, f"manual requires transformers<5, kernel has {transformers.__version__} ({sys.executable})"
from IPython.display import Image, Markdown, display

from alignment import data as D
from alignment.train_sft import SFTConfig, train_sft
from alignment.train_dpo import DPOConfig, train_dpo
from alignment.train_lora_dpo import LoraDPOConfig, train_lora_dpo
from alignment.train_common import SimulatedCrash
from alignment.generate import GenerationConfig, generate_eval_set, sanity_check
from alignment.pairwise import build_pairs, summarize_judgments
from alignment import plots, report
from alignment.utils import free_cuda, read_json, read_jsonl, write_json

SEED = 42
OUT = ROOT / ("outputs_dryrun" if DRY_RUN else "outputs")
OUT.mkdir(exist_ok=True)
PROC = ROOT / "data" / "processed"
FIG = OUT / "figures"
TABLES = OUT / "report_tables"
GEN_DIR = OUT / "generations"
EVAL_DIR = OUT / "eval"
for d in (FIG, TABLES, GEN_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
PERSONA_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
PRECISION = "fp16"  # manual: fp16 autocast + fp32 weights + GradScaler. Loops abort with a clear error if fp16 overflows.

T0 = time.time()
def stage(name):
    print(f"\n{'=' * 100}\n[{time.strftime('%H:%M:%S')}] {name}   (elapsed {(time.time() - T0) / 60:.1f} min)\n{'=' * 100}")

stage(f"MODE = {'DRY RUN' if DRY_RUN else 'FULL RUN'} -> {OUT}")
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| trl", trl.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__)
assert torch.cuda.is_available(), "CUDA GPU required"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 2**30:.2f} GiB, count={torch.cuda.device_count()}, bf16={torch.cuda.is_bf16_supported()}")
""")

# =============================================================================
md("## 0. Tests: public adapter smoke tests + our own edge-case tests")
code(r"""
stage("0. tests")
res = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], cwd=ROOT, capture_output=True, text=True,
                     env={**os.environ, "PYTHONPATH": str(ROOT)})
print(res.stdout[-3000:], res.stderr[-2000:])
assert res.returncode == 0, "tests failed - fix before training"
""")

# =============================================================================
md(r"""
## 1. Data (fixed seed-42 subsets, saved to `data/processed/` + `eval_sets/`)
Data prep always uses the **full** sizes (30k/1k SFT, 15k/1k per DPO track), because the files are shared by both modes. The dry run only subsamples them at training time.
""")
code(r"""
stage("1. data")
SIZES = {"sft_train": 30_000, "sft_val": 1_000, "dpo_train": 15_000, "dpo_val": 1_000}
data_stats = D.prepare_all(ROOT, SIZES)
print(json.dumps(data_stats, indent=1))
assert data_stats["persona_eval_overlap_with_train"] == 0

ALPACA = read_json(ROOT / "eval_sets" / "alpaca_eval_300_seed42.json")
HH_EVAL = read_json(ROOT / "eval_sets" / "hh_rlhf_eval_300_seed42.json")
PERSONA_EVAL = read_json(ROOT / "eval_sets" / "persona_style_eval.json")
if DRY_RUN:
    ALPACA, HH_EVAL, PERSONA_EVAL = ALPACA[:4], HH_EVAL[::75], PERSONA_EVAL[::10]
print(f"eval sets: alpaca={len(ALPACA)} hh={len(HH_EVAL)} persona={len(PERSONA_EVAL)}")
""")
code(r"""
# Token-length statistics on 2,000 rows per file: evidence for the 512 / 256 truncation choice in the report.
from transformers import AutoTokenizer
from alignment.formatting import format_prompt
_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
def _pct(xs, p): xs = sorted(xs); return xs[int(p * (len(xs) - 1))]
length_rows = []
for name, kind in (("sft_train", "sft"), ("hh_train", "pref"), ("uf_train", "pref")):
    rows = read_jsonl(PROC / f"{name}.jsonl")[:2000]
    p_len = [len(_tok.encode(format_prompt(r["prompt"]))) for r in rows]
    if kind == "sft":
        total = [p + len(_tok.encode(r["response"])) + 1 for p, r in zip(p_len, rows)]
        length_rows.append({"file": name, "prompt_p50": _pct(p_len, .5), "prompt_p95": _pct(p_len, .95),
                            "total_p50": _pct(total, .5), "total_p95": _pct(total, .95),
                            "frac_total_gt_512": sum(t > 512 for t in total) / len(total)})
    else:
        total = [p + max(len(_tok.encode(r["chosen"])), len(_tok.encode(r["rejected"]))) + 1 for p, r in zip(p_len, rows)]
        length_rows.append({"file": name, "prompt_p50": _pct(p_len, .5), "prompt_p95": _pct(p_len, .95),
                            "frac_prompt_gt_256": sum(p > 256 for p in p_len) / len(p_len),
                            "total_p50": _pct(total, .5), "total_p95": _pct(total, .95),
                            "frac_total_gt_512": sum(t > 512 for t in total) / len(total)})
length_df = pd.DataFrame(length_rows)
(TABLES / "token_lengths.md").write_text(report.df_to_markdown(length_df.round(3)))
display(length_df.round(3))
""")

# =============================================================================
md(r"""
## 2. Memory/speed pilot (resource-driven decisions)
Each probe runs a few optimizer steps on **worst-case full-length** random sequences in a subprocess, so an OOM cannot break this kernel. The first configuration that fits within the VRAM budget wins. Candidates are ordered by closeness to the manual recipe:
- **SFT:** 8-bit AdamW is required (fp32 AdamW needs about 8 GB for weights + grads + states alone). Micro-batch tries 4 → 2 → 1, and gradient accumulation keeps the effective batch at 32.
- **DPO**, in order of preference:
  1. online fp32 reference + RMSprop (identical numerics to the policy, so step-0 reward accuracy is exactly 0)
  2. online fp16 reference + RMSprop (the manual's suggestion)
  3. pre-computed fp32 reference + RMSprop
  4. pre-computed fp32 reference + 8-bit AdamW
  5. online fp16 reference + 8-bit AdamW

Each probe runs real optimizer steps with the loss scale starting at 1, so the optimizer state (2 GB for RMSprop) is allocated and counted.

The result is hardware-specific, so it is stored once in `outputs/pilot.json` and shared by the dry and full runs.
""")
code(r"""
stage("2. memory / speed pilot")
PILOT_PATH = ROOT / "outputs" / "pilot.json"
PILOT_PATH.parent.mkdir(exist_ok=True)

def probe(**kw):
    args = [sys.executable, "-m", "alignment.probe"] + [f"--{k}={v}" for k, v in kw.items()]
    res = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    line = next((l for l in res.stdout.splitlines() if l.startswith("PROBE_RESULT ")), None)
    out = json.loads(line[len("PROBE_RESULT "):]) if line else {"ok": False, "error": res.stderr[-500:], **kw}
    print({k: out.get(k) for k in ("kind", "micro_batch", "optimizer", "ref_mode", "ok", "peak_gb", "reserved_gb", "sec_per_micro", "error")})
    return out

def fits(r, margin_gb=1.0):  # margin for the desktop session + CUDA context + fragmentation
    return r["ok"] and r["reserved_gb"] <= r["total_gpu_gb"] - margin_gb

if PILOT_PATH.exists():
    pilot = read_json(PILOT_PATH)
else:
    sft_trials = []
    for mb in (4, 2, 1):
        r = probe(kind="sft", micro_batch=mb, optimizer="adamw8bit", precision=PRECISION)
        sft_trials.append(r)
        if fits(r):
            break
    sft_choice = next(r for r in sft_trials if fits(r))
    dpo_trials, dpo_choice = [], None
    for ref_mode, opt, ref_dtype in (("online", "rmsprop", "fp32"), ("online", "rmsprop", "fp16"), ("precomputed", "rmsprop", "fp32"),
                                     ("precomputed", "adamw8bit", "fp32"), ("online", "adamw8bit", "fp16")):
        r = probe(kind="dpo", micro_batch=1, optimizer=opt, ref_mode=ref_mode, ref_dtype=ref_dtype, precision=PRECISION)
        dpo_trials.append(r)
        if fits(r):
            dpo_choice = r
            break
    assert dpo_choice is not None, "no DPO configuration fits"
    pilot = {"sft": {"micro_batch_size": sft_choice["micro_batch"], "optimizer": "adamw8bit", "trials": sft_trials},
             "dpo": {"optimizer": dpo_choice["optimizer"], "ref_mode": dpo_choice["ref_mode"], "ref_dtype": dpo_choice["ref_dtype"], "trials": dpo_trials},
             "gpu": props.name}
    write_json(PILOT_PATH, pilot)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "trials"} if isinstance(v, dict) else v for k, v in pilot.items()}, indent=1))
""")

# =============================================================================
md(r"""
## 3. Task 3 — Supervised fine-tuning (from-scratch loop)
Guide recipe:
- 30k UltraChat examples, ctx 512, effective batch 32, LR 2e-5, 3% warmup + cosine, weight decay 0.1, clip 1.0, 1 epoch.
- Response-only token-mean loss.
- fp16 autocast with fp32 weights and GradScaler.
- Validation loss every 100 steps; `best/` = lowest validation loss.

The dry run also checks **kill-and-resume**: it simulates a crash after one step, resumes, and compares the result to an uninterrupted run.
""")
code(r"""
stage("3. SFT")
mb = pilot["sft"]["micro_batch_size"]
SFT_CFG = SFTConfig(
    run_dir=str(OUT / "sft"), train_path=str(PROC / "sft_train.jsonl"), val_path=str(PROC / "sft_val.jsonl"),
    model_name=BASE_MODEL, max_length=512, micro_batch_size=mb, grad_accum=32 // mb, lr=2e-5, warmup_ratio=0.03,
    scheduler="cosine", weight_decay=0.1, max_grad_norm=1.0, epochs=1, optimizer=pilot["sft"]["optimizer"],
    precision=PRECISION, eval_every=100, eval_batch_size=8, save_every=25, seed=SEED,
)
if DRY_RUN:
    SFT_CFG = replace(SFT_CFG, micro_batch_size=2, grad_accum=4, max_train=24, max_val=16, eval_every=1, save_every=1)
    reference_run = train_sft(replace(SFT_CFG, run_dir=str(OUT / "sft_uninterrupted")))
    try:
        train_sft(replace(SFT_CFG, stop_after_steps=1))
    except SimulatedCrash as exc:
        print("simulated crash ->", exc)
    free_cuda()

sft_summary = train_sft(SFT_CFG)
free_cuda()
print(json.dumps(sft_summary, indent=1))
SFT_BEST = OUT / "sft" / "best"
assert (SFT_BEST / "config.json").exists()
""")
code(r"""
if DRY_RUN:  # resumed run must reproduce the uninterrupted run (up to GPU non-determinism)
    a = [r for r in read_jsonl(OUT / "sft_uninterrupted" / "metrics.jsonl") if r["type"] == "train"]
    b = [r for r in read_jsonl(OUT / "sft" / "metrics.jsonl") if r["type"] == "train"]
    assert [r["step"] for r in a] == [r["step"] for r in b], "resumed run has different steps"
    diffs = [abs(x["loss"] - y["loss"]) / max(1e-8, abs(x["loss"])) for x, y in zip(a, b)]
    print("relative loss differences per step (uninterrupted vs crash+resume):", [f"{d:.2e}" for d in diffs])
    assert max(diffs) < 1e-2, "resume does not reproduce training"
display(Image(filename=str(plots.plot_sft(OUT / "sft", FIG / "sft_curves"))))
""")

# =============================================================================
md(r"""
## 4. Task 4 — Two from-scratch DPO runs (same code, two datasets)
Both runs start from the **SFT best checkpoint**, and a second frozen copy of it is the reference. Settings: β = 0.1, LR 1e-6 fixed, 1 pair per micro-batch × 32 accumulation, ctx 512 / prompt 256, clip 1.0, 1 epoch.
- Validation at step 0 and every 50 steps: DPO loss, reward accuracy (strict r_w > r_l), and reward margin.
- `best/` = highest validation reward accuracy, with reward margin as the tie-break.
""")
code(r"""
stage("4. DPO (HH-RLHF, UltraFeedback)")
def dpo_cfg(track):
    cfg = DPOConfig(
        run_dir=str(OUT / f"dpo_{track}"), sft_checkpoint=str(SFT_BEST),
        train_path=str(PROC / f"{track}_train.jsonl"), val_path=str(PROC / f"{track}_val.jsonl"),
        max_length=512, max_prompt_length=256, micro_batch_size=1, grad_accum=32, lr=1e-6, warmup_steps=0,
        scheduler="constant", beta=0.1, optimizer=pilot["dpo"]["optimizer"], weight_decay=0.0, max_grad_norm=1.0,
        epochs=1, precision=PRECISION, ref_mode=pilot["dpo"]["ref_mode"], ref_dtype=pilot["dpo"]["ref_dtype"],
        eval_every=50, eval_batch_size=4, save_every=25, seed=SEED,
    )
    if DRY_RUN:
        cfg = replace(cfg, grad_accum=4, max_train=12, max_val=8, eval_every=1, save_every=1)
    return cfg

dpo_summaries = {}
for track in ("hh", "uf"):
    cfg = dpo_cfg(track)
    if DRY_RUN and track == "hh":  # exercise kill-and-resume in the DPO loop too
        try:
            train_dpo(replace(cfg, stop_after_steps=1))
        except SimulatedCrash as exc:
            print("simulated crash ->", exc)
        free_cuda()
    dpo_summaries[track] = train_dpo(cfg)
    free_cuda()
    print(json.dumps(dpo_summaries[track], indent=1))

for track in ("hh", "uf"):
    first_eval = next(r for r in read_jsonl(OUT / f"dpo_{track}" / "metrics.jsonl") if r["type"] == "eval")
    print(f"{track}: step-0 val reward accuracy={first_eval['val_reward_accuracy']:.3f} "
          f"margin={first_eval['val_reward_margin']:.5f} (policy == reference)")
display(Image(filename=str(plots.plot_dpo({"HH-RLHF DPO": OUT / "dpo_hh", "UltraFeedback DPO": OUT / "dpo_uf"}, FIG / "dpo_curves"))))
""")

# =============================================================================
md(r"""
## 4b. Decoding settings (fixed before any comparison is generated)
- **Decoding:** greedy, `max_new_tokens = 512`, the same template and settings for every model in a comparison.
- **Sanity check** (manual Task 6): 5 AlpacaEval prompts through the SFT model, printed for inspection.
- **Anti-repetition rule:** the SFT model generates for 24 AlpacaEval prompts with `repetition_penalty = 1.0`. If more than 15% of outputs loop (they hit the token limit with fewer than 50% distinct 4-grams), every Qwen generation uses `repetition_penalty = 1.1`. The same rule is applied separately to the base SmolLM2 model for the persona pair. The decision and its evidence are saved.
""")
code(r"""
stage("4b. decoding settings")
def loop_rate(outputs):
    def distinct4(text):
        w = text.split()
        grams = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
        return len(set(grams)) / max(1, len(grams))
    return sum(o["hit_max_new_tokens"] and distinct4(o["output"]) < 0.5 for o in outputs) / max(1, len(outputs))

QWEN_GEN = GenerationConfig(max_new_tokens=32 if DRY_RUN else 512, batch_size=4 if DRY_RUN else 16, dtype="fp16", prompt_style="chatml")
SMOL_GEN = replace(QWEN_GEN, dtype="bf16", prompt_style="chat_template")

DECODING_PATH = OUT / "decoding_decision.json"
if DECODING_PATH.exists():
    decoding = read_json(DECODING_PATH)
else:
    probe_prompts = [r["instruction"] for r in read_json(ROOT / "eval_sets" / "alpaca_eval_300_seed42.json")[:24 if not DRY_RUN else 4]]
    sanity_check(str(SFT_BEST), probe_prompts[:5] if not DRY_RUN else probe_prompts[:2], QWEN_GEN)
    sft_out = sanity_check(str(SFT_BEST), probe_prompts, QWEN_GEN)
    smol_out = sanity_check(PERSONA_MODEL, [r["instruction"] for r in PERSONA_EVAL[:4]], SMOL_GEN)
    decoding = {"qwen_loop_rate_rp1.0": loop_rate(sft_out), "smol_loop_rate_rp1.0": loop_rate(smol_out)}
    decoding["qwen_repetition_penalty"] = 1.1 if decoding["qwen_loop_rate_rp1.0"] > 0.15 else 1.0
    decoding["smol_repetition_penalty"] = 1.1 if decoding["smol_loop_rate_rp1.0"] > 0.15 else 1.0
    decoding["rule"] = "rp=1.1 if >15% of probe outputs hit max_new_tokens with <50% distinct 4-grams, else 1.0"
    write_json(DECODING_PATH, decoding)
print(decoding)
QWEN_GEN = replace(QWEN_GEN, repetition_penalty=decoding["qwen_repetition_penalty"])
SMOL_GEN = replace(SMOL_GEN, repetition_penalty=decoding["smol_repetition_penalty"])
""")
# =============================================================================
md(r"""
## 5. Task 5 — "Idiot Sandwich": TRL DPOTrainer + LoRA on SmolLM2-360M-Instruct
The base model with the adapter disabled is the reference. Data: the provided 220 persona pairs, split 200 train / 20 validation (seed 42).

A small sweep over LoRA rank and learning rate backs up the parameter choices.
- **Checkpoint within a run:** lowest validation DPO loss. Validation reward accuracy saturates at 1.0 within the first epoch on 20 held-out pairs, so it cannot rank checkpoints.
- **Configuration:** validation DPO loss alone rewards over-optimization. The lowest-loss runs push the margin far up and produce looping, abusive text. So each configuration generates for the 20 **validation** prompts (with the persona decoding from 4b), and the submitted adapter is the one with the **largest validation reward margin among configurations whose degenerate-output rate is at most 5%**. Degenerate means the output hits the token limit with fewer than 50% distinct 4-grams. Only validation data is used; the persona eval set is never looked at.
""")
code(r"""
stage("5. LoRA-DPO persona")
LORA_BASE = dict(train_path=str(PROC / "persona_train.jsonl"), val_path=str(PROC / "persona_val.jsonl"),
                 model_name=PERSONA_MODEL, beta=0.1, epochs=5, per_device_batch_size=4, grad_accum=4,
                 warmup_ratio=0.1, scheduler="cosine", max_length=512, bf16=True, eval_steps=5, seed=SEED)
SWEEP = [
    dict(name="r16_lr5e-5", lora_r=16, lora_alpha=32, lr=5e-5),
    dict(name="r16_lr1e-4", lora_r=16, lora_alpha=32, lr=1e-4),
    dict(name="r16_lr2e-4", lora_r=16, lora_alpha=32, lr=2e-4),
    dict(name="r16_lr5e-4", lora_r=16, lora_alpha=32, lr=5e-4),
    dict(name="r8_lr2e-4", lora_r=8, lora_alpha=16, lr=2e-4),
    dict(name="r32_lr2e-4", lora_r=32, lora_alpha=64, lr=2e-4),
]
if DRY_RUN:
    SWEEP = SWEEP[:1]
    LORA_BASE.update(epochs=1, eval_steps=2, grad_accum=1)  # 4 optimizer steps so the LR is non-zero after warmup

from alignment.generate import generate_batch, load_for_generation

def degenerate(o):
    w = o["output"].split()
    grams = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
    return o["hit_max_new_tokens"] and len(set(grams)) / max(1, len(grams)) < 0.5

val_prompts = [r["prompt"] for r in read_jsonl(PROC / "persona_val.jsonl")][: 4 if DRY_RUN else None]

def val_generation_stats(run_dir, adapter):
    # Generations for the persona *validation* prompts, cached per run.
    cache = run_dir / "val_generations.json"
    if cache.exists():
        outs = read_json(cache)
    else:
        model, tok = load_for_generation(PERSONA_MODEL, SMOL_GEN.dtype, adapter)
        outs = generate_batch(model, tok, val_prompts, replace(SMOL_GEN, batch_size=len(val_prompts)))
        del model
        free_cuda()
        write_json(cache, outs)
    n = len(outs)
    return {"val_degenerate_rate": sum(degenerate(o) for o in outs) / n,
            "val_hit_max_rate": sum(o["hit_max_new_tokens"] for o in outs) / n,
            "val_exclaims_per_output": sum(o["output"].count("!") for o in outs) / n,
            "val_mean_words": sum(len(o["output"].split()) for o in outs) / n}

sweep_rows = [{"config": "base (no adapter)", **val_generation_stats(OUT / "lora", None)}]
(OUT / "lora").mkdir(parents=True, exist_ok=True)
for spec in SWEEP:
    spec = dict(spec)
    name = spec.pop("name")
    extra = {"max_train": 16, "max_val": 4} if DRY_RUN else {}
    summ = train_lora_dpo(LoraDPOConfig(run_dir=str(OUT / "lora" / name), **LORA_BASE, **spec, **extra))
    free_cuda()
    fe = summ["final_eval_of_best"]
    sweep_rows.append({"config": name, **spec, "eval_reward_acc": fe.get("eval_rewards/accuracies"),
                       "eval_margin": fe.get("eval_rewards/margins"), "eval_loss": fe.get("eval_loss"),
                       "trainable_params": summ["trainable_params"], "steps": summ["global_steps"],
                       **val_generation_stats(OUT / "lora" / name, str(OUT / "lora" / name / "best_adapter"))})
sweep_df = pd.DataFrame(sweep_rows)
display(sweep_df)
fmt = sweep_df.assign(lr=sweep_df["lr"].map(lambda v: "-" if pd.isna(v) else f"{v:.0e}"))
(TABLES / "lora_sweep.md").write_text(report.df_to_markdown(fmt.round(4)))
eligible = sweep_df[(sweep_df["config"] != "base (no adapter)") & (sweep_df["val_degenerate_rate"] <= 0.05)]
if eligible.empty:  # fall back to the least degenerate adapter
    eligible = sweep_df[sweep_df["config"] != "base (no adapter)"].nsmallest(1, "val_degenerate_rate")
chosen = eligible.sort_values("eval_margin", ascending=False).iloc[0]["config"]
LORA_SELECTED = OUT / "lora" / chosen
write_json(OUT / "lora" / "selection.json", {
    "selected": chosen,
    "rule": "max validation reward margin among configs with <=5% degenerate validation generations "
            "(checkpoint within each run: min validation DPO loss)"})
print("selected LoRA config:", LORA_SELECTED.name)
display(Image(filename=str(plots.plot_lora(read_json(LORA_SELECTED / "log_history.json"), FIG / "lora_dpo_curves"))))
""")

# =============================================================================
md(r"""
## 6. Generation — Task 2 baseline + Task 6 evaluation outputs
- **Decoding:** greedy, `max_new_tokens = 512`, the same template and settings for every model in a comparison.
- **Sanity check first** (manual Task 6): 5 prompts through the SFT model, printed for inspection.
- Decoding settings come from section 4b; they were fixed before any comparison was generated.
""")
code(r"""
stage("6. generation")
# Task 6 sanity check on the submitted DPO / LoRA checkpoints with the final decoding settings (2-5 prompts each).
if not (GEN_DIR / "sanity_done").exists():
    sanity_check(str(OUT / "dpo_uf" / "best"), [r["instruction"] for r in ALPACA[:3]], QWEN_GEN)
    sanity_check(str(OUT / "dpo_hh" / "best"), [r["instruction"] for r in HH_EVAL[:3]], QWEN_GEN)
    sanity_check(PERSONA_MODEL, [r["instruction"] for r in PERSONA_EVAL[:3]], SMOL_GEN,
                 adapter_path=str(LORA_SELECTED / "best_adapter"))
    (GEN_DIR / "sanity_done").write_text("ok\n")
""")
code(r"""
GENERATIONS = {
    # file stem: (model path, adapter, eval rows, generator name, decoding)
    "alpaca_eval__qwen2.5-0.5b-base":  (BASE_MODEL, None, ALPACA, "qwen2.5-0.5b-base", QWEN_GEN),   # Task 2 baseline
    "alpaca_eval__qwen2.5-0.5b-sft":   (str(SFT_BEST), None, ALPACA, "qwen2.5-0.5b-sft", QWEN_GEN),
    "alpaca_eval__qwen2.5-0.5b-dpo-uf": (str(OUT / "dpo_uf" / "best"), None, ALPACA, "qwen2.5-0.5b-dpo-uf", QWEN_GEN),
    "hh_eval__qwen2.5-0.5b-base":      (BASE_MODEL, None, HH_EVAL, "qwen2.5-0.5b-base", QWEN_GEN),     # own analysis
    "hh_eval__qwen2.5-0.5b-sft":       (str(SFT_BEST), None, HH_EVAL, "qwen2.5-0.5b-sft", QWEN_GEN),
    "hh_eval__qwen2.5-0.5b-dpo-hh":    (str(OUT / "dpo_hh" / "best"), None, HH_EVAL, "qwen2.5-0.5b-dpo-hh", QWEN_GEN),
    "persona_eval__smollm2-360m-instruct": (PERSONA_MODEL, None, PERSONA_EVAL, "smollm2-360m-instruct", SMOL_GEN),
    "persona_eval__smollm2-360m-instruct-lora-dpo": (PERSONA_MODEL, str(LORA_SELECTED / "best_adapter"), PERSONA_EVAL,
                                                     "smollm2-360m-instruct-lora-dpo", SMOL_GEN),
}
gen_configs = {}
for stem, (path, adapter, rows, name, gcfg) in GENERATIONS.items():
    generate_eval_set(path, rows, GEN_DIR / f"{stem}.json", name, gcfg, adapter_path=adapter)
    free_cuda()
    gen_configs[stem] = {"model": path, "adapter": adapter, "decoding": "greedy", "temperature": "n/a (greedy)",
                         "top_p": "n/a (greedy)", "max_new_tokens": gcfg.max_new_tokens,
                         "repetition_penalty": gcfg.repetition_penalty, "template": gcfg.prompt_style}
gen_stats = pd.DataFrame([
    {"file": stem, "n": len(g := read_json(GEN_DIR / f"{stem}.json")),
     "mean_new_tokens": sum(x["new_tokens"] for x in g) / len(g),
     "frac_hit_max_new_tokens": sum(x["hit_max_new_tokens"] for x in g) / len(g)}
    for stem in GENERATIONS])
display(gen_stats.round(3))
(TABLES / "generation_stats.md").write_text(report.df_to_markdown(gen_stats.round(3)))
""")

# =============================================================================
md(r"""
## 7. Blinded pairwise files (written before judging)
The A/B order is randomized per pair with seed 42. Model identities are stored only in `hidden_metadata`, which the judge never sees.
""")
code(r"""
stage("7. pairwise files")
COMPARISONS = {
    "uf_dpo_vs_sft": dict(treatment="alpaca_eval__qwen2.5-0.5b-dpo-uf", baseline="alpaca_eval__qwen2.5-0.5b-sft",
                          eval_rows=ALPACA, rubric="general"),
    "hh_dpo_vs_sft": dict(treatment="hh_eval__qwen2.5-0.5b-dpo-hh", baseline="hh_eval__qwen2.5-0.5b-sft",
                          eval_rows=HH_EVAL, rubric="safety"),
    "persona_lora_vs_base": dict(treatment="persona_eval__smollm2-360m-instruct-lora-dpo",
                                 baseline="persona_eval__smollm2-360m-instruct", eval_rows=PERSONA_EVAL, rubric="persona"),
}
for name, c in COMPARISONS.items():
    pairs = build_pairs(GEN_DIR / f"{c['treatment']}.json", GEN_DIR / f"{c['baseline']}.json",
                        EVAL_DIR / f"{name}_pairs.jsonl", c["treatment"].split("__")[1], c["baseline"].split("__")[1],
                        name, seed=SEED, eval_rows=c["eval_rows"])
    pos = [p["hidden_metadata"]["treatment_position"] for p in pairs]
    print(f"{name}: {len(pairs)} pairs, treatment shown as A in {pos.count('A')}/{len(pos)}")

# Bundle for the Kaggle judging notebook (attach as a Kaggle dataset, or commit to the repo).
bundle = OUT / "kaggle_judge_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for p in EVAL_DIR.glob("*_pairs.jsonl"):
        z.write(p, f"eval/{p.name}")
    for p in list((ROOT / "alignment").glob("*.py")) + [ROOT / "scripts" / "judge_pairwise_kaggle.py"]:
        z.write(p, str(p.relative_to(ROOT)))
print("bundle:", bundle)
""")

# =============================================================================
md(r"""
## 8. Task 7 — Judging
- **Dry run:** the offline `mock` judge exercises the full CLI, retry, resume and summary path.
- **Full run:** judge with `notebooks/kaggle_setup_and_judge.ipynb` on Kaggle. It uses `kaggle_benchmarks` with `google/gemini-3.6-flash`, 5 pairs per request, ≥5 s sleep, 3 retries, and resumable output. Copy the `*_judged.jsonl` files back into `outputs/eval/` and re-run this notebook; every finished stage is skipped. Alternatively, set `RUN_JUDGE_LOCALLY = True` after `kaggle b init -y`.
""")
code(r"""
stage("8. judging")
def judge(name, model, sleep):
    c = COMPARISONS[name]
    cmd = [sys.executable, "scripts/judge_pairwise_kaggle.py", "--pairs_path", str(EVAL_DIR / f"{name}_pairs.jsonl"),
           "--output_path", str(EVAL_DIR / f"{name}_judged.jsonl"), "--judge_model", model, "--rubric", c["rubric"],
           "--batch_size", "5", "--sleep_seconds", str(sleep), "--max_retries", "3", "--reuse_judgments"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(res.stdout[-2500:], res.stderr[-1500:])
    assert res.returncode == 0

if DRY_RUN:
    for name in COMPARISONS:
        judge(name, "mock", 0)
        judge(name, "mock", 0)  # second call must reuse every judgment (resume path)
elif RUN_JUDGE_LOCALLY:
    for name in COMPARISONS:
        judge(name, JUDGE_MODEL, 5)
else:
    missing = [n for n in COMPARISONS if not (EVAL_DIR / f"{n}_judged.jsonl").exists()]
    print("Judged files missing (run the Kaggle judging notebook, then copy results here):", missing or "none")
""")

# =============================================================================
md("## 9. Results and report tables")
code(r"""
stage("9. results")
config_df, gen_df = report.experiment_config_table(OUT, LORA_SELECTED, gen_configs)
display(config_df.T)
(TABLES / "experiment_config.md").write_text(report.df_to_markdown(config_df) + "\n\n" + report.df_to_markdown(gen_df))

judged_specs = {}
for name, c in COMPARISONS.items():
    crit = ["style_winner", "quality_winner"] if c["rubric"] == "persona" else ["winner"]
    for cr in crit:
        judged_specs[f"{name} [{cr}]"] = {"pairs": EVAL_DIR / f"{name}_pairs.jsonl",
                                          "judged": EVAL_DIR / f"{name}_judged.jsonl",
                                          "failures": EVAL_DIR / f"{name}_judged_failures.jsonl", "criterion": cr}
results_df = report.results_table(EVAL_DIR, judged_specs)
if results_df.empty:
    print("no judgments yet")
else:
    display(results_df)
    (TABLES / "results.md").write_text(report.df_to_markdown(results_df))
    for name, spec in judged_specs.items():
        if not Path(spec["judged"]).exists():
            continue
        cat = report.per_category_table(spec["pairs"], spec["judged"], spec["failures"], spec["criterion"])
        safe = name.replace(" ", "_").replace("[", "").replace("]", "")
        (TABLES / f"per_category__{safe}.md").write_text(report.df_to_markdown(cat))
        (TABLES / f"examples__{safe}.md").write_text(report.pick_examples(spec["pairs"], spec["judged"], spec["criterion"]))
        print(f"\n{name}"); display(cat)

training_df = pd.DataFrame([
    {"run": "SFT", **{k: v for k, v in read_json(OUT / "sft" / "summary.json").items() if not isinstance(v, (dict, list))}},
    *[{"run": f"DPO-{t.upper()}", **{k: v for k, v in read_json(OUT / f"dpo_{t}" / "summary.json").items()
                                       if not isinstance(v, (dict, list))}} for t in ("hh", "uf")],
])
display(training_df)
(TABLES / "training_summary.md").write_text(report.df_to_markdown(training_df))
stage("PIPELINE COMPLETE")
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
print("tables :", sorted(p.name for p in TABLES.glob("*.md")))
""")


def build(path: Path) -> None:
    cells = []
    for kind, src, tags in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {"tags": tags} if tags else {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "ML (.venv)", "language": "python", "name": "ml"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "notebooks" / "PA2_local_pipeline.ipynb")
