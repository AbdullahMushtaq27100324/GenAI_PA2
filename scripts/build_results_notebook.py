"""Generate and execute notebooks/results_summary.ipynb: a readable summary of every result.

It only reads saved artifacts under outputs/ (no training, no GPU), so it runs in a few seconds:
    python scripts/build_results_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_notebook as nbb  # noqa: E402

nbb.CELLS.clear()
md, code = nbb.md, nbb.code

# =============================================================================
md(r"""
# Results summary — SFT + DPO on Qwen2.5-0.5B, and LoRA persona DPO

This notebook only **reads saved results** from `outputs/`; it doesn't train anything. Every number here comes from files in the repo.

## TL;DR

| Experiment | Compared on | Result (judge: `google/gemini-3.6-flash`) |
|---|---|---|
| **HH-RLHF DPO vs SFT** | 300 helpful/harmless prompts | **DPO wins clearly**: 72% of decided pairs (95% CI 66–77%) |
| **UltraFeedback DPO vs SFT** | 300 AlpacaEval prompts | **No significant gain; slightly behind**: 46% (CI 39–52%). DPO became wordier and lost on long answers |
| **LoRA persona DPO vs base SmolLM2** | 40 style prompts | **Much more in persona** (95% of decided style judgments) but **less useful** (39% on quality) |

What we learned, in one line each:
- **DPO does what the preference data rewards.** HH data rewards safe, deflecting answers, and the model learned exactly that: its biggest wins are on cyber-abuse, privacy and harmful-instruction prompts.
- **UltraFeedback's preferred answers are longer**, so DPO learned to write more, and the judge didn't reward the extra length.
- **Validation metrics alone can be misleading.** The persona run with the "best" validation loss produced looping, abusive text; we had to select checkpoints using validation *generations*.
""")

code(r"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
OUT, EVAL, GEN, TAB = ROOT / "outputs", ROOT / "outputs" / "eval", ROOT / "outputs" / "generations", ROOT / "outputs" / "report_tables"
pd.set_option("display.max_colwidth", 120)

def rj(p): return json.loads(Path(p).read_text())
def rjl(p): return [json.loads(l) for l in Path(p).read_text().split("\n") if l.strip()]  # not splitlines(): outputs contain U+2028

BLUE, ORANGE, GRAY, INK2 = "#2a78d6", "#eb6834", "#b8b6ad", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "font.size": 10, "legend.frameon": False})

def outcome(judgment, pair, key="winner"):
    v, pos = judgment[key], pair["hidden_metadata"]["treatment_position"]
    return "tie" if v == "tie" else ("win" if v == pos else "loss" if v in ("A", "B") else "unknown")
""")

# =============================================================================
md(r"""
## 1. What we trained, on what

**Pipeline:** `Qwen2.5-0.5B` (base) → **SFT** on 30k UltraChat instruction/response pairs → two separate **DPO** runs from the SFT model:
- **HH-RLHF:** 15k single-turn helpful/harmless pairs.
- **UltraFeedback:** 15k general-quality pairs.

Separately, **LoRA-DPO** with TRL on `SmolLM2-360M-Instruct`, using 200 "Gordon Ramsay persona" pairs.

The SFT and both DPO runs use our own training loops (no HF `Trainer`, no TRL), as the manual requires. All subsets use seed 42, and their exact IDs are saved in `data/processed/*.ids.json`.
""")
code(r"""
s = rj(ROOT / "data/processed/all.stats.json")
display(pd.DataFrame([
    {"track": "SFT (UltraChat 200k, first turn)", "train": s["sft"]["train"], "val": s["sft"]["val"], "pool": s["sft"]["pool_train"]},
    {"track": "DPO HH-RLHF (single-turn, 50/50 harmless/helpful)", "train": s["hh"]["train"], "val": s["hh"]["val"], "pool": sum(s["hh"]["pool_train"].values())},
    {"track": "DPO UltraFeedback (binarized, score_chosen > score_rejected)", "train": s["uf"]["train"], "val": s["uf"]["val"], "pool": s["uf"]["pool_train"]},
    {"track": "LoRA persona (provided Ramsay pairs)", "train": s["persona"]["train"], "val": s["persona"]["val"], "pool": s["persona"]["total"]},
]).set_index("track"))
display(Markdown("**Token lengths** (2,000-row sample; prompt template included). This is why we used a max length of 512 and a max prompt length of 256:"))
display(Markdown((TAB / "token_lengths.md").read_text()))
""")
md(r"""
The preference datasets differ in one important way: **how long the preferred answer is compared with the rejected one**. DPO pushes the model toward whatever separates chosen from rejected answers, so this difference matters later.
""")
code(r"""
rows = []
for name in ("hh", "uf"):
    data = rjl(ROOT / f"data/processed/{name}_train.jsonl")
    c = [len(r["chosen"].split()) for r in data]; rj_ = [len(r["rejected"].split()) for r in data]
    rows.append({"dataset": {"hh": "HH-RLHF", "uf": "UltraFeedback"}[name], "mean words (chosen)": round(sum(c) / len(c), 1),
                 "mean words (rejected)": round(sum(rj_) / len(rj_), 1),
                 "chosen longer than rejected": f"{sum(a > b for a, b in zip(c, rj_)) / len(c):.0%}"})
display(pd.DataFrame(rows).set_index("dataset"))
""")
md(r"""
- **UltraFeedback:** preferred answers average about 27 words longer, and are the longer one in 55% of pairs. That's a length preference DPO can latch onto.
- **HH-RLHF:** answers are short overall and chosen ≈ rejected in length (51%). The preference is about *content* (comply vs decline), not length.
""")

# =============================================================================
md(r"""
## 2. SFT: teaching the base model to answer instructions

Settings: effective batch 32, LR 2e-5 with 3% warmup and cosine decay, 1 epoch (938 steps), loss on response tokens only, fp16 autocast with fp32 weights.
""")
code(r"""
display(Image(filename=str(OUT / "figures/sft_curves.png")))
ev = [r for r in rjl(OUT / "sft/metrics.jsonl") if r["type"] == "eval"]
display(pd.DataFrame(ev)[["step", "val_loss", "val_ppl"]].round(4).set_index("step").T)
""")
md(r"""
- **Validation loss only drops from 1.400 to 1.356.** Qwen2.5 "base" already saw a lot of chat-style data in pretraining, so the loss starts low. It rises slightly for the first ~200 steps while the model adapts to our ChatML format, then falls steadily.
- **The best checkpoint is the final one** (still improving), so one epoch is, if anything, slightly under-trained. That's normal for this recipe.
- **The real gain shows in behaviour, not loss.** The base model rambles to the 512-token limit on 66% of AlpacaEval prompts; after SFT that falls to 15% (section 4).
""")

# =============================================================================
md(r"""
## 3. DPO training: two datasets, same code

Both runs start from the SFT best checkpoint. A frozen copy of that checkpoint is the reference. Settings: β = 0.1, LR 1e-6 (constant), effective batch 32, 1 epoch (469 steps). The best checkpoint is the one with the highest **validation reward accuracy**: the fraction of held-out pairs where the model's implicit reward for the chosen answer beats the rejected one.
""")
code(r"""
display(Image(filename=str(OUT / "figures/dpo_curves.png")))
rows = []
for t, name in (("hh", "HH-RLHF DPO"), ("uf", "UltraFeedback DPO")):
    s = rj(OUT / f"dpo_{t}/summary.json")
    ev = [r for r in rjl(OUT / f"dpo_{t}/metrics.jsonl") if r["type"] == "eval"]
    rows.append({"run": name, "best step": s["best_step"], "val reward acc (best)": round(s["best_val_reward_accuracy"], 3),
                 "val margin (best)": round(s["best_val_reward_margin"], 3), "val DPO loss (end)": round(ev[-1]["val_loss"], 4),
                 "train hours": s["train_hours"]})
display(pd.DataFrame(rows).set_index("run"))
""")
md(r"""
**Reading the curves:**
- **UltraFeedback is easier to learn.** Validation reward accuracy jumps to about 0.70 within 50 steps and plateaus at about 0.72. HH climbs slowly to 0.65 and is still rising at the end, so HH would probably benefit from more training.
- **The reward margin grows mostly because the rejected answers' likelihood falls** (dotted lines in the bottom-right plot), not because the chosen answers' likelihood rises. This is the typical DPO signature: it pushes probability *away* from bad answers.
- **Step-0 accuracy is exactly 0**, as the manual predicts: policy and reference are identical, every reward pair ties, and the comparison is strict. A first run logged 0.25/0.39 here because the per-sequence log-prob sum used CUDA atomic adds, whose order varies, so the same model scored slightly differently on each pass. We made the summation deterministic (with a regression test) and re-ran both DPO runs. Final accuracies barely moved (HH 0.646 → 0.647, UltraFeedback 0.728 → 0.726).
- **Neither validation curve gets worse late in training**, so there's no sign of over-fitting within one epoch.
""")

# =============================================================================
md(r"""
## 4. What the models actually generate

All models use greedy decoding with at most 512 new tokens, and each comparison uses identical settings. The anti-repetition rule (decided on the SFT model before any comparison) kept Qwen at repetition penalty 1.0. SmolLM2 loops more on its own (25% of probe outputs), so the persona pair uses 1.1.
""")
code(r"""
display(Markdown((TAB / "generation_stats.md").read_text()))
""")
md(r"""
- **Base → SFT:** the base model often never stops (it hit the token limit on 66% of AlpacaEval prompts). SFT taught it to finish its answer (15%).
- **SFT → UltraFeedback DPO:** answers get **37% longer** (215 → 294 tokens) and hit the limit more often (15% → 20%). This is UltraFeedback's length preference showing up.
- **SFT → HH DPO:** answers get **shorter** (183 → 160 tokens). HH-RLHF responses are short overall (about 35 words, against about 200 in UltraFeedback), and DPO moved the model toward that register. Chosen and rejected HH answers are nearly the same length, so this isn't a learned "shorter is preferred" signal.
""")

# =============================================================================
md(r"""
## 5. Head-to-head judgments (Task 7)

Each comparison puts the DPO model against SFT on the same prompts with the same decoding. Model identities were hidden and the A/B order randomized; Gemini judged 5 pairs per request.

| Comparison | Judge rubric |
|---|---|
| UltraFeedback DPO vs SFT | general: helpfulness, correctness, instruction-following, concision, no hallucination |
| HH DPO vs SFT | safety-aware: also rewards appropriate refusals, privacy protection, safe redirection, and penalizes over-refusal |
| Persona LoRA vs base | two separate verdicts: **style** (how Ramsay-like) and **quality** (helpful, correct, safe, ignoring style) |

All 640 pairs were judged with **0 failed requests and 0 retries**. One UltraFeedback verdict was invalid and is counted as *unknown*. **Judge consistency:** the 40 persona pairs were byte-identical in two separate Kaggle judging runs, and the judge agreed with itself on **95%** of them, for both style and quality.
""")
code(r"""
import sys; sys.path.insert(0, str(ROOT))
from alignment.report import results_table
specs = {}
for name, crits in (("uf_dpo_vs_sft", ["winner"]), ("hh_dpo_vs_sft", ["winner"]), ("persona_lora_vs_base", ["style_winner", "quality_winner"])):
    for cr in crits:
        specs[f"{name} [{cr}]"] = {"pairs": EVAL / f"{name}_pairs.jsonl", "judged": EVAL / f"{name}_judged.jsonl",
                                   "failures": EVAL / f"{name}_judged_failures.jsonl", "criterion": cr}
res = results_table(EVAL, specs)
show = res[["comparison", "n", "wins", "ties", "losses", "win_rate_excl_ties", "ci95", "win_rate_ties_half"]]
display(show.set_index("comparison"))

fig, ax = plt.subplots(figsize=(9, 3.3))
ax.set_axisbelow(True)
labels = ["UltraFeedback DPO vs SFT", "HH DPO vs SFT", "Persona LoRA vs base: style", "Persona LoRA vs base: quality"]
for i, (_, r) in enumerate(res.iterrows()):
    n = r["n"]; w, t, l = r["wins"] / n, r["ties"] / n, r["losses"] / n
    ax.barh(i, w, color=BLUE, edgecolor="white", linewidth=2)
    ax.barh(i, t, left=w, color=GRAY, edgecolor="white", linewidth=2)
    ax.barh(i, l, left=w + t, color=ORANGE, edgecolor="white", linewidth=2)
    ax.text(w / 2, i, f"{r['wins']}", ha="center", va="center", color="white", fontsize=9)
    ax.text(w + t + l / 2, i, f"{r['losses']}", ha="center", va="center", color="white", fontsize=9)
ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("share of pairs"); ax.set_title("Treatment model (DPO / LoRA) wins, ties, losses")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GRAY, ORANGE)],
          labels=["treatment wins", "tie", "treatment loses"], loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
plt.tight_layout(); plt.show()
""")
md(r"""
### 5a. HH-RLHF DPO: a clear safety improvement, concentrated where you'd expect
""")
code(r"""
pairs = {p["pair_id"]: p for p in rjl(EVAL / "hh_dpo_vs_sft_pairs.jsonl")}
cat = pd.DataFrame([{"category": pairs[j["pair_id"]]["category"], "o": outcome(j, pairs[j["pair_id"]])}
                    for j in rjl(EVAL / "hh_dpo_vs_sft_judged.jsonl")])
tab = cat.groupby("category")["o"].value_counts().unstack(fill_value=0)
tab["win rate (excl. ties)"] = (tab["win"] / (tab["win"] + tab["loss"])).round(3)
tab = tab.sort_values("win rate (excl. ties)")
display(tab)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.set_axisbelow(True)
ax.barh(tab.index, tab["win rate (excl. ties)"], color=BLUE)
ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
for y, v in enumerate(tab["win rate (excl. ties)"]):
    ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlim(0, 1.05); ax.set_xlabel("HH DPO win rate vs SFT (ties excluded), 30 prompts per category")
plt.tight_layout(); plt.show()
""")
md(r"""
- **The biggest wins are exactly where HH-RLHF data teaches refusal:** harmful instructions (0.96), privacy/doxxing (0.91), cyber abuse (0.88), legal/financial (0.83), self-harm support (0.70).
- **No alignment tax on benign requests:** DPO also wins on plain helpfulness (0.68), so it didn't become a blanket refuser. Its one weak spot is **calibration and uncertainty (0.42)**, where shorter, more assertive answers lose to SFT's hedged ones. Medical safety is about a coin flip (0.57).
- **When DPO's answer was shorter than SFT's, DPO won 109 to 21** (section 5c). Short, decisive deflections beat SFT's longer, sometimes complying answers.
""")
md(r"""
### 5b. UltraFeedback DPO: no measurable gain on AlpacaEval
""")
code(r"""
pairs = {p["pair_id"]: p for p in rjl(EVAL / "uf_dpo_vs_sft_pairs.jsonl")}
cat = pd.DataFrame([{"AlpacaEval subset": pairs[j["pair_id"]]["category"], "o": outcome(j, pairs[j["pair_id"]])}
                    for j in rjl(EVAL / "uf_dpo_vs_sft_judged.jsonl")])
tab = cat.groupby("AlpacaEval subset")["o"].value_counts().unstack(fill_value=0)
tab["win rate (excl. ties)"] = (tab["win"] / (tab["win"] + tab["loss"])).round(3)
display(tab.sort_values("win rate (excl. ties)"))
""")
md(r"""
- **Overall 103 wins, 73 ties, 123 losses** (1 unknown). The 95% interval [0.39, 0.52] still contains 0.5, so UltraFeedback DPO is **not significantly different from SFT**; if anything it's slightly behind.
- **It varies by subset:** around even on `oasst` (0.57) and `vicuna` (0.53), worse on `helpful_base` (0.35) and `koala` (0.40). With only 40–95 prompts per subset, these are hints, not conclusions.
- **Why no gain, despite the healthy training metrics** (validation reward accuracy 0.73)? The next table points to **length**.
""")
md(r"""
### 5c. Did length decide the outcome?
""")
code(r"""
rows = []
for name, label in (("uf_dpo_vs_sft", "UltraFeedback DPO vs SFT"), ("hh_dpo_vs_sft", "HH DPO vs SFT")):
    pairs = {p["pair_id"]: p for p in rjl(EVAL / f"{name}_pairs.jsonl")}
    for j in rjl(EVAL / f"{name}_judged.jsonl"):
        p = pairs[j["pair_id"]]; pos = p["hidden_metadata"]["treatment_position"]
        t, b = (p["response_a"], p["response_b"]) if pos == "A" else (p["response_b"], p["response_a"])
        lt, lb = len(t.split()), len(b.split())
        bucket = "DPO >10% longer" if lt > 1.1 * lb else ("DPO >10% shorter" if lt < 0.9 * lb else "similar length")
        rows.append({"comparison": label, "length": bucket, "o": outcome(j, p), "judge picked": j["winner"]})
df = pd.DataFrame(rows)
tab = df.groupby(["comparison", "length"])["o"].value_counts().unstack(fill_value=0)
tab["win rate (excl. ties)"] = (tab["win"] / (tab["win"] + tab["loss"])).round(3)
display(tab)
display(Markdown("**Position check** (how often the judge picked the response shown first vs second):"))
display(df.groupby("comparison")["judge picked"].value_counts().unstack(fill_value=0))
""")
md(r"""
- **The judge did *not* simply prefer longer answers.** For UltraFeedback, when DPO's answer was much longer, DPO **lost** 64–101. When it was shorter, it **won** 19–6. So UltraFeedback DPO picked up the dataset's length preference (chosen answers are longer in UltraFeedback), and the judge penalized the padding and the truncated rambles.
- **There's a mild position bias toward Response A** (UltraFeedback: 127 A vs 99 B; HH: 152 vs 113). Randomizing the A/B order per pair spreads this bias evenly across both models, so it doesn't favour either.
""")

# =============================================================================
md(r"""
## 6. LoRA persona DPO ("Idiot Sandwich")

TRL `DPOTrainer` + PEFT LoRA on `SmolLM2-360M-Instruct`: all attention and MLP projections, β = 0.1, effective batch 16, 5 epochs, bf16, 200 train / 20 validation pairs. We swept LoRA rank and learning rate.
""")
code(r"""
sw = rj(OUT / "lora/selection.json"); print("selected:", sw["selected"], "|", sw["rule"])
display(Markdown((TAB / "lora_sweep.md").read_text()))
display(Image(filename=str(OUT / "figures/lora_dpo_curves.png")))
""")
md(r"""
**This sweep is the most instructive result of the whole assignment.**
- **Validation metrics saturate immediately.** Reward accuracy hits 1.0 for every configuration (the persona pairs are trivially separable), and validation DPO loss keeps falling the harder you push.
- **The lowest-loss runs are broken.** Selecting by lowest validation loss would pick lr 5e-4, whose generations loop and turn abusive on 95–100% of prompts (classic **DPO over-optimization**: the model maximizes the chosen/rejected gap in ways that ruin fluent generation).
- **So we selected on validation generations instead:** the strongest persona (highest validation margin) among configurations whose 20 validation generations are at most 5% degenerate. That picks **r=16, lr 1e-4**: it uses about 14× more exclamations than the base model, with 0% degeneration.
""")
code(r"""
pairs = {p["pair_id"]: p for p in rjl(EVAL / "persona_lora_vs_base_pairs.jsonl")}
J = rjl(EVAL / "persona_lora_vs_base_judged.jsonl")
rows = [{"category": pairs[j["pair_id"]]["category"], "style": outcome(j, pairs[j["pair_id"]], "style_winner"),
         "quality": outcome(j, pairs[j["pair_id"]], "quality_winner")} for j in J]
df = pd.DataFrame(rows)
out = pd.concat({k: df.groupby("category")[k].value_counts().unstack(fill_value=0) for k in ("style", "quality")}, axis=1)
display(out)
""")
md(r"""
- **Style:** the LoRA model is judged more Ramsay-like on 18 prompts, with 21 ties and only 1 loss. The ties are mostly cases where both answers are fairly neutral.
- **Quality:** it loses 14–22 (4 ties). It is **less useful** on cooking, coding and general help (the persona adds theatrics and sometimes worse advice); see the per-category table for safety-sensitive prompts.
- **Safety:** it isn't safer. Like the base model, it still answers the chloroform prompt. The persona data teaches tone, not refusal.
- **Answer to the task's question:** yes, LoRA-DPO made the model more stylistically faithful, **but** at a measurable cost to usefulness. With 200 pairs there's no free lunch between style strength and quality; the learning-rate sweep shows the dial.
""")

# =============================================================================
md("## 7. Example outputs")
code(r"""
def show_examples(name, key="winner", picks=(("win", 2), ("loss", 1)), category=None, max_chars=600):
    pairs = {p["pair_id"]: p for p in rjl(EVAL / f"{name}_pairs.jsonl")}
    judged = rjl(EVAL / f"{name}_judged.jsonl")
    for want, k in picks:
        cands = []
        for j in judged:
            p = pairs[j["pair_id"]]
            if outcome(j, p, key) != want or (category and p["category"] not in category):
                continue
            pos = p["hidden_metadata"]["treatment_position"]
            t, b = (p["response_a"], p["response_b"]) if pos == "A" else (p["response_b"], p["response_a"])
            cands.append((len(t) + len(b), p, t, b, j.get("reason", "")))
        for _, p, t, b, reason in sorted(cands, key=lambda c: c[0])[:k]:
            m = p["hidden_metadata"]
            display(Markdown(
                f"**{want.upper()} for {m['treatment']}** · `{p['category']}`\n\n> **Prompt:** {p['instruction'][:max_chars]}\n\n"
                f"**{m['treatment']}:**\n```\n{t[:max_chars]}\n```\n**{m['baseline']}:**\n```\n{b[:max_chars]}\n```\n"
                f"*Judge:* {reason}\n\n---"))

display(Markdown("### HH DPO vs SFT"))
show_examples("hh_dpo_vs_sft", picks=(("win", 2), ("loss", 1)), category={"harmful_instructions", "cyber_abuse", "privacy_and_doxxing", "benign_helpfulness"})
""")
code(r"""
display(Markdown("### UltraFeedback DPO vs SFT"))
show_examples("uf_dpo_vs_sft", picks=(("win", 1), ("loss", 2)))
""")
code(r"""
display(Markdown("### Persona LoRA vs base (style win, quality loss)"))
show_examples("persona_lora_vs_base", key="quality_winner", picks=(("loss", 2), ("win", 1)))
""")

# =============================================================================
md(r"""
## 8. Deviations from the manual's guide configuration

| What | Guide | Ours | Why (type) |
|---|---|---|---|
| Hardware | Kaggle 2× T4 (15 GB each) | Local RTX 3050 (8 GB); Kaggle used for Task 1 + judging | Ran overnight locally (resource) |
| SFT optimizer | AdamW | **8-bit AdamW** (bitsandbytes) | fp32 AdamW needs ~8 GB for weights + grads + states alone (resource) |
| SFT micro-batch × accumulation | 2 × 16 | 4 × 8 (effective 32 unchanged) | Measured to fit (resource) |
| DPO optimizer | RMSprop or AdamW | **8-bit AdamW** | Measured: RMSprop runs out of memory in every reference setup (resource) |
| DPO reference | frozen copy, run every step | frozen fp32 copy, **scores all pairs once before training** | Mathematically identical, frees 2 GB (resource) |
| Gradient checkpointing | – | on | Memory (resource) |
| Persona LoRA selection | – | validation-generation degeneracy filter + max margin | Validation loss selected over-optimized, looping models (performance) |
| SmolLM2 decoding | greedy | greedy + repetition penalty 1.1 | Base SmolLM2 looped on 25% of probe prompts (decided before comparison) |

All memory decisions are backed by measured probes in `outputs/pilot.json`.

## 9. Limitations
- **One judge model, one seed, 300 prompts per set.** Confidence intervals are about ±6 points, so small effects (like UltraFeedback's) can't be resolved. The judge also isn't perfectly repeatable: it agreed with itself on 95% of identical persona pairs.
- **The judge has a mild preference for Response A** (randomized order spreads it evenly) and its own style preferences.
- **HH DPO was still improving at the end of its single epoch.** More steps or a larger β sweep could change the size of the gain.
- **The persona evaluation uses only 40 prompts**, and validation selection used 20.
- **The refusal behaviour of these 0.5B models is crude:** they often deflect with unrelated tips instead of explaining the refusal.

**Where everything lives:** configs and metrics in `outputs/<run>/`, generations in `outputs/generations/`, pairs and judgments in `outputs/eval/`, tables in `outputs/report_tables/`, figures in `outputs/figures/`. The full pipeline is in `notebooks/PA2_local_pipeline.ipynb`; the Kaggle runs are in `notebooks/kaggle_*_executed.ipynb`.
""")


def execute(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"executed {path}")


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "notebooks" / "results_summary.ipynb"
    nbb.build(target, kernel=("python3", "Python 3"))
    execute(target)
