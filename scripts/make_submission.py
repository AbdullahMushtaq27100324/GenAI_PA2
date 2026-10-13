"""Assemble submission/ from the committed artifacts (manual Section 16 checklist).

    python scripts/make_submission.py

submission/
  PA2_report.pdf                 final report (copy of report/PA2_report.pdf)
  training_plots/                SFT, DPO and LoRA-DPO training curves (+ results overview)
  evaluation_summary.md          DPO-vs-SFT (and persona) judged results, built from outputs/eval/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment import report  # noqa: E402

OUT = ROOT / "submission"
EVAL = ROOT / "outputs" / "eval"
FIG = ROOT / "outputs" / "figures"


def specs():
    s = {}
    for name, label, crits in (("uf_dpo_vs_sft", "UltraFeedback DPO vs SFT (AlpacaEval, 300 prompts)", ["winner"]),
                               ("hh_dpo_vs_sft", "HH-RLHF DPO vs SFT (HH eval set, 300 prompts)", ["winner"]),
                               ("persona_lora_vs_base", "Persona LoRA-DPO vs base SmolLM2 (40 prompts)",
                                ["style_winner", "quality_winner"])):
        for cr in crits:
            key = label if cr == "winner" else f"{label}: {cr.split('_')[0]}"
            s[key] = {"pairs": EVAL / f"{name}_pairs.jsonl", "judged": EVAL / f"{name}_judged.jsonl",
                      "failures": EVAL / f"{name}_judged_failures.jsonl", "criterion": cr}
    return s


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "training_plots").mkdir(parents=True)
    shutil.copy(ROOT / "report" / "PA2_report.pdf", OUT / "PA2_report.pdf")
    for name in ("sft_curves", "dpo_curves", "lora_dpo_curves", "results_overview"):
        shutil.copy(FIG / f"{name}.png", OUT / "training_plots" / f"{name}.png")

    sp = specs()
    df = report.results_table(EVAL, sp)
    cols = ["comparison", "n", "wins", "ties", "losses", "unknown", "failures", "win_rate_excl_ties", "ci95",
            "win_rate_ties_half"]
    lines = [
        "# PA2 evaluation summary: DPO vs SFT (and persona LoRA vs base)",
        "",
        "- **Judge:** `google/gemini-3.6-flash` via `kaggle_benchmarks` (Kaggle Benchmarks task notebook), temperature 0.",
        "- **Blinding:** A/B order randomized per pair (seed 42); model identities hidden from the judge.",
        "- **Reliability:** 5 pairs per request, ≥ 5 s sleep before every attempt, up to 3 attempts, checkpointed after every batch, failures logged.",
        "- **Same prompts and decoding** for both models in a comparison: greedy, at most 512 new tokens. Repetition penalty 1.0 for the Qwen models, 1.1 for the SmolLM2 pair.",
        "- **Win rate (ties excluded)** = W/(W+L) with a Wilson 95% CI. **Win rate (ties = ½)** = (W + ½T)/(W+T+L). *Unknown* (invalid verdict) is excluded from both.",
        "- **Run outcome:** all 640 pairs judged; 0 failed requests; 0 retries; 1 unknown.",
        "",
        "## Overall",
        "",
        report.df_to_markdown(df[cols]),
        "",
        "## HH-RLHF DPO vs SFT by category",
        "",
        report.df_to_markdown(report.per_category_table(**{k: v for k, v in sp["HH-RLHF DPO vs SFT (HH eval set, 300 prompts)"].items()})),
        "",
        "## UltraFeedback DPO vs SFT by AlpacaEval subset",
        "",
        report.df_to_markdown(report.per_category_table(**{k: v for k, v in sp["UltraFeedback DPO vs SFT (AlpacaEval, 300 prompts)"].items()})),
        "",
        "## Persona LoRA-DPO vs base by category (quality criterion)",
        "",
        report.df_to_markdown(report.per_category_table(**{k: v for k, v in sp["Persona LoRA-DPO vs base SmolLM2 (40 prompts): quality"].items()})),
        "",
        "## Headline",
        "",
        "- **HH-RLHF DPO beats SFT** (0.72 [0.66, 0.77]). The gains are concentrated in harmful-instruction, privacy and cyber-abuse prompts; benign helpfulness is 0.68 (no over-refusal), and calibration (0.42) is the only losing category. On the 185 distinct prompts: 0.71 [0.64, 0.78].",
        "- **UltraFeedback DPO does not beat SFT** (0.46 [0.39, 0.52]). Its answers are 37% longer, and it loses mainly where it hits the 512-token limit (10–26), usually in a repetition loop. Uncapped answers are even (93–97).",
        "- **Persona LoRA-DPO is far more in character (style 0.95) but less useful (quality 0.39).** The two are reported separately and not combined into one score.",
        "",
        "Details, plots and examples: `PA2_report.pdf`; raw judgments in `outputs/eval/`.",
        "",
    ]
    (OUT / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT, sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
