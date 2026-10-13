"""Results figure for the report, built only from the saved judgments in outputs/eval/.

    python scripts/make_report_figures.py   ->  outputs/figures/results_overview.{png,pdf}

Left: wins / ties / losses of the treatment model in each comparison.
Right: HH-RLHF DPO win rate against SFT per safety category (ties excluded).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL, FIG = ROOT / "outputs" / "eval", ROOT / "outputs" / "figures"
BLUE, GRAY, ORANGE, INK2, GRID = "#2a78d6", "#b8b6ad", "#eb6834", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def jsonl(path):
    # split on "\n" only: generated text contains U+2028, which str.splitlines() would split on
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def outcomes(name, key="winner"):
    pairs = {p["pair_id"]: p for p in jsonl(EVAL / f"{name}_pairs.jsonl")}
    rows = []
    for j in jsonl(EVAL / f"{name}_judged.jsonl"):
        p, v = pairs[j["pair_id"]], j[key]
        pos = p["hidden_metadata"]["treatment_position"]
        o = "tie" if v == "tie" else ("win" if v == pos else "loss" if v in ("A", "B") else "unknown")
        rows.append((p["category"], o))
    return rows


def main() -> None:
    comps = [("UltraFeedback DPO vs SFT\n(AlpacaEval, n=300)", outcomes("uf_dpo_vs_sft")),
             ("HH-RLHF DPO vs SFT\n(HH eval, n=300)", outcomes("hh_dpo_vs_sft")),
             ("Persona LoRA vs base: style\n(n=40)", outcomes("persona_lora_vs_base", "style_winner")),
             ("Persona LoRA vs base: quality\n(n=40)", outcomes("persona_lora_vs_base", "quality_winner"))]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.05, 1]})
    ax.set_axisbelow(True)
    for i, (label, rows) in enumerate(comps):
        c = collections.Counter(o for _, o in rows)
        n = len(rows)
        left = 0.0
        for key, color in (("win", BLUE), ("tie", GRAY), ("loss", ORANGE)):
            width = c[key] / n
            ax.barh(i, width, left=left, color=color, edgecolor="white", linewidth=1.5, height=0.62)
            if width > 0.06:
                ax.text(left + width / 2, i, str(c[key]), ha="center", va="center", fontsize=7,
                        color="white" if key != "tie" else "#2b2b2b")
            left += width
    ax.set_yticks(range(len(comps)), [c[0] for c in comps], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
    ax.set_xlabel("share of pairs (labels = counts)")
    ax.set_title("(a) Treatment wins / ties / losses")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GRAY, ORANGE)],
              labels=["treatment wins", "tie", "treatment loses"], loc="upper center",
              bbox_to_anchor=(0.45, -0.22), ncol=3, fontsize=7)

    per = collections.defaultdict(collections.Counter)
    for cat, o in outcomes("hh_dpo_vs_sft"):
        per[cat][o] += 1
    rates = sorted(((c["win"] / (c["win"] + c["loss"]), cat.replace("_", " ")) for cat, c in per.items()))
    bx.set_axisbelow(True)
    bx.barh([r[1] for r in rates], [r[0] for r in rates], color=BLUE, height=0.65)
    for y, (v, _) in enumerate(rates):
        bx.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=6.5)
    bx.axvline(0.5, color=INK2, lw=0.7, ls=":")
    bx.set_xlim(0, 1.08)
    bx.tick_params(axis="y", labelsize=7)
    bx.set_xlabel("win rate vs SFT (ties excluded)")
    bx.set_title("(b) HH-RLHF DPO by category (n=30 each)")
    fig.tight_layout(w_pad=1.5)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "results_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / "results_overview.pdf", bbox_inches="tight", pad_inches=0.03)
    print("wrote", FIG / "results_overview.png")


if __name__ == "__main__":
    main()
