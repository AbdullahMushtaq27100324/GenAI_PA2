"""Judge saved pairwise JSONL files with a Kaggle model-proxy judge.

Example (manual Section 13.2):
    python scripts/judge_pairwise_kaggle.py \
      --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
      --output_path outputs/eval/dpo_vs_sft_judged.jsonl \
      --judge_model google/gemini-3.6-flash \
      --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

Runs inside a Kaggle notebook (kaggle_benchmarks preinstalled) or locally after
``kaggle b init -y``. ``--judge_model mock`` runs an offline fake judge for testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judge import run_judging  # noqa: E402
from alignment.pairwise import summarize_judgments  # noqa: E402
from alignment.utils import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--judge_model", default="google/gemini-3.6-flash")
    ap.add_argument("--rubric", choices=["general", "safety", "persona"], default="general")
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--sleep_seconds", type=float, default=5.0)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--reuse_judgments", action="store_true", help="skip pairs already in --output_path")
    ap.add_argument("--failures_path", default=None)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N pairs (smoke test)")
    args = ap.parse_args()
    if args.sleep_seconds < 5 and args.judge_model != "mock":
        ap.error("--sleep_seconds must be at least 5 to avoid overloading the model proxy")

    result = run_judging(
        args.pairs_path, args.output_path, args.judge_model, args.rubric,
        batch_size=args.batch_size, sleep_seconds=args.sleep_seconds, max_retries=args.max_retries,
        reuse_judgments=args.reuse_judgments, failures_path=args.failures_path, limit=args.limit,
    )
    criteria = ["style_winner", "quality_winner"] if args.rubric == "persona" else ["winner"]
    summary = {c: summarize_judgments(args.pairs_path, args.output_path, result["failures_path"], c) for c in criteria}
    out = Path(args.output_path)
    write_json(out.with_name(out.stem + "_summary.json"), summary)
    print(json.dumps({c: s["overall"] for c, s in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
