"""Blinded pairwise comparison files and judgment summaries.

Pair file (JSONL, written *before* judging): one row per prompt with the two
responses in a per-row random order (seed 42). The judge only ever sees
``instruction``, ``response_a``, ``response_b`` (and, for the safety set, the
prompt's expected behaviour). Model identities live in ``hidden_metadata`` and
are used only after judging to map A/B back to models.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from alignment.utils import read_json, read_jsonl, write_jsonl


def build_pairs(
    treatment_path: str | Path,
    baseline_path: str | Path,
    out_path: str | Path,
    treatment: str,
    baseline: str,
    eval_name: str,
    seed: int = 42,
    eval_rows: list[dict] | None = None,
) -> list[dict]:
    """Pair treatment (e.g. DPO) vs baseline (e.g. SFT) outputs on the same prompts."""
    treat = {r["id"]: r for r in read_json(treatment_path)}
    base = {r["id"]: r for r in read_json(baseline_path)}
    extra = {r["id"]: r for r in (eval_rows or [])}
    ids = [r["id"] for r in read_json(baseline_path) if r["id"] in treat]
    rng = random.Random(seed)
    rows = []
    for pid in ids:
        t, b = treat[pid], base[pid]
        assert t["instruction"] == b["instruction"], f"prompt mismatch for {pid}"
        treatment_first = rng.random() < 0.5
        a, bb = (t, b) if treatment_first else (b, t)
        rows.append(
            {
                "pair_id": f"{eval_name}::{pid}",
                "id": pid,
                "eval_set": eval_name,
                "category": b.get("category", ""),
                "instruction": b["instruction"],
                "expected_behavior": extra.get(pid, {}).get("expected_behavior", ""),
                "response_a": a["output"],
                "response_b": bb["output"],
                "hidden_metadata": {
                    "model_a": a["generator"],
                    "model_b": bb["generator"],
                    "treatment": treatment,
                    "baseline": baseline,
                    "treatment_position": "A" if treatment_first else "B",
                },
            }
        )
    write_jsonl(out_path, rows)
    return rows


def outcome_for_treatment(verdict: str, treatment_position: str) -> str:
    if verdict == "tie":
        return "tie"
    if verdict not in ("A", "B"):
        return "unknown"
    return "win" if verdict == treatment_position else "loss"


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rates(counts: Counter, n_pairs: int) -> dict:
    w, l, t = counts["win"], counts["loss"], counts["tie"]
    decided = w + l
    judged = w + l + t
    lo, hi = wilson_interval(w, decided)
    return {
        "n_pairs": n_pairs,
        "wins": w,
        "losses": l,
        "ties": t,
        "unknown": counts["unknown"],
        "failures": counts["failure"],
        "pending": counts["pending"],
        "win_rate_excl_ties": w / decided if decided else float("nan"),
        "win_rate_excl_ties_ci95": [lo, hi],
        "win_rate_ties_half": (w + 0.5 * t) / judged if judged else float("nan"),
        "win_share_all": w / n_pairs if n_pairs else float("nan"),
        "tie_rate": t / n_pairs if n_pairs else float("nan"),
        "unknown_rate": counts["unknown"] / n_pairs if n_pairs else float("nan"),
        "failure_rate": counts["failure"] / n_pairs if n_pairs else float("nan"),
    }


def summarize_judgments(pairs_path, judged_path, failures_path=None, criterion: str = "winner") -> dict:
    """Win/loss/tie/unknown/failure counts for the treatment model, overall and per category."""
    pairs = read_jsonl(pairs_path)
    judged = {r["pair_id"]: r for r in read_jsonl(judged_path)}
    failed = {r["pair_id"] for r in read_jsonl(failures_path)} if failures_path else set()
    overall, per_cat = Counter(), defaultdict(Counter)
    retries = Counter()
    for p in pairs:
        j = judged.get(p["pair_id"])
        if j is None:
            # Never judged: either logged as an unresolved failure, or the run has not reached it yet.
            outcome = "failure" if p["pair_id"] in failed else "pending"
        else:
            verdict = j.get(criterion, j.get("winner", "unknown"))
            outcome = outcome_for_treatment(verdict, p["hidden_metadata"]["treatment_position"])
            retries[j.get("attempts", 1)] += 1
        overall[outcome] += 1
        per_cat[p.get("category") or "all"][outcome] += 1
    meta = pairs[0]["hidden_metadata"] if pairs else {}
    any_judged = next(iter(judged.values()), {})
    return {
        "treatment": meta.get("treatment"),
        "baseline": meta.get("baseline"),
        "criterion": criterion,
        "judge_model": any_judged.get("judge_model"),
        "overall": _rates(overall, len(pairs)),
        "per_category": {c: _rates(cnt, sum(cnt.values())) for c, cnt in sorted(per_cat.items())},
        "attempts_histogram": dict(sorted(retries.items())),
    }


def length_stats(pairs_path) -> dict:
    """Mean response length (words) for treatment vs baseline - a verbosity check."""
    pairs = read_jsonl(pairs_path)
    t_len, b_len = [], []
    for p in pairs:
        pos = p["hidden_metadata"]["treatment_position"]
        t, b = (p["response_a"], p["response_b"]) if pos == "A" else (p["response_b"], p["response_a"])
        t_len.append(len(t.split()))
        b_len.append(len(b.split()))
    n = max(1, len(pairs))
    return {"treatment_mean_words": sum(t_len) / n, "baseline_mean_words": sum(b_len) / n,
            "treatment_longer_frac": sum(a > b for a, b in zip(t_len, b_len)) / n}
