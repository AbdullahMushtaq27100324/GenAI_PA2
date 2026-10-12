"""Generate notebooks/kaggle_setup_and_judge.ipynb (Task 1 setup check + Task 7 judging on Kaggle).

Run:  python scripts/build_kaggle_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_notebook as nbb  # noqa: E402

nbb.CELLS.clear()
md, code = nbb.md, nbb.code

md(r"""
# PA2 — Kaggle notebook: Task 1 setup check + Task 7 judging

Training and generation run locally (see `notebooks/PA2_local_pipeline.ipynb`). This notebook covers the two parts that need Kaggle:

- **Part A (Task 1):** run with **Accelerator = GPU T4 x2**. Clones the repo, installs dependencies without reinstalling PyTorch, confirms two T4s, prints versions, and runs the adapter tests.
- **Part B (Task 7):** run as a notebook with **Kaggle Benchmarks / model access enabled**, so `kaggle_benchmarks` can call `google/gemini-3.6-flash`. It judges the saved pairwise JSONL files only; nothing is regenerated here.
  - Batching: 5 pairs per request, at least 5 s sleep between attempts, up to 3 retries.
  - Durability: checkpoints after each batch and keeps a failure log.
  - Resume: rerunning skips pairs already judged.

Everything is written under `/kaggle/working`. Save a notebook version when it finishes. To resume after a disconnect, attach the previous version's output as an input dataset; the copy cell restores the existing judgments first.
""")

code(r"""
RUN_PART_A_SETUP = True      # Task 1 (needs GPU T4 x2)
RUN_PART_B_JUDGING = True    # Task 7 (needs Kaggle model access)
REPO_URL = "https://github.com/<your-username>/<your-repo>.git"   # <-- set this (private repo: use a token URL or attach as dataset)
JUDGE_MODEL = "google/gemini-3.6-flash"
PREVIOUS_JUDGMENTS = None    # e.g. "/kaggle/input/pa2-judge-v1/judgments" to resume from an earlier version
""", tags=["parameters"])

code(r"""
import os, sys, json, shutil, subprocess, zipfile
from pathlib import Path
WORK = Path("/kaggle/working")
REPO = WORK / "pa2"
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
print("repo at", REPO, "| commit", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
""")

md("## Part A — Task 1: environment and GPU check")
code(r"""
if RUN_PART_A_SETUP:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    # Manual Section 5.2: keep Kaggle's CUDA PyTorch, avoid the stale torchao / PEFT conflict.
    subprocess.run("python -m pip uninstall -y -q torchao || true", shell=True)
    subprocess.run('python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub '
                   'accelerate tqdm pyyaml sentencepiece safetensors pytest', shell=True, check=True)
    subprocess.run("python -m pip install -q -e . --no-deps", shell=True, check=True)
""")
code(r"""
if RUN_PART_A_SETUP:
    import torch, transformers, datasets
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB")
    assert n == 2 and all("T4" in torch.cuda.get_device_name(i) for i in range(n)), "expected two Tesla T4 GPUs"
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_public_adapter_smoke.py", "tests/test_own_alignment.py"],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    print(res.stdout[-3000:], res.stderr[-1000:])
""")

md(r"""
## Part B — Task 7: pairwise judging
Pair files are read from `outputs/eval/*_pairs.jsonl` in the repo. They are committed by the local pipeline, or can come from an attached `kaggle_judge_bundle.zip` dataset.
""")
code(r"""
if RUN_PART_B_JUDGING:
    import kaggle_benchmarks as kbench
    print("available judge models:", sorted(kbench.llms)[:50])
    JUDGE_DIR = WORK / "judgments"
    JUDGE_DIR.mkdir(exist_ok=True)
    pair_sources = list((REPO / "outputs" / "eval").glob("*_pairs.jsonl")) or list(Path("/kaggle/input").rglob("*_pairs.jsonl"))
    for p in pair_sources:
        shutil.copy(p, JUDGE_DIR / p.name)
    if PREVIOUS_JUDGMENTS:  # resume: restore judged/failure files from an earlier saved version
        for p in Path(PREVIOUS_JUDGMENTS).glob("*_judged*.jsonl"):
            shutil.copy(p, JUDGE_DIR / p.name)
    print(sorted(p.name for p in JUDGE_DIR.iterdir()))
""")
code(r"""
RUBRICS = {"uf_dpo_vs_sft": "general", "hh_dpo_vs_sft": "safety", "persona_lora_vs_base": "persona"}
if RUN_PART_B_JUDGING:
    for name, rubric in RUBRICS.items():
        pairs = JUDGE_DIR / f"{name}_pairs.jsonl"
        if not pairs.exists():
            print("missing", pairs); continue
        cmd = [sys.executable, "scripts/judge_pairwise_kaggle.py", "--pairs_path", str(pairs),
               "--output_path", str(JUDGE_DIR / f"{name}_judged.jsonl"), "--judge_model", JUDGE_MODEL,
               "--rubric", rubric, "--batch_size", "5", "--sleep_seconds", "5", "--max_retries", "3", "--reuse_judgments"]
        print(" ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
        print(res.stdout[-4000:], res.stderr[-2000:])
""")
code(r"""
if RUN_PART_B_JUDGING:
    for p in sorted(JUDGE_DIR.glob("*_summary.json")):
        print(p.name, json.dumps({k: v["overall"] for k, v in json.load(open(p)).items()}, indent=1))
    with zipfile.ZipFile(WORK / "judgments.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in JUDGE_DIR.iterdir():
            z.write(p, f"judgments/{p.name}")
    print("Saved", WORK / "judgments.zip", "- Save Version now, then copy *_judged*.jsonl into outputs/eval/ locally.")
""")

if __name__ == "__main__":
    nbb.build(Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_setup_and_judge.ipynb")
