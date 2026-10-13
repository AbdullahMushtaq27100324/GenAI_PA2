# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

This repo contains:
- supervised fine-tuning (SFT) of `Qwen/Qwen2.5-0.5B`, followed by two **from-scratch** DPO runs (HH-RLHF and UltraFeedback);
- a TRL `DPOTrainer` + LoRA persona run on `HuggingFaceTB/SmolLM2-360M-Instruct`.

Every DPO model is compared with SFT by a blinded pairwise judge (`google/gemini-3.6-flash` via Kaggle Benchmarks). It builds on the course starter repository (`ahmad903501/small-model-alignment-starter`).

- **Report:** [`report/PA2_report.pdf`](report/PA2_report.pdf) (7 pages)
- **Results walk-through:** [`notebooks/results_summary.ipynb`](notebooks/results_summary.ipynb)
- **Submission bundle:** [`submission/`](submission/), mapped to the checklist in [`SUBMISSION.md`](SUBMISSION.md)

## Results at a glance

| Comparison (evaluation set) | W / T / L | Win rate, ties excluded [95% CI] |
|---|---|---|
| HH-RLHF DPO vs SFT (HH eval, 300) | 190 / 35 / 75 | **0.717** [0.66, 0.77] |
| UltraFeedback DPO vs SFT (AlpacaEval, 300) | 103 / 73 / 123 (+1 unknown) | **0.456** [0.39, 0.52] |
| Persona LoRA vs base SmolLM2 (style, 40) | 18 / 21 / 1 | **0.947** [0.75, 0.99] |
| Persona LoRA vs base SmolLM2 (quality, 40) | 14 / 4 / 22 | **0.389** [0.25, 0.55] |

## Where things are

| Path | Contents |
|---|---|
| `submission_adapter.py` | Grading interface: thin wrappers around `alignment/` |
| `alignment/batching.py` | Prompt/response tokenization, EOS-then-truncate, `-100` masks, right padding |
| `alignment/logprobs.py` | Response-only summed log-probabilities (fp32, deterministic), including a memory-saving HF path |
| `alignment/losses.py` | SFT loss (token-mean reduction) and DPO loss/rewards |
| `alignment/train_sft.py` | From-scratch SFT loop (no HF `Trainer`) |
| `alignment/train_dpo.py` | From-scratch DPO loop, shared by both preference datasets |
| `alignment/train_lora_dpo.py` | TRL `DPOTrainer` + PEFT LoRA (persona task only) |
| `alignment/data.py` | Dataset download, filtering and seed-42 subsets |
| `alignment/generate.py` | Batched greedy generation, resumable |
| `alignment/pairwise.py`, `alignment/judge.py` | Blinded A/B pair files, resumable batched judging, summaries |
| `scripts/judge_pairwise_kaggle.py` | Judge CLI (manual §13.2) |
| `notebooks/PA2_local_pipeline.ipynb` | **Driver notebook for every experiment** (`DRY_RUN` toggle), with the outputs of the full run |
| `notebooks/kaggle_setup_and_judge.ipynb` | Task 1 Kaggle T4×2 check and Task 7 judging |
| `notebooks/kaggle_task1_executed.ipynb`, `notebooks/kaggle_judging_executed.ipynb` | The executed Kaggle runs |
| `notebooks/results_summary.ipynb` | Every result, plot and discussion, read from `outputs/` |
| `eval_sets/` | Fixed evaluation sets: HH (given), AlpacaEval 300 / seed 42, persona style set |
| `data/processed/*.ids.json` | Exact IDs of every train/val subset used |
| `outputs/` | Configs, metrics, plots, generations, pair files, judgments, report tables (checkpoints are not in git) |
| `report/` | Report source (`report.md`), build script, DOCX and PDF |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch                        # any CUDA build; not pinned (tested with 2.5.1+cu121)
pip install -r requirements.txt ipykernel papermill
PYTHONPATH=. python -m pytest -q tests/  # 25 tests
```

`trl==0.24.0` is pinned because newer TRL releases require torch ≥ 2.6.

## Running

Open `notebooks/PA2_local_pipeline.ipynb`, keep `DRY_RUN = True` for a tiny end-to-end test, then set it to `False` and run all cells. To run it headless instead:

```bash
bash scripts/run_pipeline.sh dry    # every stage end to end with tiny sizes -> outputs_dryrun/
bash scripts/run_pipeline.sh full   # full experiments -> outputs/
```

The runs are crash-safe:
- every stage is idempotent;
- training resumes from `<run>/last/`, which is written atomically;
- generation and judging resume from their partial files.

After an interruption, run it again. Downstream stages always use each run's `best/` checkpoint:
- **SFT:** lowest validation loss.
- **DPO:** highest validation reward accuracy.
- **LoRA:** within a run, lowest validation DPO loss. Across the sweep, the largest validation reward margin among configs whose generations on the 20 validation prompts are at most 5% degenerate (the lowest-loss runs over-optimize).

Judging runs on Kaggle (`notebooks/kaggle_setup_and_judge.ipynb`, Part B in a Kaggle Benchmarks task notebook), or locally after `kaggle b init -y` with `RUN_JUDGE_LOCALLY=True`.

Rebuild the report and the submission bundle:

```bash
python scripts/make_report_figures.py && python3 report/build_report.py && python scripts/make_submission.py
```

## Public adapter tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

`compute_sft_loss` uses the **mean over all unmasked response tokens** reduction.
