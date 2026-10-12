# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

Supervised fine-tuning and two **from-scratch** DPO runs (HH-RLHF, UltraFeedback) on `Qwen/Qwen2.5-0.5B`, plus a TRL `DPOTrainer` + LoRA persona run on `HuggingFaceTB/SmolLM2-360M-Instruct`. Models are evaluated with blinded pairwise judgments from a Kaggle model-proxy judge.

Built on the course starter repository (`ahmad903501/small-model-alignment-starter`).

## Where things are

| Path | Contents |
|---|---|
| `submission_adapter.py` | Grading interface: thin wrappers around `alignment/` |
| `alignment/batching.py` | Prompt/response tokenization, EOS-then-truncate, `-100` masks, right padding |
| `alignment/logprobs.py` | Response-only summed log-probabilities (fp32), including a memory-saving HF path |
| `alignment/losses.py` | SFT loss (token-mean reduction) and DPO loss/rewards |
| `alignment/train_sft.py` | From-scratch SFT loop (no HF `Trainer`) |
| `alignment/train_dpo.py` | From-scratch DPO loop, shared by both preference datasets |
| `alignment/train_lora_dpo.py` | TRL `DPOTrainer` + PEFT LoRA (persona task only) |
| `alignment/data.py` | Dataset download, filtering, and seed-42 subsets |
| `alignment/generate.py` | Batched greedy generation, resumable |
| `alignment/pairwise.py`, `alignment/judge.py` | Blinded A/B pair files, resumable batched judging, summaries |
| `scripts/judge_pairwise_kaggle.py` | Judge CLI (manual §13.2) |
| `notebooks/PA2_local_pipeline.ipynb` | **Driver notebook for every experiment** (`DRY_RUN` toggle) |
| `notebooks/kaggle_setup_and_judge.ipynb` | Task 1 Kaggle T4×2 check and Task 7 judging |
| `notebooks/kaggle_*_executed.ipynb` | The executed Kaggle runs (Task 1 evidence, judging log) |
| `notebooks/results_summary.ipynb` | **Read this first:** every result, plot and discussion |
| `eval_sets/` | Fixed evaluation sets: HH (given), AlpacaEval 300 / seed 42, persona style set |
| `data/processed/*.ids.json` | Exact IDs of every train/val subset used |
| `outputs/` | Metrics, plots, configs, generations, pair files, judgments, report tables |

## Setup (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch            # any CUDA build; not pinned
pip install -r requirements.txt papermill ipykernel
pip install -e . --no-deps
python -m ipykernel install --user --name pa2 --display-name "PA2 (.venv)"
PYTHONPATH=. python -m pytest -q tests/
```

## Running

```bash
bash scripts/run_pipeline.sh dry    # every stage end to end with tiny sizes -> outputs_dryrun/
bash scripts/run_pipeline.sh full   # full experiments -> outputs/
```

The runs are crash-safe. Every stage is idempotent, training resumes from `<run>/last/` (written atomically), and generation and judging resume from partial files. If the run is interrupted, re-run the same command. Downstream stages always use each run's `best/` checkpoint:
- SFT: lowest validation loss
- DPO: highest validation reward accuracy
- LoRA: within a run, lowest validation DPO loss; across the sweep, the largest validation reward margin among
  configs whose generations on the 20 validation prompts are at most 5% degenerate (lowest-loss runs over-optimize)

Judging runs on Kaggle (`notebooks/kaggle_setup_and_judge.ipynb`), or locally after `kaggle b init -y` by setting `RUN_JUDGE_LOCALLY=True`.

## Public adapter tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

`compute_sft_loss` uses the **mean over all unmasked response tokens** reduction.
