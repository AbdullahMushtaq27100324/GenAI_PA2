# Submission checklist (manual §16)

| # | Required | Where |
|---|---|---|
| 1 | GitHub repository URL | https://github.com/AbdullahMushtaq27100324/GenAI_PA2 |
| 2 | Short final report (PDF, ≤ 8 pages) | [`submission/PA2_report.pdf`](submission/PA2_report.pdf) (7 pages; source `report/report.md`, built by `report/build_report.py`) |
| 3 | Training plots for SFT and DPO runs | [`submission/training_plots/`](submission/training_plots/): `sft_curves.png`, `dpo_curves.png`, `lora_dpo_curves.png` (+ `results_overview.png`); PDFs in `outputs/figures/` |
| 4 | Evaluation summary comparing DPO against SFT | [`submission/evaluation_summary.md`](submission/evaluation_summary.md); raw data in `outputs/eval/` (`*_pairs.jsonl`, `*_judged.jsonl`, `*_judged_summary.json`) |
| 5 | Root-level `submission_adapter.py` connected to the implementation, public tests passing | [`submission_adapter.py`](submission_adapter.py) → `alignment/`. `PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py` gives 7 passed; the full suite (`tests/`) gives 25 passed |

## Task-by-task evidence

| Task | Evidence |
|---|---|
| 1 Repository + Kaggle setup | `notebooks/kaggle_task1_executed.ipynb` (2× Tesla T4, package versions, tests) |
| 2 Baseline generation | `outputs/generations/alpaca_eval__qwen2.5-0.5b-base.json` (AlpacaEval 300, seed 42; `eval_sets/alpaca_eval_300_seed42.json`) |
| 3 SFT | `alignment/train_sft.py`; `outputs/sft/{config.json, metrics.jsonl, summary.json}` |
| 4 Two from-scratch DPO runs | `alignment/train_dpo.py`; `outputs/dpo_hh/`, `outputs/dpo_uf/` |
| 5 LoRA persona (TRL DPOTrainer + PEFT) | `alignment/train_lora_dpo.py`; `outputs/lora/` (sweep, `selection.json`, submitted adapter `r16_lr1e-4/best_adapter`) |
| 6 Evaluation outputs + blinded pair files | `outputs/generations/`, `outputs/eval/*_pairs.jsonl`; sanity checks in `notebooks/PA2_local_pipeline.ipynb` |
| 7 Kaggle-model judging | `scripts/judge_pairwise_kaggle.py`, `alignment/judge.py`; `notebooks/kaggle_judging_executed.ipynb`; `outputs/eval/*_judged*.{jsonl,json}` |
| 8 Report | `report/PA2_report.pdf` |

**Before submitting:** the repository is private. Make it public, or add the course staff as collaborators, so the link in the report works for them.
