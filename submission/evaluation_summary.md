# PA2 evaluation summary: DPO vs SFT (and persona LoRA vs base)

- **Judge:** `google/gemini-3.6-flash` via `kaggle_benchmarks` (Kaggle Benchmarks task notebook), temperature 0.
- **Blinding:** A/B order randomized per pair (seed 42); model identities hidden from the judge.
- **Reliability:** 5 pairs per request, ≥ 5 s sleep before every attempt, up to 3 attempts, checkpointed after every batch, failures logged.
- **Same prompts and decoding** for both models in a comparison: greedy, at most 512 new tokens. Repetition penalty 1.0 for the Qwen models, 1.1 for the SmolLM2 pair.
- **Win rate (ties excluded)** = W/(W+L) with a Wilson 95% CI. **Win rate (ties = ½)** = (W + ½T)/(W+T+L). *Unknown* (invalid verdict) is excluded from both.
- **Run outcome:** all 640 pairs judged; 0 failed requests; 0 retries; 1 unknown.

## Overall

| comparison | n | wins | ties | losses | unknown | failures | win_rate_excl_ties | ci95 | win_rate_ties_half |
|---|---|---|---|---|---|---|---|---|---|
| UltraFeedback DPO vs SFT (AlpacaEval, 300 prompts) | 300 | 103 | 73 | 123 | 1 | 0 | 0.456 | [0.39, 0.52] | 0.467 |
| HH-RLHF DPO vs SFT (HH eval set, 300 prompts) | 300 | 190 | 35 | 75 | 0 | 0 | 0.717 | [0.66, 0.77] | 0.692 |
| Persona LoRA-DPO vs base SmolLM2 (40 prompts): style | 40 | 18 | 21 | 1 | 0 | 0 | 0.947 | [0.75, 0.99] | 0.713 |
| Persona LoRA-DPO vs base SmolLM2 (40 prompts): quality | 40 | 14 | 4 | 22 | 0 | 0 | 0.389 | [0.25, 0.55] | 0.4 |

## HH-RLHF DPO vs SFT by category

| category | n | wins | ties | losses | unknown | win_rate_excl_ties | win_rate_ties_half |
|---|---|---|---|---|---|---|---|
| benign_helpfulness | 30 | 19 | 2 | 9 | 0 | 0.679 | 0.667 |
| calibration_and_uncertainty | 30 | 10 | 6 | 14 | 0 | 0.417 | 0.433 |
| cyber_abuse | 30 | 22 | 5 | 3 | 0 | 0.88 | 0.817 |
| harmful_instructions | 30 | 24 | 5 | 1 | 0 | 0.96 | 0.883 |
| hate_or_harassment | 30 | 15 | 5 | 10 | 0 | 0.6 | 0.583 |
| legal_financial_safety | 30 | 25 | 0 | 5 | 0 | 0.833 | 0.833 |
| medical_safety | 30 | 16 | 2 | 12 | 0 | 0.571 | 0.567 |
| privacy_and_doxxing | 30 | 21 | 7 | 2 | 0 | 0.913 | 0.817 |
| safe_transformation | 30 | 17 | 3 | 10 | 0 | 0.63 | 0.617 |
| self_harm_support | 30 | 21 | 0 | 9 | 0 | 0.7 | 0.7 |

## UltraFeedback DPO vs SFT by AlpacaEval subset

| category | n | wins | ties | losses | unknown | win_rate_excl_ties | win_rate_ties_half |
|---|---|---|---|---|---|---|---|
| helpful_base | 51 | 12 | 17 | 22 | 0 | 0.353 | 0.402 |
| koala | 60 | 18 | 15 | 27 | 0 | 0.4 | 0.425 |
| oasst | 54 | 25 | 9 | 19 | 1 | 0.568 | 0.557 |
| selfinstruct | 95 | 32 | 22 | 41 | 0 | 0.438 | 0.453 |
| vicuna | 40 | 16 | 10 | 14 | 0 | 0.533 | 0.525 |

## Persona LoRA-DPO vs base by category (quality criterion)

| category | n | wins | ties | losses | unknown | win_rate_excl_ties | win_rate_ties_half |
|---|---|---|---|---|---|---|---|
| coding | 10 | 3 | 2 | 5 | 0 | 0.375 | 0.4 |
| cooking | 10 | 3 | 0 | 7 | 0 | 0.3 | 0.3 |
| general_help | 10 | 3 | 1 | 6 | 0 | 0.333 | 0.35 |
| safety_sensitive | 10 | 5 | 1 | 4 | 0 | 0.556 | 0.55 |

## Headline

- **HH-RLHF DPO beats SFT** (0.72 [0.66, 0.77]). The gains are concentrated in harmful-instruction, privacy and cyber-abuse prompts; benign helpfulness is 0.68 (no over-refusal), and calibration (0.42) is the only losing category. On the 185 distinct prompts: 0.71 [0.64, 0.78].
- **UltraFeedback DPO does not beat SFT** (0.46 [0.39, 0.52]). Its answers are 37% longer, and it loses mainly where it hits the 512-token limit (10–26), usually in a repetition loop. Uncapped answers are even (93–97).
- **Persona LoRA-DPO is far more in character (style 0.95) but less useful (quality 0.39).** The two are reported separately and not combined into one score.

Details, plots and examples: `PA2_report.pdf`; raw judgments in `outputs/eval/`.
