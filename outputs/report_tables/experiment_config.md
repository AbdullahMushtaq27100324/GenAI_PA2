| run | init | trainable | optimizer | peak_lr | schedule | weight_decay | grad_clip | micro_batch | grad_accum | effective_batch | precision | max_len | epochs/steps | beta | reference | seed | selected_ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT | Qwen/Qwen2.5-0.5B | full fine-tuning | adamw8bit | 2e-05 | cosine (warmup 29 steps) | 0.1 | 1.0 | 4 | 8 | 32 | fp16 autocast, fp32 weights, GradScaler | 512 | 1 ep / 938 steps | - | - | 42 | best val loss @ step 938 |
| HH-RLHF DPO | SFT best | full fine-tuning | adamw8bit | 1e-06 | constant (warmup 0 steps) | 0.0 | 1.0 | 1 pair | 32 | 32 | fp16 autocast, fp32 weights, GradScaler | 512 (prompt <= 256) | 1 ep / 469 steps | 0.1 | frozen SFT best (precomputed, fp32) | 42 | best val reward acc @ step 469 |
| UltraFeedback DPO | SFT best | full fine-tuning | adamw8bit | 1e-06 | constant (warmup 0 steps) | 0.0 | 1.0 | 1 pair | 32 | 32 | fp16 autocast, fp32 weights, GradScaler | 512 (prompt <= 256) | 1 ep / 469 steps | 0.1 | frozen SFT best (precomputed, fp32) | 42 | best val reward acc @ step 400 |
| LoRA-DPO (persona) | HuggingFaceTB/SmolLM2-360M-Instruct | LoRA r=16 a=32 drop=0.05 on q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj | adamw (TRL default) | 0.0001 | cosine (warmup ratio 0.1) | 0.0 | 1.0 | 4 | 4 | 16 | bf16 autocast, fp32 weights | 512 | 5 ep / 65 steps | 0.1 | base model, adapter disabled | 42 | min val DPO loss (outputs/lora/r16_lr1e-4/trainer/checkpoint-60) |

| generation_set | model | adapter | decoding | temperature | top_p | max_new_tokens | repetition_penalty | template |
|---|---|---|---|---|---|---|---|---|
| alpaca_eval__qwen2.5-0.5b-base | Qwen/Qwen2.5-0.5B | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.0 | chatml |
| alpaca_eval__qwen2.5-0.5b-sft | outputs/sft/best | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.0 | chatml |
| alpaca_eval__qwen2.5-0.5b-dpo-uf | outputs/dpo_uf/best | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.0 | chatml |
| hh_eval__qwen2.5-0.5b-base | Qwen/Qwen2.5-0.5B | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.0 | chatml |
| hh_eval__qwen2.5-0.5b-sft | outputs/sft/best | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.0 | chatml |
| hh_eval__qwen2.5-0.5b-dpo-hh | outputs/dpo_hh/best | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.0 | chatml |
| persona_eval__smollm2-360m-instruct | HuggingFaceTB/SmolLM2-360M-Instruct | - | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.1 | chat_template |
| persona_eval__smollm2-360m-instruct-lora-dpo | HuggingFaceTB/SmolLM2-360M-Instruct | outputs/lora/r16_lr1e-4/best_adapter | greedy | n/a (greedy) | n/a (greedy) | 512 | 1.1 | chat_template |