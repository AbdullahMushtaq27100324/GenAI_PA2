| run | status | total_steps | best_step | best_val_loss | train_hours | train_examples | val_examples | warmup_steps | effective_batch | best_val_reward_accuracy | best_val_reward_margin | train_pairs | val_pairs | reference_checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT | done | 938 | 938 | 1.3557164914284405 | 1.437 | 30000.0 | 1000.0 | 29.0 | 32 | - | - | - | - | - |
| DPO-HH | done | 469 | 469 | - | 0.8 | - | - | - | 32 | 0.647 | 0.2421069710701704 | 15000.0 | 1000.0 | outputs/sft/best |
| DPO-UF | done | 469 | 400 | - | 1.746 | - | - | - | 32 | 0.726 | 0.6593659233301878 | 15000.0 | 1000.0 | outputs/sft/best |