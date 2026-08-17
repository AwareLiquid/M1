# PC-Liquid-Core validation report

Config: `{'n_seq': 320, 'seq_len': 96, 'd_in': 1, 'epochs': 40, 'batch': 64, 'lr': 0.003, 'prime': 48, 'seed': 0}`  |  runtime 227.5s

Persistence reference (predict x_t+1=x_t) 1-step MSE: **0.32812**

## Main task (multi-timescale signal)

| model | #params | 1-step MSE | rollout MSE |
|---|---:|---:|---:|
| PCLiquidCore | 14356 | 0.01650 | 0.12869 |
| GRU | 15401 | 0.02575 | 0.05092 |
| LSTM | 15181 | 0.03448 | 0.03188 |
| Transformer | 22305 | 0.28095 | 0.87553 |

## Continual-learning probe (train A -> train B, re-measure A)

| model | A before | A after B | forgetting (lower=better) |
|---|---:|---:|---:|
| PCLiquidCore | 0.00413 | 1.33673 | +1.33261 |
| GRU | 0.00615 | 2.12697 | +2.12082 |
| LSTM | 0.00891 | 2.13354 | +2.12463 |
| Transformer | 0.01009 | 1.15673 | +1.14665 |
