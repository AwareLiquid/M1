# PC-Liquid-Core validation report

Config: `{'n_seq': 320, 'seq_len': 96, 'd_in': 1, 'epochs': 40, 'batch': 64, 'lr': 0.003, 'prime': 48, 'seed': 0, 'ss_max': 0.5}`  |  runtime 202.1s

Persistence reference (predict x_t+1=x_t) 1-step MSE: **0.32812**

## Main task (multi-timescale signal)

| model | #params | 1-step MSE | rollout MSE |
|---|---:|---:|---:|
| PCLiquidCore | 14359 | 0.01855 | 0.09444 |
| GRU | 15401 | 0.21076 | 0.15361 |
| LSTM | 15181 | 0.16731 | 0.15919 |
| Transformer | 22305 | 0.28713 | 0.72323 |

## Continual-learning probe (train A -> train B, re-measure A)

| model | A before | A after B | forgetting (lower=better) |
|---|---:|---:|---:|
| PCLiquidCore | 0.00393 | 2.75671 | +2.75278 |
| GRU | 0.00600 | 2.32507 | +2.31907 |
| LSTM | 0.00789 | 2.22890 | +2.22101 |
| Transformer | 0.01096 | 1.27162 | +1.26066 |
