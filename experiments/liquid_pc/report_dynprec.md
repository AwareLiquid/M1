# PC-Liquid-Core validation report

Config: `{'n_seq': 320, 'seq_len': 96, 'd_in': 1, 'epochs': 40, 'batch': 64, 'lr': 0.003, 'prime': 48, 'seed': 0, 'ss_max': 0.0, 'pc_kwargs': {'dynamic_precision': True}}`  |  runtime 168.3s

Persistence reference (predict x_t+1=x_t) 1-step MSE: **0.32812**

## Main task (multi-timescale signal)

| model | #params | 1-step MSE | rollout MSE |
|---|---:|---:|---:|
| PCLiquidCore | 14506 | 0.01478 | 0.13886 |
| GRU | 15401 | 0.02599 | 0.83247 |
| LSTM | 15181 | 0.03187 | 0.01451 |
| Transformer | 22305 | 0.28462 | 0.78906 |

## Continual-learning probe (train A -> train B, re-measure A)

| model | A before | A after B | forgetting (lower=better) |
|---|---:|---:|---:|
| PCLiquidCore | 0.00352 | 1.39618 | +1.39267 |
| GRU | 0.00609 | 2.60961 | +2.60352 |
| LSTM | 0.00868 | 2.24637 | +2.23769 |
| Transformer | 0.00990 | 1.52230 | +1.51240 |
