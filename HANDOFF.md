# MT-LNN / M1 — 会话交接文档 (HANDOFF)

> 新会话开始时：**先读这份 HANDOFF.md**，再读 `PUBLICATION_READINESS.md`，即可无缝接续。
> 最后更新：2026-07-19 · 分支 `physics-informed-head`

---

## 0. 目标

**产品/研究方向**：MT-LNN（微管启发的液态神经网络）—— 替代 Transformer 的高效长上下文架构。
核心卖点：**O(1) 恒定工作记忆**（无 KV cache 膨胀）+ 13 通道微管液态层 + 每参数能力密度更高。

### 🎯 总目标（长期北极星）

让 MT-LNN 成为**高效长上下文架构的标杆** —— 「提到高效长上下文架构，绕不开 MT-LNN」。

- **技术上**：证明一个 O(1) 记忆的循环/液态架构，在同参数/同预算下质量-效率优于 Transformer 与主流高效注意力（Mamba/GLA/...），且能 scale。
- **落地上**：O(1) 恒定内存让长上下文能在**端侧/旧设备**跑，大幅砍掉长上下文的训练与推理成本；通过 adapter 嫁接到 Qwen/Llama 生态被采用。

### 🚩 近期目标（本阶段，3-6 个月）

**冲一篇 ICLR / NeurIPS / ICML 主会论文（Track A 工程/架构路线）**。

- **意识 / Φ / 麻醉验证降级**（争议大、证据站不住），聚焦 1-2 个证据充分的硬主张：质量-效率折中 + 长上下文。
- **可交付的证据门槛**：P0-1 多种子和 P0-2 20K 收敛训练已完成；下一步补公平强 baseline、fp16 根因、scaling law、真实长上下文和效率曲线。
- 完整施工图见 `PUBLICATION_READINESS.md`（P0/P1/P2 清单）。

---

## 1. 当前状态

- 主仓库 `E:\M1`，分支 `physics-informed-head`（**公开** GitHub `everest-an/M1`）。
- 姊妹项目 `E:\O1` / `E:\O1-Anti`（**独立仓库，只读参考、绝不直接搬数**）。
- 本地 GPU：RTX 5060 Laptop **8GB**，torch 2.11.0+cu128，Python `E:\Python311`（`py -3.11`）。
- 远程训练目录曾用 `~/autodl-tmp/M1`；结果已同步回本地 `E:\M1\scaling_fp32`。

## 2. 已完成（真实验证过）

| 项 | 结果 / 位置 |
|---|---|
| **P0-1 2K 多种子** | mt_lnn **257.15±4.89** vs transformer **373.68±8.97**（n=3, fp32, 全 stable，mt_lnn 领先约 31%，约 11σ）。JSON 在 `scaling_fp32/train_*_s*.json` |
| **P0-2 20K 收敛多种子** | mt_lnn **88.93±0.33** vs transformer **94.14±0.78** val PPL（seeds 0,1,2；20,000 steps；fp32；全 stable；n=3）。mt_lnn 相对 transformer 平均 PPL 降低约 **5.5%**。JSON 在 `scaling_fp32/converge_probe/train_*_s*.json` |
| **P0-2 合并日志与汇总** | 三 seed 训练日志：`scaling_fp32/converge_probe_s012.log`；汇总表：`scaling_fp32/converge_probe/scaling_train_20000_summary.txt` |
| **checkpoint/resume** | `benchmarks/scaling_comparison.py` 已加入 `--ckpt_every N` 与 `--resume/--no-resume`；默认每 500 step 保存 `model + optim + scaler + step + cursor + RNG`，中断后可从 `out_dir/checkpoints/` 恢复 |
| **P0 训练口径固定** | WikiText-103-raw-v1，GPT-2 tokenizer，vocab 50257，seq_len=512，batch=4，lr=3e-4，`--dtype fp32` 关闭 autocast/scaler |

### P0-2 20K 收敛结果

```text
========================================================================
SCALING TRAIN | WikiText-103 | 20000 steps | seeds [0, 1, 2]
arch               params  stable     val_ppl (mean±std)   n
mt_lnn        126,041,819    True       88.93 ± 0.33      3
transformer   142,051,520    True       94.14 ± 0.78      3
========================================================================

Per-seed val_ppl:
mt_lnn       s0=89.2819, s1=88.8849, s2=88.6194
transformer  s0=94.6340, s1=94.5436, s2=93.2441
```

说明：

- P0-2 已从原来的“seed 0 单跑”扩展为 **seeds 0,1,2 三种子完整结果**。
- mt_lnn 在同一 20K/fp32/WikiText-103 口径下稳定优于自建 transformer baseline。
- O1 的 214.8 是 84M/3000步/AMP/不同项目口径，**只能作为参考锚，不能混入 M1 论文主表做直接对比**。

## 3. 进行中 / 卡点

- **P0-3 强 baseline 正在进行中**：当前 transformer 是自建 simple baseline；论文级结论还需要 Mamba-2/GLA/DeltaNet 等现代高效架构对照，最好在 Linux + CUDA kernel + A100 上跑。
- **当前本地工作区有强 baseline 未提交改动**：`benchmarks/baselines.py`、`benchmarks/scaling_comparison.py` 正在改动中，接手时不要 `reset --hard` / `checkout` 覆盖，先 `git diff` 读清楚。
- **fp16/AMP 根因未解决**：历史上 MT-LNN 在 fp16 AMP 下出现非有限 loss，需要定位是液态层动态、归一化、激活尺度、梯度尺度还是优化器状态导致。
- **scaling law 未完成**：还需要至少 3 个模型规模，统一 token budget、训练步数/样本量和 eval 口径，确认优势是否随规模保持。
- **长上下文证据仍需补齐**：O(1) working memory 的核心卖点需要 decode/profile/真实任务支撑，不能只靠 WikiText PPL。

## 4. 下一步（按优先级）

1. **P0-2 归档已完成**：`HANDOFF.md`、`benchmarks/scaling_comparison.py`、seed 1/2 JSON、`scaling_train_20000_summary.txt`、`converge_probe_s012.log` 已推送到 `physics-informed-head`；checkpoint `.pt` 仍不提交。
2. **当前主线任务：跑强 baseline**：优先补 Mamba-2/GLA/DeltaNet 或同类高效架构；注意 Windows Mamba 无 CUDA kernel，强 baseline 建议迁到 Linux/A100。
3. **强 baseline 结束后更新结果文档和论文材料**：把 P0-2 三种子结果与强 baseline 结果写入 README/BENCHMARKS/RESULTS/论文草稿/deck，明确标注训练口径和 O1 参考锚限制。
4. **并行待办：做 fp16 诊断**：最小复现 fp16 发散，记录 loss scale、梯度范数、激活范围、NaN 首发层，并与 fp32 stable 结果对照。
5. **扩 scaling law**：至少 3 个参数规模，固定 tokenizer/data/seq_len/batch/token budget，输出均值±标准差和效率曲线。
6. **补真实长上下文实验**：用 decode state / memory profile / 长上下文任务证明 O(1) working memory 的实际价值。

## 5. ⚠️ 要避免的坑（血泪教训）

1. **绝不信二手结论**：引用任何数据/文件/API 前必须亲自 Read/Grep/检查 JSON；尤其是跨会话、跨项目的结果。
2. **绝不把 O1 的数字搬进 M1 论文主表**：O1（84M/3000步/AMP）和 M1（126M/fp32/20K）不同口径，混用会造成不严谨甚至学术风险。
3. **长跑必须 checkpoint/resume**：无 checkpoint 时机器睡眠/SSH 断开/进程重启都会导致当前 arch 从 0 重跑。现在 `scaling_comparison.py` 已支持每 N 步保存。
4. **不要提交 checkpoint `.pt`**：`scaling_fp32/converge_probe/checkpoints/*.pt` 单个文件可达 1GB+，只用于本地/服务器恢复，不进 GitHub。
5. **8GB GPU 不适合并行训练**：M1 P0-2 约 7GB，占用时不要并行 O1 或其他 GPU 训练。
6. **transformer baseline 硬编码 `n_heads=13`**：`--d_model` 必须能被 13 整除（用 832 或 104，别用 128）。
7. **公开仓库自曝短板**：`PUBLICATION_READINESS.md`、本 HANDOFF 都包含未完成项和风险，提交前确认可以公开。

## 6. 关键命令速查

```powershell
# 本地 P0-1 2K 多种子（已完成）
py -3.11 benchmarks/scaling_comparison.py --mode train --steps 2000 `
  --seeds 0,1,2 --archs transformer,mt_lnn --dtype fp32 `
  --train_token_cap 50000000 --out_dir E:/M1/scaling_fp32

# 本地 P0-2 20K 收敛（已完成；支持 checkpoint/resume）
py -3.11 benchmarks/scaling_comparison.py --mode train --steps 20000 `
  --seeds 0,1,2 --archs mt_lnn,transformer --dtype fp32 `
  --ckpt_every 500 --resume `
  --out_dir E:/M1/scaling_fp32/converge_probe

# 查看 P0-2 汇总
Get-Content E:\M1\scaling_fp32\converge_probe\scaling_train_20000_summary.txt

# 查 GPU 争抢
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## 7. 关键文件

- 训练脚本：`benchmarks/scaling_comparison.py`（`--mode train`，含 `--ckpt_every` / `--resume`）
- P0-1 结果：`scaling_fp32/train_*_s*.json`
- P0-2 结果：`scaling_fp32/converge_probe/train_mt_lnn_s0.json`、`train_mt_lnn_s1.json`、`train_mt_lnn_s2.json`、`train_transformer_s0.json`、`train_transformer_s1.json`、`train_transformer_s2.json`
- P0-2 汇总：`scaling_fp32/converge_probe/scaling_train_20000_summary.txt`
- P0-2 合并日志：`scaling_fp32/converge_probe_s012.log`
- checkpoint（不提交）：`scaling_fp32/converge_probe/checkpoints/*.pt`
- 施工图：`PUBLICATION_READINESS.md`
- 模型：`mt_lnn/model.py`、`mt_lnn/mt_lnn_layer.py`、`mt_lnn/mt_lnn_v2.py`
