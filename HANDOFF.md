# MT-LNN / M1 — 会话交接文档 (HANDOFF)

> 新会话开始时：**先读这份 HANDOFF.md**，再读 `PUBLICATION_READINESS.md`，即可无缝接续。
> 最后更新：2026-07-19 · 分支 `physics-informed-head`（公开 GitHub：`everest-an/M1`）

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
- **可交付的证据门槛**：补齐 P0（多种子 ✓ / 收敛训练 ✓ / 公平强 baseline / fp16 根因），做到 rebuttal-proof。
- **当前状态**：P0-1 多种子和 P0-2 20K 收敛训练已完成；下一步进入强 baseline、fp16 根因、scaling law 与长上下文证据补齐。
- 完整施工图见 `PUBLICATION_READINESS.md`（P0/P1/P2 清单）。

---

## 1. 当前状态

- 主仓库 `E:\M1`，分支 `physics-informed-head`（**公开** GitHub `everest-an/M1`）。
- 姊妹项目 `E:\O1` / `E:\O1-Anti`（**独立仓库，只读参考、绝不直接搬数**）。
- 本地 GPU：RTX 5060 Laptop **8GB**，torch 2.11.0+cu128，Python `E:\Python311`（`py -3.11`）。
- 远程训练目录：`~/autodl-tmp/M1`；本地已同步更新到 `E:\M1`。

## 2. 已完成（真实验证过）

| 项 | 结果 / 位置 |
|---|---|
| **P0-1 多种子** | mt_lnn **257.15±4.89** vs transformer **373.68±8.97**（n=3, fp32, 全 stable，mt_lnn 领先约 31%，约 11σ）。JSON 在 `scaling_fp32/train_*_s*.json` |
| **P0-2 20K 收敛训练** | mt_lnn **89.28** vs transformer **94.63** val PPL（seed=0, 20,000 steps, fp32, 全 stable）。mt_lnn 相对 transformer PPL 降低约 **5.66%**。JSON 在 `scaling_fp32/converge_probe/train_*_s0.json` |
| **checkpoint/resume** | `benchmarks/scaling_comparison.py` 已加入 `--ckpt_every N` 与 `--resume/--no-resume`；默认每 500 step 保存 `model + optim + scaler + step + cursor + RNG`，中断后可从 `out_dir/checkpoints/` 恢复 |
| **Mamba/Windows 口径** | mamba 使用 HF `MambaForCausalLM` + mamba.py 后端；Windows 无 CUDA kernel，训练会显著慢于 Linux CUDA kernel |
| **P0 训练口径固定** | WikiText-103-raw-v1，GPT-2 tokenizer，vocab 50257，seq_len=512，batch=4，lr=3e-4，`--dtype fp32` 关闭 autocast/scaler |

### P0-2 具体结果

命令口径：

```bash
python benchmarks/scaling_comparison.py \
  --mode train \
  --steps 20000 \
  --seeds 0 \
  --archs mt_lnn,transformer \
  --dtype fp32 \
  --ckpt_every 500 \
  --resume \
  --out_dir ~/autodl-tmp/M1/scaling_fp32/converge_probe
```

本地归档位置：

- `scaling_fp32/converge_probe/train_mt_lnn_s0.json`
- `scaling_fp32/converge_probe/train_transformer_s0.json`
- `scaling_fp32/converge_probe.log`

结果：

```text
mt_lnn:
  params:     126,041,819
  stable:     true
  steps:      20000
  final_loss: 4.6430840492248535
  val_ppl:    89.28189341868541

transformer:
  params:     142,051,520
  stable:     true
  steps:      20000
  final_loss: 4.691380500793457
  val_ppl:    94.63402377681449
```

说明：

- P0-2 目标原本是看 mt_lnn 长训练是否继续收敛，并以 O1 项目的 214.8 作为参考锚。
- 当前 P0-2 结果显示 mt_lnn 在同一 20K/fp32/WikiText-103 口径下仍优于自建 transformer baseline。
- O1 的 214.8 是 84M/3000步/AMP/不同项目口径，**只能作为参考锚，不能混入 M1 论文主表做直接对比**。

## 3. 进行中 / 卡点

- **P0-3/P0-4 强 baseline**：需要更公平的现代高效架构 baseline（Mamba-2/GLA/DeltaNet 等），多数需要 Linux + CUDA kernel，建议云 A100。
- **fp16/AMP 根因**：历史上 MT-LNN 在 fp16 AMP 下出现非有限 loss，需要定位是数值尺度、归一化、优化器状态、还是液态层动态导致。
- **scaling law**：需要 3 个规模以上模型，统一 token budget、训练口径和 eval 口径。
- **真实长上下文证据**：O(1) working memory 需要用 decode/profile/真实任务把优势讲清楚，避免只停留在训练 PPL。

## 4. 下一步（按优先级）

1. **提交归档**：把 `HANDOFF.md`、`benchmarks/scaling_comparison.py`、P0-2 两个 JSON 结果推送到 `physics-informed-head`；不要提交 checkpoint `.pt` 大文件。
2. **更新论文/deck/RESULTS**：把 P0-2 的 20K 收敛结果写入结果表，但注明 O1 214.8 只是参考锚。
3. **跑强 baseline**：在 Linux/A100 上补 Mamba-2/GLA/DeltaNet 或同类高效架构，保证参数量、token budget、训练口径尽量公平。
4. **补 fp16 诊断**：最小复现 fp16 发散，记录 loss scale、梯度范数、激活范围、NaN 首发层。
5. **扩 scaling law**：至少 3 个规模，统一 steps/token budget，确认 mt_lnn 优势是否随规模保持。

## 5. ⚠️ 要避免的坑（血泪教训）

1. **绝不信 sub-agent 的二手结论**：引用任何数据/文件/API 前必须亲自 Read/Grep/检查 JSON。
2. **绝不把 O1 的数字搬进 M1 论文主表**：O1（84M/3000步/AMP）和 M1（126M/fp32/20K）不同口径，混用会造成不严谨甚至学术风险。
3. **长跑必须 checkpoint/resume**：无 checkpoint 时机器睡眠/SSH 断开/进程重启都会导致当前 arch 从 0 重跑。现在 `scaling_comparison.py` 已支持每 N 步保存。
4. **不要提交 checkpoint `.pt`**：`scaling_fp32/converge_probe/checkpoints/*.pt` 单个文件可达 1GB+，只用于本地/服务器恢复，不进 GitHub。
5. **8GB GPU 装不下多个训练**：M1 P0-2 约 7GB，占用时不要并行 O1 或其他 GPU 训练。
6. **transformer baseline 硬编码 `n_heads=13`**：`--d_model` 必须能被 13 整除（用 832 或 104，别用 128）。
7. **公开仓库自曝短板**：`PUBLICATION_READINESS.md`、本 HANDOFF 都包含未完成项和风险，提交前确认可以公开。

## 6. 关键命令速查

```powershell
# 本地 P0-1 多种子（已完成）
py -3.11 benchmarks/scaling_comparison.py --mode train --steps 2000 `
  --seeds 0,1,2 --archs transformer,mt_lnn --dtype fp32 `
  --train_token_cap 50000000 --out_dir E:/M1/scaling_fp32

# 本地 P0-2 收敛（已完成；支持 checkpoint/resume）
py -3.11 benchmarks/scaling_comparison.py --mode train --steps 20000 `
  --seeds 0 --archs mt_lnn,transformer --dtype fp32 `
  --ckpt_every 500 --resume `
  --out_dir E:/M1/scaling_fp32/converge_probe

# 服务器 P0-2 收敛（已完成；支持 checkpoint/resume）
cd ~/autodl-tmp/M1
python benchmarks/scaling_comparison.py \
  --mode train \
  --steps 20000 \
  --seeds 0 \
  --archs mt_lnn,transformer \
  --dtype fp32 \
  --ckpt_every 500 \
  --resume \
  --out_dir ~/autodl-tmp/M1/scaling_fp32/converge_probe

# 查看 P0-2 结果
python - <<'PY'
import json, pathlib
base = pathlib.Path("scaling_fp32/converge_probe")
for f in ["train_mt_lnn_s0.json", "train_transformer_s0.json"]:
    r = json.load(open(base / f))
    print(f, "steps=", r["steps"], "stable=", r["stable"],
          "final_loss=", r["final_loss"], "val_ppl=", r["val_ppl"])
PY
```

## 7. 关键文件

- 训练脚本：`benchmarks/scaling_comparison.py`（`--mode train`，含 `--ckpt_every` / `--resume`）
- P0-1 结果：`scaling_fp32/train_*_s*.json`
- P0-2 结果：`scaling_fp32/converge_probe/train_mt_lnn_s0.json`、`scaling_fp32/converge_probe/train_transformer_s0.json`
- P0-2 日志：`scaling_fp32/converge_probe.log`
- checkpoint（不提交）：`scaling_fp32/converge_probe/checkpoints/*.pt`
- 施工图：`PUBLICATION_READINESS.md`
- 模型：`mt_lnn/model.py`、`mt_lnn/mt_lnn_layer.py`、`mt_lnn/mt_lnn_v2.py`
