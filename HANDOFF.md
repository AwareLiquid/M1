# MT-LNN / M1 — 会话交接文档 (HANDOFF)

> 新会话开始时：**先读这份 HANDOFF.md**，再读 `PUBLICATION_READINESS.md`，即可无缝接续。
> 最后更新：2026-07-19 · 分支 `physics-informed-head`（= `main`，已追平）

---

## 0. 目标（北极星 + 路线）

**产品/研究方向**：MT-LNN（微管启发的液态神经网络）—— 替代 Transformer 的高效长上下文架构。
核心卖点：**O(1) 恒定工作记忆**（无 KV cache 膨胀）+ 13 通道微管液态层 + 每参数能力密度更高。

**目标（已定 Track A）**：
- 走**工程/架构路线**，冲 **ICLR / NeurIPS / ICML 主会**（不是 Nature/意识路线）。
- **意识 / Φ / 麻醉验证降级**（争议大、证据站不住），聚焦 1–2 个硬主张：质量-效率折中 + 长上下文。
- **北极星一句话**：「提到高效长上下文架构，绕不开 MT-LNN」。
- 施工图见 `PUBLICATION_READINESS.md`（P0/P1/P2 全清单）。

---

## 1. 当前状态

- 主仓库 `E:\M1`，分支 `physics-informed-head`（**公开** GitHub `everest-an/M1`），`main` 已 FF 追平。
- 姊妹项目 `E:\O1` / `E:\O1-Anti`（**独立仓库，别人的会话在跑，只读参考、绝不改**）。
- 本地 GPU：RTX 5060 Laptop **8GB**，torch 2.11.0+cu128，Python `E:\Python311`（`py -3.11`）。

## 2. 已完成（真实验证过）

| 项 | 结果 / 位置 |
|---|---|
| **P0-1 多种子** | mt_lnn **257.15±4.89** vs transformer **373.68±8.97** vs mamba 414（n=3, fp32, 全稳定, 领先31%≈11σ）。JSON 在 `scaling_fp32/train_*_s*.json` |
| **文档更新+推送** | README/BENCHMARKS/RESULTS/中英文论文/中英文 deck + 4 份 PDF |
| **checkpoint/resume** | `scaling_comparison.py` 加 `--ckpt_every N`，CPU+GPU 双验证（commit 83bb5e0） |
| **NaN 修复抢救** | delta 写规则未归一化 key 爆 NaN 的修复 cherry-pick 进主线（commit 152a3a2，原作者 everest-an） |
| **main 收敛** | `main` FF 到 physics（152a3a2），两线曾分叉，现已合 |
| **mamba Windows 修复** | mamba.py 后端（commit e5d034b） |

## 3. 进行中 / 卡点

- **P0-2 收敛训练**：mt_lnn 跑 20K 步（对标 O1 的 214.8）。用**交互式计划任务** `M1_converge` 启动。
  - 续跑命令：`schtasks /run /tn "M1_converge"`（从最近 checkpoint 恢复）
  - checkpoint：`scaling_fp32\converge_probe\ckpt_mt_lnn_s0.pt`，每 500 步原子更新
- **GPU 争抢**：与 O1 侧实验（needle/pytest/diagnose）共享 8GB，会互拖甚至挤死。O1 优先，M1 排队。

## 4. 下一步（按优先级）

1. **P0-2 完成** → 拿 mt_lnn 收敛 PPL（对标 214.8）。
2. **分支归档**（可选，未做）：`mtp-seam-wireup`（null 结果）、`exp/learn-tau`（落后62，过期）建议 tag 后删；`cleanup-dead-symbols`（删死码）需先查 physics 是否还引用 `quantum_coupling` 再决定 cherry-pick。删远程分支前**必须先打 tag 保命**，且注意可能有协作者（Codex [CX]）在用。
3. **P0-3/P0-4 + P1**（多数需**云 A100**）：公平强 baseline（Mamba-2/GLA/DeltaNet，需 Linux+CUDA kernel）、fp16 发散根因、scaling law（3 规模）、现代 benchmark、真实长上下文、消融、效率曲线。
4. 数据都靠谱后：更新论文/deck 数据 + nature-skill 改论点论据。

## 5. ⚠️ 要避免的坑（血泪教训）

1. **绝不信 sub-agent 的二手结论**：本会话两个 agent **幻觉**了「O1 有 ETT/PhysioNet/irregular 时序脚本和数字」——真查文件系统发现**脚本和结果全不存在**。**引用任何数据/文件/API 前必须亲自 Read/Grep 确认**（CLAUDE.md 反幻觉铁律）。
2. **绝不把 O1 的数字搬进 M1 论文**：O1（84M/3000步/AMP）和 M1（126M/fp32）不同口径，混用=学术不端。要「汇流」只能移植代码在 M1 重跑产出 M1 数据。
3. **长跑进程会死，根因是编排不是睡眠**：`Start-Process` 满速但绑 Claude 会话、重启即死；`schtasks` 无 `/IT`（session-0）拿不到 GPU 只能 CPU 龟速（28 tok/s）。**唯一可靠**：`schtasks` **带 `/IT`**（交互式）→ 满速 1200 tok/s + 抗会话重启。P0-2 每次死在 tokenize→训练交界、500步前（还没存档），所以加了 checkpoint + 50M token cap（tokenize 83s→22s，压缩脆弱窗口）。
4. **8GB 装不下两个训练**：M1 P0-2(~7GB) + O1 任一 GPU 活 = OOM 或龟速。检查争抢用 `nvidia-smi --query-compute-apps`。
5. **Bash 嵌套 PowerShell 输出常乱码**：查进程/GPU 用纯 PowerShell（`Get-CimInstance` / `nvidia-smi`），别用 Bash 套 powershell。
6. **transformer baseline 硬编码 `n_heads=13`**：`--d_model` 必须能被 13 整除（用 832 或 104，别用 128）。
7. **公开仓库自曝短板**：`PUBLICATION_READINESS.md`、本 HANDOFF 都列了弱点，M1 是 public repo，提交前想清楚（用户已知情选择公开 readiness）。

## 6. 关键命令速查

```powershell
# 续跑 P0-2（从 checkpoint 恢复）
schtasks /run /tn "M1_converge"
# 查 P0-2 进度
Get-Content 'E:\M1\scaling_fp32\converge_probe.log' -Tail 3
# 查 GPU 争抢
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 多种子（已完成，resume-safe 会跳过已有 seed）
schtasks /run /tn "M1_multiseed"
```

## 7. 关键文件

- 训练脚本：`benchmarks/scaling_comparison.py`（`--mode train`，含 `--ckpt_every`）
- 运行脚本：`scaling_fp32/converge_p2.cmd`（P0-2）、`run_multiseed.cmd`（P0-1）
- 结果：`scaling_fp32/*.json`（gitignored）
- 施工图：`PUBLICATION_READINESS.md`
- 模型：`mt_lnn/model.py`、`mt_lnn/mt_lnn_layer.py`（ProtofilamentLTC 已带 dt）、`mt_lnn/mt_lnn_v2.py`（delta 记忆，NaN 已修）
