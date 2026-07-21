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
- **可交付的证据门槛**：P0-1 多种子和 P0-2 20K 收敛训练已完成；P0-3 已补第一类强 baseline（modern Transformer），结果显示现代 Transformer 明显强于当前 MT-LNN；下一步继续补 Mamba/Mamba-2/GLA/DeltaNet、fp16 根因、scaling law、真实长上下文和效率曲线。
- 完整施工图见 `PUBLICATION_READINESS.md`（P0/P1/P2 清单）。

---

## 1. 当前状态

- 主仓库 `E:\M1`，分支 `physics-informed-head`（**公开** GitHub `everest-an/M1`）。
- 姊妹项目 `E:\O1` / `E:\O1-Anti`（**独立仓库，只读参考、绝不直接搬数**）。
- 本地 GPU：RTX 5060 Laptop **8GB**，torch 2.11.0+cu128，Python `E:\Python311`（`py -3.11`）。
- 远程训练目录：`/root/autodl-tmp/M1`（SSH：`root@tulong91.imwork.net -p 54511`）；结果已同步回本地 `E:\M1\scaling_fp32`。

> **分工现状（2026-07-19）**：本地 8GB 能做的高价值项**已做完**（P0-4 fp16 根因修复、O(1) 证据扩展到 1M、文档全面更正）。剩余项**都需要 AutoDL/A100**：P0-3 剩余强 baseline（Mamba/Mamba-2/GLA/DeltaNet）、大预算验证摘要的 14.7%、scaling law 三规模、真实长上下文任务。本地不要再尝试 20K 级长跑——8GB + 会话中断 + 与 O1 抢卡，实测反复失败。

## 2. 已完成（真实验证过）

| 项 | 结果 / 位置 |
|---|---|
| **P0-1 2K 多种子** | mt_lnn **257.15±4.89** vs transformer **373.68±8.97**（n=3, fp32, 全 stable，mt_lnn 领先约 31%，约 11σ）。JSON 在 `scaling_fp32/train_*_s*.json` |
| **P0-2 20K 收敛多种子** | mt_lnn **88.93±0.33** vs transformer **94.14±0.78** val PPL（seeds 0,1,2；20,000 steps；fp32；全 stable；n=3）。mt_lnn 相对 transformer 平均 PPL 降低约 **5.5%**。JSON 在 `scaling_fp32/converge_probe/train_*_s*.json` |
| **P0-3 modern Transformer 强 baseline** | modern_transformer **78.86±0.25** val PPL（seeds 0,1,2；20,000 steps；fp32；全 stable；n=3；144.1M 参数）。结果强于 mt_lnn **88.93±0.33** 和 simple transformer **94.14±0.78**。JSON 在 `scaling_fp32/converge_probe/train_modern_transformer_s*.json` |
| **P0-2/P0-3 日志与汇总整理** | 已将日志统一放入 `scaling_fp32/converge_probe/`：`scaling_train_20000_mt_lnn.log`、`scaling_train_20000_transformer.log`、`scaling_train_20000_modern_transformer.log`；三模型汇总表：`scaling_fp32/converge_probe/scaling_train_20000_summary.txt` |
| **checkpoint/resume** | `benchmarks/scaling_comparison.py` 已加入 `--ckpt_every N` 与 `--resume/--no-resume`；默认每 500 step 保存 `model + optim + scaler + step + cursor + RNG`，中断后可从 `out_dir/checkpoints/` 恢复 |
| **P0 训练口径固定** | WikiText-103-raw-v1，GPT-2 tokenizer，vocab 50257，seq_len=512，batch=4，lr=3e-4，`--dtype fp32` 关闭 autocast/scaler |
| **P0-4 fp16 发散根因（已修复）** | 根因：`global_coherence.py` 的 `(Q@K)/scale` 在 d_head=64 累加**之后**才缩放 → 中间乘积 ~2e5 溢出 fp16(65504) → `Inf×0`(因果掩码) = NaN → sigmoid 污染整层。修复：`(Q/scale)@K`×4 处 + `_gate_energy` 用 where/fp32累加/clamp_min。**验证**：原失败的 2000 步配方跑满，`stable: true`，val PPL **257.91** vs 同配方 fp32 **257.48**（差 0.17%，在 ±4.89 种子方差内）→ fp16 已回到 fp32 同等质量。审计：其余注意力全用 SDPA，coherence 是唯一手写的。工具：`benchmarks/diagnose_fp16_divergence.py` |
| **O(1) 恒定内存证据扩展到 1M** | `--mode decode`：上下文 512→1,048,576（**增长 2048×**），ARR 携带状态**恒定 0.381 MB**，llama KV 线性增长到 **3,072 MB** → **8,063×**。ARR 为实测快照字节，KV 为精确解析式。**边界**：推理携带状态、仅无注意力 O 系列（非训练内存、非 hybrid）|

### P0-2/P0-3 20K 收敛结果

```text
========================================================================
SCALING TRAIN | WikiText-103 | 20000 steps | seeds [0, 1, 2]
Protocol: GPT-2 tokenizer | seq_len=512 | batch=4 | lr=3e-4 | fp32
========================================================================
arch                       params  stable     val_ppl (mean±std)   n
mt_lnn                126,041,819    True       88.93 ± 0.33      3
transformer           142,051,520    True       94.14 ± 0.78      3
modern_transformer    144,070,784    True       78.86 ± 0.25      3
========================================================================

Per-seed val_ppl:
mt_lnn               s0=89.2819, s1=88.8849, s2=88.6194
transformer          s0=94.6340, s1=94.5436, s2=93.2441
modern_transformer   s0=79.1465, s1=78.6632, s2=78.7693
```

说明：

- P0-2 已从原来的“seed 0 单跑”扩展为 **seeds 0,1,2 三种子完整结果**。
- mt_lnn 在同一 20K/fp32/WikiText-103 口径下稳定优于自建 simple transformer baseline（PPL 降低约 **5.5%**）。
- P0-3 的 modern_transformer 是 RoPE + RMSNorm + SwiGLU baseline，结果 **78.86±0.25**，比 simple transformer 低约 **16.2%**，比当前 mt_lnn 低约 **11.3%**。这说明原 simple baseline 偏弱，论文主张不能再写成“MT-LNN 在 PPL 质量上优于强 Transformer baseline”；当前更稳妥的主线应转向“质量差距待优化 + O(1) working memory / 长上下文效率优势”。
- O1 的 214.8 是 84M/3000步/AMP/不同项目口径，**只能作为参考锚，不能混入 M1 论文主表做直接对比**。

## 3. 进行中 / 卡点

- **P0-3 强 baseline 正在进行中**：modern_transformer 已完成；Mamba/Mamba-2/GLA/DeltaNet 等现代高效架构仍需继续跑。当前脚本已支持 `mamba`，但 Windows/无 CUDA kernel 环境的速度结果不能用于论文效率对比；强 baseline 建议继续在 Linux + CUDA kernel + A100/AutoDL 上跑。
- **P0-3 modern_transformer 阶段成果已推送**：`benchmarks/baselines.py` 增加 `ModernCausalTransformer`；`benchmarks/scaling_comparison.py` 增加 `modern_transformer` arch；`scaling_fp32/converge_probe/scaling_train_20000_summary.txt` 已更新为三模型对比；modern_transformer 三个 JSON 与三份标准化日志已上传到 `physics-informed-head`。接手时仍需先 `git status` 确认本地是否有新实验结果或远端同步差异。
- ~~**fp16/AMP 根因未解决**~~ → **已解决（2026-07-19）**：根因是 `mt_lnn/global_coherence.py` 的注意力缩放顺序 `(Q@K)/scale`——在 d_head=64 维累加**之后**才缩放，Q/K 增大后中间乘积 ~2e5 在矩阵乘内部溢出 fp16（上限 65504），产生的 Inf 与因果掩码零相遇触发 `Inf*0=NaN`，经 sigmoid 污染整层。修复：改为 `(Q/scale)@K`（4 处）+ `_gate_energy` 用 where 代替乘法、fp32 累加、`clamp_min(1e-6)` 替换在 fp16 下下溢成 0 的 `1e-9`。**验证**：原本第 875 步发散的同配方现已跑满 **2000 步** `stable: true`，val PPL **257.91**，与同配方 fp32 的 257.48 相差仅 0.17%（远小于 ±4.89 种子方差）——fp16 已恢复到 fp32 同等质量。审计确认：其余注意力实现均用 SDPA，coherence 是唯一手写的。诊断工具：`benchmarks/diagnose_fp16_divergence.py`。
- **scaling law 未完成**：还需要至少 3 个模型规模，统一 token budget、训练步数/样本量和 eval 口径，确认优势是否随规模保持。
- **长上下文证据仍需补齐**：O(1) working memory 的核心卖点需要 decode/profile/真实任务支撑，不能只靠 WikiText PPL。

## 4. 下一步（按优先级）

1. **当前主线任务：继续跑强 baseline**：P0-3 modern_transformer 阶段成果已推送；下一步优先补 `mamba`，随后补 Mamba-2/GLA/DeltaNet 或同类高效架构；注意 Windows Mamba 无 CUDA kernel，强 baseline 和效率曲线建议迁到 Linux/A100。
2. **继续归档新 baseline 结果**：Mamba/Mamba-2/GLA/DeltaNet 每跑完一个模型，都同步三 seed JSON、run.log/标准化日志和更新后的 `scaling_train_20000_summary.txt`；checkpoint `.pt` 仍不提交。
3. **更新结果文档和论文材料**（2026-07-19 已完成第一轮）：P0-2 三种子 + P0-3 modern_transformer 结果已写入 README/BENCHMARKS/RESULTS/中英文论文/中英文 deck，并已明确标注 modern_transformer 领先 MT-LNN 11.3%、2K 旧结论已撤回、O1 参考锚限制。
4. **⚠️ 验证论文摘要的 14.7% 主张**（新增，重要）：论文摘要/结论的「比同参数 Transformer 低 14.7% PPL」来自大预算（100K 步 A100）实验，但**几乎肯定也是对着同一个 simple-reference 弱基线测的**。已先加限定语（"vs simple-reference，非现代基线"）作为止血，但**需要在大预算下补一轮 `modern_transformer` 对照**才能确认这个头号主张是否成立。若同样反转，摘要必须重写。建议在 AutoDL 上与其他强 baseline 一起排队。
5. ~~**并行待办：做 fp16 诊断**~~ → **已完成（2026-07-19）**：根因定位 + 修复 + 2000 步验证 + 全仓库审计，见上表。剩余可选：在 AutoDL 上跑 fp16 20K 确认长程（本地已验证 2000 步且与 fp32 质量持平）。
6. **扩 scaling law**：至少 3 个参数规模，固定 tokenizer/data/seq_len/batch/token budget，输出均值±标准差和效率曲线。
7. **补真实长上下文实验**：用 decode state / memory profile / 长上下文任务证明 O(1) working memory 的实际价值。

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

# 远程 P0-3 modern_transformer（已完成）
ssh root@tulong91.imwork.net -p 54511
cd /root/autodl-tmp/M1
python benchmarks/scaling_comparison.py --mode train --steps 20000 \
  --seeds 0,1,2 \
  --archs modern_transformer \
  --dtype fp32 \
  --ckpt_every 500 --resume \
  --train_token_cap 50000000 \
  --out_dir /root/autodl-tmp/M1/scaling_fp32/p0_3_modern_transformer \
  2>&1 | tee /root/autodl-tmp/M1/scaling_fp32/p0_3_modern_transformer/run.log

# 远程下一步：跑 mamba（进行中/待完成）
cd /root/autodl-tmp/M1
python benchmarks/scaling_comparison.py --mode train --steps 20000 \
  --seeds 0,1,2 \
  --archs mamba \
  --dtype fp32 \
  --ckpt_every 500 --resume \
  --train_token_cap 50000000 \
  --out_dir /root/autodl-tmp/M1/scaling_fp32/p0_3_mamba \
  2>&1 | tee /root/autodl-tmp/M1/scaling_fp32/p0_3_mamba/run.log

# 查看 P0-2/P0-3 汇总
Get-Content E:\M1\scaling_fp32\converge_probe\scaling_train_20000_summary.txt

# 查 GPU 争抢
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## 7. 关键文件

- 训练脚本：`benchmarks/scaling_comparison.py`（`--mode train`，含 `--ckpt_every` / `--resume`）
- 强 baseline 代码：`benchmarks/baselines.py`（新增 `ModernCausalTransformer`：RoPE + RMSNorm + SwiGLU）
- P0-1 结果：`scaling_fp32/train_*_s*.json`
- P0-2/P0-3 结果：`scaling_fp32/converge_probe/train_mt_lnn_s0.json`、`train_mt_lnn_s1.json`、`train_mt_lnn_s2.json`、`train_transformer_s0.json`、`train_transformer_s1.json`、`train_transformer_s2.json`、`train_modern_transformer_s0.json`、`train_modern_transformer_s1.json`、`train_modern_transformer_s2.json`
- P0-2/P0-3 汇总：`scaling_fp32/converge_probe/scaling_train_20000_summary.txt`
- 标准化日志：`scaling_fp32/converge_probe/scaling_train_20000_mt_lnn.log`、`scaling_train_20000_transformer.log`、`scaling_train_20000_modern_transformer.log`
- checkpoint（不提交）：`scaling_fp32/converge_probe/checkpoints/*.pt`
- 施工图：`PUBLICATION_READINESS.md`
- 模型：`mt_lnn/model.py`、`mt_lnn/mt_lnn_layer.py`、`mt_lnn/mt_lnn_v2.py`
