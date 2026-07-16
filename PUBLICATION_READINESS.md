# MT-LNN Publication Readiness — Track A (Systems / Architecture)

> ⚠️ **内部文档，勿推公开仓库。** 本文件逐条列出当前论文的弱点与未完成实验，
> `github.com/everest-an/M1` 为**公开**仓库，公开这些内容等于向审稿人/竞品自曝短板。
> 建议保留在本地、或迁入私有内部文档系统。默认已排除出 git（见文末）。
>
> 最后更新：2026-07-16 · 路线：**A（工程/架构）** · 目标场馆：ICLR / NeurIPS / ICML 主会

---

## 0. 定位（Track A 的核心取舍）

把论文重构为一篇**扎实的高效长上下文架构**论文，而不是"意识 AI"论文。

- **主张收敛到 1–2 个**：`O(1)` 循环状态 + 多尺度液态动力学，在**同参数/同数据/同预算**下相对现代高效序列模型的**质量-效率折中优势**。
- **意识 / Φ / 麻醉验证降级**：从主结果移出，最多作为 exploratory appendix，且**全程不用 "consciousness" 一词**，Φ 改称"信息整合的工程代理指标（information-integration proxy）"。RESULTS.md 中已 retract/inert 的条目不得回流正文。
- **Abstract 重写**：现版塞了 14.7% PPL / 2.2× Φ / 麻醉 / O(1) / 可解释性 五个半成品主张 → 收敛为"新架构 + 质量-效率优势 + 长上下文能力"，每个都要有充分证据背书。

---

## 1. P0 — Desk-reject killers（不补必被拒）

| # | 项 | 现状 | 目标（验收标准） | 复用/涉及 |
|---|----|------|------------------|-----------|
| P0-1 | **多种子 + 误差棒** | n=1 | 每个数字 **n≥3**，报 mean±std，主对比做显著性检验（paired bootstrap / t-test） | `benchmarks/scaling_comparison.py`（已支持 `--seeds`，扩到 0,1,2）|
| P0-2 | **训练到收敛** | 2000 步欠训练，PPL 257/370（远高于同规模 SOTA ~20–30） | 训到 loss 平台；报**充分训练**下的对比，不用 undertrained 数字 | `scaling_comparison.py` `--steps` 拉满 + LR schedule |
| P0-3 | **公平强 baseline** | 自建 "simple Transformer" + **宽度不匹配**的 Mamba | 同参数/数据/预算、**各自调优**的：现代 Transformer(RoPE+SwiGLU)、**架构匹配**的 Mamba-2、GLA、DeltaNet、RWKV-7 | `benchmarks/baselines.py` + `compare_baselines.py`（扩 baseline 家族）|
| P0-4 | **fp16 发散根因** | mt_lnn 第 629 步发散（open bug） | 根因修复，或给出稳定混合精度方案（选择性 fp32 层 / loss scaling）并证明 2000+ 步稳定 | 复现脚本 + 定位到具体层/算子 |

> P0 全绿之前，连扎实的 workshop paper 都偏紧。

---

## 2. P1 — Competitiveness（决定能不能中）

| # | 项 | 现状 | 目标 | 复用/涉及 |
|---|----|------|------|-----------|
| P1-1 | **Scaling law** | 单点 125M | ≥3 规模（125M/350M/1.3B）画曲线，证明优势**随规模保持/扩大** | `scaling_comparison.py` 多 `--d_model/--n_layers` |
| P1-2 | **现代 benchmark** | WikiText-103 word-PPL（过时） | The Pile / SlimPajama / FineWeb 子集报 **bits-per-byte** + 下游 zero-shot（LAMBADA/PIQA/HellaSwag/ARC） | `benchmarks/capability_eval.py` 扩数据源 |
| P1-3 | **真实长上下文** | 200K 参数沙盒 needle | 真 benchmark：PG19 / RULER / 真实 needle-in-haystack / 长文档 QA，**声称 128K 就要有 128K 数据** | `long_context.py`、`long_context_kv_vs_state.py`、`selective_copy.py` |
| P1-4 | **完整消融** | 部分组件已有 ablation | 13 通道数 / 多尺度 τ / GWT bottleneck / skip-gating / predictive-coding loss 各单独 on-off | `o1_module_ablation.py`、`sparse_resonance_ablation.py`、`cross_layer_gate_sharing_ablation.py`、`attribution_ablation.py` |
| P1-5 | **真实效率曲线** | `mamba.py` 纯 PyTorch 慢实现（1200 tok/s，不可代表 Mamba） | throughput/显存 vs 序列长度实测，对手用 **FlashAttention / Mamba CUDA kernel**（换 Linux + mamba-ssm） | `operator_compression_report.py`、`length_streaming_eval.py` |

> ⚠️ **效率对比红线**：绝不能用本地 Windows 上的 `mamba.py` 慢实现代表 Mamba 速度——一眼被拆穿。效率数必须在装了 CUDA kernel 的环境跑。

---

## 3. P2 — Nice-to-have（加分项）

- **理论**：pscan 复杂度证明、`O(1)` 记忆的容量/表达力分析、与 SSM 的形式化关系。顶会看重理论深度。
- **可复现**：确定性训练脚本 + 完整超参/seed/硬件 + 锁定依赖（已有 code+weights 开源，是实打实加分）。
- **相关工作更新**：补 2024–2025 高效注意力（Mamba-2、GLA、DeltaNet、Titans、RWKV-7）；当前引用偏旧。

---

## 4. 现有资产映射（很多是"扩展"而非"从零建"）

已具备的脚手架，直接复用：
- 训练/对比：`scaling_comparison.py`（多 arch/seed/dtype，本轮已加 mamba.py 后端）、`baselines.py`、`compare_baselines.py`、`run_all.py`、`run_benchmark.py`
- 长上下文/记忆：`long_context.py`、`long_context_kv_vs_state.py`、`selective_copy.py`、`cross_window_recall.py`、`cross_session_recall.py`、`memory_recall_validation.py`、`state_carry_train.py`、`state_only_streaming.py`、`weight_consolidation.py`
- 消融：`o1_module_ablation.py`、`sparse_resonance_ablation.py`、`cross_layer_gate_sharing_ablation.py`、`attribution_ablation.py`
- 评估：`capability_eval.py`、`hallucination_eval.py`、`length_streaming_eval.py`、`physics_rollout_eval.py`、`wikitext_recall_validation.py`
- 效率：`operator_compression_report.py`

**缺口**（需新建/大改）：架构匹配的 Mamba-2/GLA/DeltaNet/RWKV baseline 封装；bits-per-byte + 下游 zero-shot harness；RULER/PG19 接入；CUDA-kernel 环境下的效率基准；fp16 稳定性修复。

---

## 5. 建议里程碑

- **M1（可投 workshop / 打底）**：P0-1 + P0-2 + P0-4，加 P0-3 的至少 2 个匹配 baseline。产出：多种子、收敛、稳定训练的干净对比。
- **M2（主会竞争力）**：P0-3 补齐 baseline 家族 + P1-1 scaling law + P1-2 现代 benchmark + P1-4 消融。
- **M3（冲击/rebuttal-proof）**：P1-3 真实长上下文 + P1-5 真实效率曲线 + P2 理论/相关工作 + abstract/定位重写。

---

## 6. 投稿前一页纸 checklist

- [ ] 所有主结果 n≥3 + 误差棒 + 显著性
- [ ] 训练收敛，非 undertrained 数字
- [ ] baseline 现代、调优、架构匹配
- [ ] scaling law ≥3 规模
- [ ] 现代语料 + 下游 zero-shot
- [ ] 真实长上下文（声称多长测多长）
- [ ] 每个组件有消融
- [ ] 效率曲线在 CUDA-kernel 环境实测
- [ ] fp16/混合精度稳定或有明确方案
- [ ] abstract 聚焦 1–2 主张，意识/Φ 降级或移除
- [ ] 相关工作覆盖 2024–2025
- [ ] code+weights+确定性脚本可复现

---

## 7. Git 处置

本文件默认**不入库**（自曝弱点 + 公开仓库）。若确认要本地留存又防误提交，在 `.gitignore` 加：

```
/PUBLICATION_READINESS.md
```

如需团队协作，迁到私有文档系统而非 public M1。
