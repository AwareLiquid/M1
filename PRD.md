# AwareLiquid (MT-LNN) — Product Requirements Document

**Version:** 3.0
**Date:** 2026-08-28
**Status:** Active
**Repo:** https://github.com/everest-an/M1
**Supersedes:** v2.2 (2026-06-06)。v2.2 的头部指标（"−28.5%/−27.7%/−34.4% adapter PPL"、
"+13.3% 云注入"）已全部撤回或重新定性（见 RESULTS.md 撤回记录与 §12）；v3.0 按
2026-08-15 战略重构（docs/DEVELOPMENT_PLAN.md）重写，主张体系从"推理 UX 产品"
转为 **S1/S2/S3 三条结构化资产主张 + 端侧/可审计产品线**。

> **证据政策（继承且加严）**：本文每个数字必须可追溯到 RESULTS.md / BENCHMARKS.md /
> ABLATIONS.md / `benchmarks/results/*.jsonl` 的可复现记录，且引用前必须过
> `benchmarks/experiment_protocol.py::publishable` 的 E0 门槛（≥3 seeds、非双峰、
> 配对检验 p<0.05）。RESULTS.md 仍是数字的最终事实源；本文与它冲突时以 RESULTS.md
> 为准。

---

## 0. 自 v2.2 以来发生了什么

| 时间 | 事件 |
|---|---|
| 2026-06-06 → 07 | v2.2 的 adapter PPL 主张被消融推翻（MT adapter 被冻结、只训了 LoRA，增益 ≈0）；混合 O(1) 与长上下文 LM 增益双双测为 null |
| 2026-07 | P0-2/P0-3 收敛实验定案：20K 步 3 seeds 下 modern Transformer（78.86±0.25）**领先** MT-LNN（88.93±0.33）11.3% —— 原生 LM 质量赛道正式放弃 |
| 2026-07-28 | M2 路线图（docs/ROADMAP_M2.md）：知识外置 RAG + 推理引擎定位 |
| 2026-08-15 | **战略重构**（docs/DEVELOPMENT_PLAN.md）：放弃 LM 质量赛道，主攻三条 Transformer 结构上无法进入的资产（S1/S2/S3） |
| 2026-08-15~17 | E1/E2/E5 系列加固：parity 分离 6 seeds Fisher p=0.0022；exp 参数化根因闭环；外推极限 = 训练长度×8 规律确认 |
| 2026-08-24 | 2B 基座 8192 课程训练完成，bf16 教训（训练动力学 3-7.5× PPL 恶化）定位并修复 |
| 2026-08-28 | 部署收口：液态裸 Parameter weight-only int8 量化打通（2B 7.6GB→~1.9GB）、serve 接入 RAG 检索层与 API 鉴权、lean core（研究脚手架移出默认 import 链）、E0 实验协议固化 |

## 1. 产品概述

**AwareLiquid / MT-LNN 是什么**：一个研究驱动的液态循环架构家族 + 围绕它生长的
轻量部署产品线。架构核心是把 Transformer 的注意力（部分或全部）替换为带
selective_decay 的液态循环核心，换取三种 Transformer 结构上拿不到的性质：

1. **O(1) 恒定推理状态**（O 系列，无注意力）—— KV cache 随上下文线性膨胀，
   我们恒定 0.381 MB；
2. **跨窗口/跨会话联想记忆**（fast-weight，M 系列 adapter）—— KV cache 一丢，
   attention 结构性记不住；我们的快照/恢复 bit-exact；
3. **抗不规则采样**（液态时间常数 τ）—— 实测丢采样 80% 时退化 +7.7%，
   LSTM/GRU +31~33%。

**一句话定位**：不做"更好的通用 LLM"，做"Transformer 进不去的场景"——端侧长流式、
跨会话记忆、工业传感器——加上可审计的部署表面。

## 2. 痛点与定位（诚实版）

| Transformer 痛点 | 我们的回答 | 边界（必须同框呈现） |
|---|---|---|
| KV cache O(T) 膨胀，长上下文按 A100 计价 | O 系列 ARR：0.381 MB 恒定，1M 处小 8063×（fp16 GQA1）/ **8567×（最强压缩 2-bit+GQA8）** | 仅推理携带状态；训练内存无此优势；out-of-window LM 质量为 null；小窗驱逐+2bit（0.202 MB）更小但窗口外召回 0.000（docs/KV_FRONTIER.md） |
| 上下文窗口一丢什么都记不住 | fast-weight 状态跨窗口 recall 0.56 vs attention/LoRA 的结构性 0.000；快照/恢复 bit-exact | 离散 K→V 绑定，不是长上下文语言建模能力 |
| LM 质量 | **不是我们的主张**：modern Transformer 在同预算下领先我们 11.3% | PPL 只作训练稳定性佐证 |
| 端侧/边缘部署 | int8 量化（nn.Linear 动态零退化 + 液态 Int8Weight）、ONNX int8 1.27MB、流式 step 图导出 | 2B 全 int8 部署数学已打通，权重未达标待复训 |

## 3. 目标与非目标

### Track A — 论文主张（P0，冲 ICLR/NeurIPS/ICML）

- **S1 电路表达力分离**：selective_decay 让液态核心逃出 TC⁰（对角/输入无关转移
  的理论上限）。已证：d16 parity 6 seeds Fisher **p=0.0022**；d32 12K 步预算墙突破
  （selective 5/6 grok vs stock 0/6）；E5d/E5e 根因闭环（**exp 参数化**决定长度
  外推，d16/d32 回归 6/6）。待办：真实算法任务桥接（E2/A5 三次非对角尝试判负，
  决胜计划见 docs/NONDIAGONAL_TRANSITION.md）。
- **S2 O(1) 状态效率**：0.381 MB 恒定、2048× 上下文增长状态逐字节不变；对最强压缩 KV
  （2-bit+GQA8）128k/1M 处仍小 1070.9×/8567×，全部未驱逐交叉点 T*∈[17,976]（账本+
  双真模型实测，docs/KV_FRONTIER.md；唯一反超 = 小窗驱逐+2bit，代价窗口外召回 0.000）；
  外推极限 = 训练长度×8（512 训练 → 16K 平稳）；2B curriculum 8192 已跑完。
- **S3 跨会话持久记忆**：0.56 vs 0.000、bit-exact 快照；**fast-weight 核心化钩子
  已落地**（`config.fast_weight_core`，低秩因果外积记忆，默认 off 位等价，
  RNG 隔离保证干净 A/B）——验收门 recall≥0.56 且 PPL 不退，待 GPU 实验。

### Track B — 产品线（P1，与论文并行）

- **B1 serve 推理栈**：FastAPI 本地推理服务（completions/流式/会话持久化/睡眠固化/
  RAG 检索增强/API 鉴权/限流），Docker 化，官网 demo 后端。**状态：核心已上线，
  RAG/鉴权/量化三项 2026-08-28 落地待部署。**
- **B2 量化部署**：`QUANTIZE_INT8=1` 一键 weight-only int8。实测 nn.Linear 动态量化
  PPL 退化 +0.0%、内存 −62%（514→196 MB）；液态裸 Parameter 经 Int8Weight 覆盖；
  2B fp32 7.6GB → 全 int8 ~1.9GB（进 7.2GB VPS 的数学成立，PPL 退化待训练机实测）。
- **B3 端侧 O 系列**：ONNX 单帧 step 图导出（O1-Sound 5.03MB fp32 / 1.27MB int8，
  携带状态 5,120 B 不随流长增长）；浏览器 WebGPU 路径已验证（被 O1 48M 权重失踪
  阻塞）；**O1-Sound 训练未跑**（官网标 untrained）。
- **B4 工业传感器 demo**：NASA 电池 SoH——整颗 B0018 留出，RMSE **0.1038**
  （55K 参数）vs LSTM 0.1171（99K）；不规则采样唯一真差异化（80% 丢弃 +7.7% vs
  LSTM +31.1% / GRU +32.8%，t 检验显著）；流式状态 2.6KB 恒定。CPU 4 分钟可复现。
- **B5 官网与发布管线**：awareliquid.ai 已上线；128M 已发布 HF；
  2B 发布管线脚手架就绪（校验→Release→HF→官网，权重未达标暂缓）。

### 非目标

- **NG1** 不承诺 LM 质量 SOTA（modern Transformer 领先 11.3% 是测量事实）。
- **NG2** 不承诺"全面对标 70B"（M2 只在数学/代码推理、长流式记忆、持续学习、
  端侧延迟四轴做 head-to-head）。
- **NG3** 意识 / Φ / 麻醉验证主张已降级为研究脚手架（AVP 失败、Φ̂ 符号反转），
  相关 21 个模块已移出默认 import 链。
- **NG4** 不做托管 SaaS 与 RLHF 对齐（交付开源权重/代码/工件）。

## 4. 用户与画像

| 画像 | 需要什么 | 对应产品线 |
|---|---|---|
| 端侧/嵌入式工程师（电池、耳机、车机、传感器） | 恒定内存、抗丢采样、小模型、CPU 可跑 | B3/B4 |
| 合规敏感企业（金融/法务文档 QA） | 每条事实可溯源、本地优先、低 TCO | B1+RAG、M2 适配器线（AwareLiquid-M2 独立仓库） |
| AI×神经科学研究者 | 可复现的架构证据、负结果、诚实协议 | Track A 论文 + E0 协议开源 |
| 开源 AI 工程师 | 一条命令可复现的实验与部署 | 全部（`quant_smoke.py`/`reasoning_depth.py`/serve 均即取即用） |

## 5. 功能需求

### F1 — serve 推理栈（B1，P0）

| 子项 | 需求 | 状态 |
|---|---|---|
| 生成端点 | completions / stream(SSE) / 会话持久化 / 睡眠固化 | ✅ 已上线 |
| **RAG 检索增强** | `RAG_INDEX` 加载 BM25 索引；请求 `"rag": true` 前缀检索上下文；`/v1/rag/search` 检索即服务；coverage 是收益上限（H_P1：big+rag 事实 EM 7.2×） | ✅ 代码就绪（2026-08-28），待部署 |
| **鉴权与限流** | `API_AUTH_MODE=off/soft/strict`；strict 零可用 key 拒绝启动；匿名滑动窗口限流 | ✅ 同上 |
| **有界等待** | 生成锁 GEN_LOCK_TIMEOUT=300s + GEN_QUEUE_CAP=8，超限 503 | ✅ 同上 |
| 健壮性 | counts.json 原子写；`/v1/model` schema 与 server_hf 对齐（base_model/adapter_loaded/is_baseline/quantization） | ✅ 同上 |

### F2 — 量化部署（B2，P0）

| 子项 | 需求 | 状态 |
|---|---|---|
| nn.Linear 动态 int8 | PPL 退化 <1%，内存 −60% 级 | ✅ 实测 +0.0% / −62%（128M） |
| 液态裸 Parameter 量化 | Int8Weight（int8 存储+惰性反量化），tau/decay 动力学路径强制 fp32 | ✅ 落地+13 测试；PPL 退化待训练机实测 |
| 引擎容错 | qnnpack 1×1 prepack 不支持 → 逐模块降级；无 QEngine → 只量化液态 | ✅ |
| serve 集成 | `QUANTIZE_INT8=1` 启动量化，`/v1/model` 上报 | ✅ |

### F3 — 端侧 O 系列（B3，P1）

ONNX 单帧 step 图 + int8（1.27MB）+ 恒定状态；WebGPU 浏览器推理路径已验证；
**阻塞项：O1 48M 权重找回（Modal workspace/服务器）、O1-Sound 训练未跑**
（两者完成后官网两块灰框转 Measured）。

### F4 — 工业传感器 demo（B4，P1）

电池 SoH demo 已可复现（训练→跨电芯预测→曲线，CPU 4 分钟）；下一步：接入官网
案例区与客户材料，扩展第二个数据集（Bearing/PHM）验证"抗不规则采样"的普适性。

### F5 — fast-weight 记忆核心化（S3 落地，P1）

`CoreFastWeight`：每层第二记忆通道（与 KV cache 平级），低秩因果外积记忆，
写/读 O(D·r)；零门控（开启未训练位等价）；RNG 隔离（同 seed A/B trunk 逐位相同）。
**验收门（G-1a）**：跨会话记忆基准 recall≥0.56 且 LM PPL 不退（≥3 seeds，
E0 协议）；跨会话 (F,z) 持久化（snapshot 进 LayerCache）为下一步。

### F6 — selective_decay 主张（S1 落地，P0）

E1-E5 证据链已入档（ABLATIONS.md）；待办按 DEVELOPMENT_PLAN：E4 大 k 长预算
裁决（Kaggle）、E2/A5 决胜计划执行、E6 文本收益重测（`--good_recipe
--sanity_steps 2000`，好配方+sanity 门已就绪）。

### F7 — E0 实验协议（横切，P0）

`benchmarks/experiment_protocol.py`：≥3 seeds 门槛、grokking 双峰检测、精确
配对符号检验/Fisher（零依赖）；`publishable()` 为引用判据。**所有对外数字引用前
必须过此关**；协议 wrapper 作为"诚实实验协议"开源卖点（W4）。

### F8 — 论文（Track A 收口，P0）

三主张各一张图 + 负结果章节（深度不替代选择性、NDIT 三连判负、混合 O(1) null）
作为可信度资产；骨架 PAPER_SKELETON.md；W6 里程碑论文 v1。

### F9 — 2B 基座与发布（B5，P1）

1.9B hybrid（35 层 d2912），8192 curriculum 已完成但**权重未达标**；fp32 衰减
修复 + scheduler 修复已入 main，复训后走既有发布管线（verify→Release→HF→官网）。
训练提逘认证：`--lnn_broadcast`（128M 实测质量无差 + ~12% 快，2B 采用前复验一次）。

## 6. 验收标准（v3.0 头部指标，全部为已测量值）

| 指标 | 值 | 来源 |
|---|---|---|
| O(1) 携带状态（O 系列） | 0.381 MB 恒定；128k 1008×、1M 8063×（fp16 GQA1）；vs 最强压缩 2-bit+GQA8：1070.9×/8567×（T*≈119） | `benchmarks/results/` decode profile + kv_frontier 账本 + kv_measured 实测 |
| 跨窗口 recall | 0.56（3 seeds 0.62/0.43/0.62）vs attention/LoRA 0.000 | RESULTS.md |
| 快照恢复 | bit-exact，控制组 chance | 同上 |
| parity 分离（S1） | d16：6 seeds Fisher p=0.0022；d32 12K 步：5/6 vs 0/6 | ABLATIONS E1/E1-d32 |
| 长度外推（S2） | 外推极限 = 训练长度×8；exp 参数化 d16/d32 回归 6/6 | ABLATIONS E5d/E5e |
| 电池 SoH（B4） | RMSE 0.1038（55K 参数）vs LSTM 0.1171；80% 丢采样 +7.7% vs +31~33%（显著） | `benchmarks/battery_*.py` |
| 量化（B2） | Linear 动态 +0.0% PPL / −62% 内存（128M）；2B 7.6→~1.9GB 数学成立 | `scripts/quant_smoke.py` |
| 训练稳定性 | 全架构 20K 步 3 seeds 无 NaN；fp16 根因已修（257.91 vs fp32 257.48） | scaling_fp32/ |
| LM 质量（**非主张，仅记录**） | mt_lnn 88.93±0.33 vs modern 78.86±0.25（落后 11.3%） | converge_probe |
| 测试套件 | 1372 passed / 0 failed（CPU） | pytest |

## 7. 路线图

- **P0（进行中）**：E 系列收尾（E4/E2-决胜/E6）→ 论文 v1（W6 末）→ 效率曲线
  Pareto（W2）。
- **P1**：2B 复训达标 → 发布；量化 PPL 实测（训练机）；fast-weight 核心化决策门；
  O1 权重找回 + O1-Sound 训练；电池 demo 产品化。
- **P2（融资/算力后）**：M2 混合 2B（Samba/Jamba 式配比）+ GRPO 后训练 + 记忆
  控制器（surprise 门控 RAG 读写 + 睡眠蒸馏固化）；蒸馏优先于预训练
  （教师 API 预算待批）。

## 8. 架构快照（v3.0）

```
M 系列（混合, attention + 液态核心）      O 系列（ARR, 无注意力）
  serve: FastAPI + RAG + 鉴权 + int8       ONNX int8 单帧 step 图
  fast-weight 核心化钩子 (默认 off)        恒定状态 (0.381MB / 5KB 级)
  2B: 35L × d2912, 8192 curriculum         O1-Sound / 浏览器 WebGPU
实验协议: E0 (≥3 seeds + 双峰检测 + 精确检验) — 引用判据
训练栈: 深监督(泊松+逐迭代CE) / --lnn_broadcast / --good_recipe + sanity 门
```

## 9. 依赖与约束

- Python ≥3.10，torch ≥2.0（量化的 QEngine：x86=FBGEMM/oneDNN，arm64 macOS
  自动降级）；CPU 推理可用，GPU 推荐。
- 算力分档：<3h 本地 8GB；>3h 上 Kaggle T4/P100（已解锁 cu118）或 AutoDL A100。
- 部署目标：7.2GB VPS（2B 全 int8 后 ~1.9GB 权重）；官网生产三容器
  （静态 bind-mount，`server.py` 改动需 rebuild `mtlnn` 容器）。
- 设计约束：13 原丝生物约束保留在 LNN（`n_protofilaments=13`）；新增机制一律
  默认 off + 位等价 + 独立消融（贡献 < seed 噪声即砍）。

## 10. 开放问题与风险

| 问题 | 风险 | 应对 |
|---|---|---|
| 论文摘要"14.7% 大预算主张"对着弱基线测 | 高（头号学术风险） | 100K 步补 modern_transformer 对照（AutoDL 排队）；反转则重写摘要 |
| S1 桥接真实任务（A5 三连判负） | 中 | NONDIAGONAL_TRANSITION 决胜计划（成本×体积×性能） |
| 2B 权重未达标 | 中 | fp32/scheduler 修复已入 main；复训后走发布管线 |
| O1 48M 权重失踪 | 中（阻塞浏览器 demo） | 人工找回（Modal/服务器） |
| Gemini/前沿模型先做出本地+可审计 | 低-中 | trace JSONL 与 E0 协议是差异化资产；保持 4-6 周领先节奏 |
| 强 baseline 缺口（Mamba-2/GLA/DeltaNet） | 中 | AutoDL ¥300-500 预算待批 |

## 11. 资源缺口（2026-08 现状）

算力（最大瓶颈：P0-3 剩余 baseline、scaling law、M2-P1 蒸馏全部堵在此）＞
教师 API（蒸馏轨迹来源未定）＞ GPU 排队实验＞ O1 权重人工找回。分工卡与
预算数字见 HANDOFF.md §3.5/§3.6。

## 12. 撤回记录（继承自 v2.2，永久保留）

- "−28.5%/−27.7%/−34.4% adapter PPL @ 0.1-0.2% 可训参数" —— **撤回**（MT
  adapter 被冻结，增益 ≈LoRA；7.98 vs 7.92）。
- "+13.3% 云注入提升" —— 重新定性为 prompt-template 效应。
- "混合架构 O(1)"与"混合长上下文 LM 增益" —— 双 null。
- Orch-OR / Φ̂ / 麻醉"意识"结果 —— 惰性（AVP 失败、Φ̂ 符号反转），降级为研究
  脚手架，已移出默认 import 链。
- 2K 步"MT-LNN 领先 31%" —— 欠训练 + 弱基线伪影，收敛后消失。

（v2.2 全文可在 git 历史中检索：`git log -- PRD.md`。）
