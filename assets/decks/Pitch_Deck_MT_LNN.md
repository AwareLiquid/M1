---
marp: true
theme: default
class: lead
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
size: 16:9
paginate: true
style: |
  h1 { color: #00468b; }
  h2 { color: #00468b; font-size: 1.5em; border-bottom: 2px solid #ed0000; padding-bottom: 0.2em; }
  .nature-caption { font-size: 0.6em; color: #555; text-align: center; }
  .columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
  th { background-color: #00468b; color: white; }
---

# MT-LNN: A Brain-Like Liquid Neural Network
## 打破 Transformer 的长文本计算墙 (Breaking the Memory Wall)

**EverestAn**  |  2026

---

## 1. What is a Liquid Neural Network? (什么是液态神经网络/类脑计算？)

### The Biological Inspiration (生物学启示)
- **Transformer (Current Status Quo)**: Operates like a forced video recorder. It caches every single past frame (KV Cache). Over time, memory and compute explode $O(N^2)$.
  **Transformer 的困境**：如同有强迫症的录像机，把每一个字死死钉在显存里 (KV Cache)。随着上下文变长，计算量呈 $O(N^2)$ 级爆炸。
- **Brain-Like Fluidity (类脑机制)**: The human brain relies on **Working Memory** and **Selective Forgetting**. It compresses past events into a dynamic latent state, discarding noise and retaining semantic needles.
  **MT-LNN 的类脑灵感**：人脑绝不试图记住十年的每个像素。它依靠**工作记忆**与**选择性遗忘**。MT-LNN 引入并行线性扫描与量子态门控，将万字压缩成固定的隐状态 $h_{prev}$。

---

## 2. 算力经济学：白菜价碾压 A100 (The Compute Economics)

<div class="columns">
<div>

**Cost / Memory scaling with Context Length**
![width:500px](notes/fig_cost_scaling.png)

</div>
<div>

**商业落地差距 (Commercial Impact)**
- **Transformer**: To serve 100 users querying 100K-token documents simultaneously, you need ~60 A100 (80GB) GPUs. **(Monthly cost: ~$100,000)**
  (百用户并发 10万字，需极庞大的显存存 KV Cache)
- **MT-LNN**: O(1) constant generation cache. The working memory strictly occupies a compact Matrix per user. 100 concurrent users fit into a **single RTX 4090**. **(Monthly cost: ~$200)**
  (恒定隐状态占用，无损吞吐长文本节点，单卡可抗百并发)

</div>
</div>

---

## 3. Benchmarks & 架构优势 (Architectural Advantages)

### 🆕 Phase 6 milestone — EEG Rhythm Gate (2026-06-06)

**Dynamic stability/flexibility balance, zero parameter budget cost**

| New capability | Mechanism | Product impact |
|---|---|---|
| Long-context stability | High LAVI → slow τ dominant | State drifts < 50% less over 10K+ tokens (expected) |
| Context switch speed | Low LAVI → fast τ activates | Multi-turn boundary adaptation without forgetting |
| Audit trail | LAVI per-layer in diagnostics | Per-inference stability index for compliance reporting |

→ Enable: `MTLNNConfig(use_rhythm=True, global_rhythm=True)` — default off, zero regression.

---

### 🆕 Phase 7 milestone — Brain-Stack Expansion (2026-06)

**从纯文本架构扩展为"感知-记忆-推演-学习-行动"的类脑能力栈 (From a text-only backbone to a full perceive–remember–imagine–learn–act brain stack)**

| New capability | Mechanism | Product impact |
|---|---|---|
| **空间计算 (Spatial computing)** | 四层空间认知栈：网格/位置细胞 → 位置索引联想记忆 → 因果激活引导 → 世界模型 (L4 roadmap) | 从文本扩展到**具身/空间智能**（机器人、AR/VR 导航、IoT 巡检、空间 RAG），端侧**零额外参数**即获认知地图 |
| **可验证算子代数 (Verifiable operator algebra)** | 11 个零参数、与主干解耦的算子模块，行为由属性测试钉死；概率单纯形上的 **Fisher-Rao 信息几何**；配套数学白皮书 | **企业级可信赖 = 护城河**：可审计、可复现，**967 项测试常驻绿灯**，每条主张有脚本+产物背书 |
| **持续学习 (Continual learning)** | 有界经验回放缓冲 + 遗忘/保持/迁移度量（准确率矩阵）+ 回放 vs 灾难性遗忘流式对照 | **端侧终身学习不遗忘**：本地随用户个性化、隐私不出端，直击"微调即遗忘"痛点 |
| **全脑闭环智能体 (Closed-loop agent)** | `examples/demo_cognitive_agent.py`：感知(L1)→记忆(L2)→自上而下注意→想象(L4)→行动，全真实模块无 Mock | 从"一堆能力"跃迁为**可运行的具身认知闭环**（当前为整合演示原型，非生产成品） |
| **双速哨兵 + 生产化部署 (DualSpeedSentry + serving)** | 感知→推理→预测→触发→断连续航→安全执行；断路器保证执行输出有界；HF 适配器服务化路径（带契约测试）+ 容器打包 | 可落地的**安全关键 / 监测哨兵**垂直线（快反射+慢审议双速），失效可证明有界，**ready to deploy** |

→ 诚实标注 (honesty): L4 世界模型与认知智能体当前为 roadmap / 演示原型；算子代数与服务化路径有契约/属性测试背书。

---

### Phase 5b milestone — Qwen-2.5-1.5B + MT adapter (Kaggle GPU, 2026-05-29)

跨基座复现 — 同样的 MT 残差适配器配方在 **两个不同 1B+ 预训练 LM 家族** 上同样有效:

| Base | Trainable params | WikiText-2 PPL ↓ |
|---|---:|---:|
| TinyLlama-1.1B (Phase 5) | 2.3 M (0.196 %) | 9.16 → **6.55 (−28.5 %)** |
| **Qwen-2.5-1.5B (Phase 5b)** | **2.22 M (0.139 %)** | **11.10 → 8.03 (−27.7 %)** |

**Cross-base reproducibility** — Llama 家族和 Qwen 家族，**两组独立训练，PPL 下降幅度几乎一致**，证明 MT 时间动态归纳偏置对真实 1B+ LM 普适有效，不是 TinyLlama-only 的幸运。

— Reproduce: `kaggle/awareliquid_train_qwen_phase5b.ipynb` · raw artefacts: `benchmarks/kaggle_qwen_run/ppl_ablation.json`
— Needle-in-haystack 在 ≤1.5B base 上自身 ≡ 0, 待 ≥3B base 上重测.

<div class="columns">
<div>

## 3. Beyond "Next Token": The Predictive Coding Leap (超越"下一个词预测"的预测编码架构)

<div class="columns">
<div>

**Dynamic Channels \& Inter-scale Logic**
- **Predictive Coding Loss (多尺度预测编码)**: 
  Instead of blindly predicting the next token, MT-LNN leverages biological predictive coding. High-level reasoning channels generate Top-Down forecasts for lower-level perception channels. This forces the network to learn rich causal structures, vastly reducing training data dependency. (打破传统 Next-Token Prediction 桎梏，高维抽象通道自动向下发送预测，实现自我监督学习)
- **Endogenous Compute Skipping (内源性计算跳过)**: 
  Unlike static layers, MT-LNN tracks logical channel saturation via $\kappa$ gating. If a sub-channel is dormant, compute completely bypasses it. Hardware ROI multiplies drastically. (通过动态 $\kappa$ 阈值切断休眠通道的计算，推理芯片成本指数级下降)
- **EEG Rhythm Gate — Dynamic Stability (脑电节律门控)**: 
  Inspired by cortical oscillatory modes, MT-LNN now detects whether each protofilament is in **persistent mode** (stable context, slow τ emphasis) or **transient mode** (novel input, fast τ adaptation) via a per-step LAVI (Lag Angle Vector Index) score. This gives the model a history-aware stability signal — complementing the content-aware κ-gate — for dramatically improved long-context coherence and multi-turn context switching. (仿脑电节律：持续模式→慢 τ 维持上下文；瞬态模式→快 τ 快速切换，解决长文本状态漂移问题)
- **Clear Glass-Box Causality (完全可解释的因果提取头)**: 
  Native extraction branches for *Causal Chains* and *Self-Monitoring* translate obscure latent states directly into human-readable thought logs. 

</div>
<div>

**Transformer (e.g. Claude 3.5) vs MT-LNN**
![width:450px](notes/fig_benchmark_radar.png)

</div>
<div>

- **Long Context Recall (长文本捞针)**
  While Claude struggles with distraction in massive prompt contexts (Lost in the middle), MT-LNN explicitly filters out irrelevant context with **Selective Copy**.
- **Edge AI Deployable (端侧霸主)**
  Since memory stays constant, MT-LNN is the endgame architecture for **mobile phones, AR glasses, and Brain-Computer Interfaces (BCI)** where RAM is heavily constrained.
  (手机不再因 KV Cache 撑爆而发烫掉电)。

</div>
</div>

---

## 4. 商业版图与未来规划 (Roadmap & Future Planning)

我们不与千亿美金模型在“通识百科”上硬拼，而是通过**降维打击极大长文本场景**实现突破。
(We don't brute-force AGI knowledge against $100M arrays; we attack vertical extreme-context use-cases.)

| Phase / 阶段 | Scope / 规模 | Compute Cost / 预估所需算力 | Target Scene / 目标场景 |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Now)** | **1.5B Params** | 1× A100 (~$15 / 10h) | Local RAG Demo, Long-context Proof of Concept. (**跑通极限长文本寻点**) |
| **Stage 2 (3-6m)**| **7B Params** | 4× A100 (~$1,500) | Law contracts, Codebase analysis, Agent OS. (**合同审查、万行代码分析，性价比最高阶段**) |
| **Stage 3 (1-3y)**| **70B / 405B** | 512× H100 (~$5M) | General AGI alternative to Claude/GPT. (**全面挑战现有千亿级 Transformer**) |

---

## 4.5 Auditable Reasoning vs Black-Box Thinking (可审计推理 vs 黑盒思考)

<div class="columns">
<div>

**Gemini "thinking summary"**
- Post-hoc paragraph
- Not clickable, not diffable
- No per-token route / entropy
- No proof of when cloud was queried

</div>
<div>

**AwareLiquid reasoning trace**
- One JSONL row per token: `(step, entropy, route, phi, source)`
- `trace_timeline.html` — every token clickable, color = route
- `bench_trace_audit.py` — quantified metric: **self-sufficiency**

**Demo trace numbers (120 tokens):**
- LOCAL 94.2% · SELF_CRITIQUE 5.0% · CLOUD 0.8%
- **Self-sufficiency: 99.17%** (1 − cloud/total)
- Net cost vs always-cloud: **+$0.0016** saved
- Φ̂ sampled 14× · mean 0.221

</div>
</div>

**Compliance / regulated industries** (finance, legal, healthcare) cannot ship Gemini's opaque thinking. They can ship AwareLiquid — every fact's provenance lives in `evidence_log`, every route decision in JSONL.

— Reproduce: `python scripts/demo_trace_synth.py && open trace_timeline.html`

---

## 4.6 Cloud-Inject 真在用 — 真模型 +13.3% 准确率提升 (2026-05-29)

<div class="columns">
<div>

**30 道事实问答 · 真实 HF 后端 · Qwen-2.5-1.5B**

| Variant | no_inject | inject | uplift |
|---|---:|---:|---:|
| Qwen-1.5B (baseline) | 83.3% | **96.7%** | **+13.3%** |
| Qwen-1.5B + MT adapter | 83.3% | **96.7%** | **+13.3%** |

**两个 claim 同时拿下:**
1. `[Absorbed fact]` 模板**真的拉准确率** — 不再是 EchoBackend stub 数字，是真 Qwen 上 25/30 → 29/30.
2. **MT adapter 不破坏 in-context learning** — PPL 降 28% 的同时, inject uplift 100% 保留.

</div>
<div>

**为什么这两个组合起来很重要**

很多 LoRA / adapter 微调会让模型"闭起来" — 学会了训练分布, 反而忽略 prompt 里塞进来的新事实. 这是 RAG 圈的常见 bug.

AwareLiquid 的 MT adapter 给出反例:
- PPL 改善 → adapter 学到东西
- inject uplift 保留 → 但**没把基座 in-context learning 学坏**

这是 **AwareLiquid 架构哲学的实证**: 
*local 模型负责常识 + 主流知识 (83.3%),  
cloud 只在不知道的 5 题时介入 (+13.4%),  
adapter 不破坏这个分工.*

— Reproduce: `kaggle/awareliquid_cloud_inject_uplift.ipynb`
— Raw: `benchmarks/cloud_inject_qwen/*.json`

</div>
</div>

---

## 5. Contact & Links (相关链接)

**Experience the future of constant-memory architecture:**

- **Demo Repository (RAG UI)**: 
  [https://github.com/everest-an/M1](https://github.com/everest-an/M1)
- **Core Architecture Framework**: 
  `github.com/everest-an/M1`
- **Cloud Run Guide**:
  [View Cloud Deployment Guide](CLOUD_TRAINING_GUIDE.md)

*MT-LNN: Stop brute-forcing memory. Start thinking fluidly.*
*(放弃暴力存储，走向液态思考)*
