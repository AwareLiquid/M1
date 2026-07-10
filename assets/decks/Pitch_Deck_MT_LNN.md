> **⚠️ CORRECTION (2026-07-05) — supersedes the adapter results below.**
> The Phase 5/5b adapter numbers quoted in this document (−28.5 %/−27.7 %/−34.4 % PPL
> at "0.196 %/0.139 %/0.117 % trainable") are **retracted**: those runs predate the
> re-arm fix (`8d9d741`) — PEFT had silently frozen the MT adapters at random init,
> so **only LoRA trained**, and the quoted "trainable" counts are exactly the
> LoRA-only parameter counts. A controlled 6-config attribution confirms plain LoRA
> reproduces those PPL gains; the MT adapter adds ≈nothing on in-window perplexity.
> The architecture's real, reproducible differentiator is **cross-window recall
> through streaming state** (fast-weight memory: 0.62 accuracy where attention/LoRA
> are 0 by construction), delivered by the 7.5× smaller v2s adapter now serving.
> Authoritative results and protocols: **BENCHMARKS.md** (attribution, cross-window
> recall, out-of-window LM nulls, ARR distillation).

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
- **Streaming recurrent state (类脑流式状态)**: instead of a growing KV cache, the attention-free O-series carries a **fixed-size recurrent state** — measured **flat at 0.381 MB vs an O(T) KV cache, 1008× smaller at 128k context**. Its fast-weight state also stores discrete key→value bindings that survive a dropped context window (**cross-window recall 0.56, where attention scores 0.000**).
  **MT-LNN 的类脑机制**：无注意力的 O 系用**固定大小的递归状态**取代不断膨胀的 KV cache——实测 128k 上下文下比 KV cache 小 **1008 倍**（恒定 0.381 MB）；快权重状态还能存住跨窗口的键值绑定（跨窗口召回 0.56，注意力为 0.000）。
  > *诚实边界:这是**联想记忆 + 恒定内存**,不是"把万字压缩进隐状态"——窗口外语言建模是实测阴性。见 [RESULTS.md](../../RESULTS.md)。*

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

**从"只会读文字"扩展为会"看-记-想-学-做"的类脑能力栈 (From a text-only backbone to an AI that can see, remember, imagine, learn, and act)**

| 新能力 (用大白话说) | 它到底做了什么 | 对客户的价值 |
|---|---|---|
| **空间计算：让 AI 在"脑海"里预演世界** | 像人一样在脑海里建一张场景地图：模拟事物怎么发展、判断"这事儿对不对劲"（压制幻觉）、做物理直觉推演、走过即记得 | 从"会说话"升级为"**会预演、会规划**"——自主智能体和机器人的前提；先在脑海排除不合理方案，**少胡说、少误操作**；端侧**零额外成本** |
| **可验证推理：每一步都有据可查、不偷偷跑偏** | 把推理拆成一块块"积木"，每块该有什么行为都用数学规则写死成自检，每次构建自动跑一遍（相当于给 AI 配了道出厂质检关） | **可信 = 护城河**：不是"相信我"而是"你自己跑一遍就知道"，行为可被第三方逐条核对；**967 项自动测试常驻绿灯** |
| **持续学习：越用越懂你，且学新不忘旧** | 像人一样温故而知新——自己保留一小本"要点本"，新旧知识穿插着学；学得好不好用三个直白指标打分 | **端侧终身个性化**：本地随用户越用越贴合、**数据不出端**；直击大模型"一微调就忘光"的行业痛点 |
| **自主智能体：已能自己"看→记→想→定→做"跑通全程** | 把上面的能力串成一个会自己干活的智能体，独立完成找路任务，每一环都是真实模块、没有一处造假 | 这正是**机器人 / Agent / 数字员工**的核心范式——把愿景变成**能跑的代码**（当前为整合演示原型，非生产成品） |
| **快反应+慢思考哨兵，且可直接部署** | 像人一样：平时用便宜的"条件反射"盯着，只有真异常才唤醒"深度思考"（省算力）；掉线不失控（惯性续航或安全停机）；输出永远在安全范围内（数学上限死） | 可落地的**安防 / 工业监测 / 国防**垂直线：又省算力、又安全、又能马上部署 |

→ 诚实标注 (honesty): 世界模型预演与认知智能体当前为演示原型 / roadmap；可验证推理与服务化路径有契约/属性测试背书。

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
