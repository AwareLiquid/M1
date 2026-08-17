# Product Requirements Document (PRD) - Awareness Network

**Version:** 2.0
**Date:** 2026-05-28
**Product Name:** Awareness Network
**Core Technology:** AwareLiquid (MT-LNN: Microtubule-inspired Liquid Neural Network)
**Architecture spec:** see [ARCHITECTURE.md](../../ARCHITECTURE.md)

> **v2.0 变更**: 在原 4 模块基础上,明确"超越 Gemini 3.1 推理体验"的产品定位与 5 层架构 (Layer 0–4)。原 4 模块归入 Layer 1–3。新增 Layer 0 (Backbone) 与 Layer 4 (Meta-Learning)。

## 1. Executive Summary (产品概要)
Awareness Network 是基于 AwareLiquid (MT-LNN) 架构打造的**“端云分离、状态主导”的混合边缘智能系统（Hybrid Edge-State Intelligence System）**。
在现有的 AI 商业模式中，企业和巨头们（如Google Gemini, OpenAI GPT-4）深陷云端算力竞赛与规模化瓶颈。Awareness Network 提出了一种降维打击的商业形态：**在端侧（设备本地）保留完全私有化的 AwareLiquid 核心作为思维与灵魂，而将所有云端巨无霸大模型（Gemini 3.1 等）弱化并收编为系统的“外部事实数据库”。**

## 2. Core Value Proposition (核心价值主张)

### 对抗“云端暴力美学”的异步竞争极点
当我们不卷百科全书式的广度时，AwareLiquid 能够基于极其严苛的资源约束展现出极强的推理与陪伴价值。
1. **云端做维基百科，端侧做核心推理：** 本地 AwareLiquid 极轻量地处理逻辑、记忆与状态演算；外部API解决客观事实检索。
2. **永远在线的陪伴生命体：** 摒弃每次都要重置上下文窗口的窘境，凭借 $O(1)$ 常数级内存流转机制，系统实现 7x24x365 的常开模式。
3. **数据主权在民：** 对于军工、金融、医疗与极端隐私的 C 端设备，灵魂网络必须彻底断网也能安全运行。

## 3. Product Architecture (系统架构与组件)
Awareness Network 包含四大核心解耦模块：

### 3.1 记忆层 —— 个人状态胶囊 (Personal State Capsule)
- **机制：** AwareLiquid 接收极长的用户日志、代码输入流，不保留任何原始文本，只利用物理规律不断刷新本地的隐状态矩阵 ($h_{prev}$)。
- **规格：** 一生的对话记录，物理占用维持在数百 KB 内（核心状态 4.1 KB）。
- **用户体验：** 用户切换设备只需转移该配置文件（State Checkpoint），新设备即刻继承“潜意识”与全部默契。它不再每次临时翻看聊天记录，而是天然“懂你”。

### 3.2 决策层 —— 内部静默演算池 (Latent Reasoning Loop)
- **机制：** 给模型加入“时间维度”。切断文本 Decoder，AwareLiquid 引擎通过多次循环流转隐藏状态来深度演算。
- **用户体验：** 告别冗长刻板的思维链 (Chain-of-Thought) 假思考系统。“请等我十秒”，系统十秒间不抛出任何废话 Token，之后直接一针见血命中核心。

### 3.3 感知层 —— 预测误差哨兵 (Predictive Error Monitor)
- **机制：** 采用顶下预测编码 (Predictive Coding)，AwareLiquid 大环境状态主动向下端（感知层）送出合理预判。如果与真实输入发生相悖，瞬间返回高数学偏差。
- **用户体验：** 从被动问答机器人，进化为**主动提醒助手**。结对编程或逻辑设计时，出现不合理规划直接高亮红灯阻断。

### 3.4 动作层 —— “无明知觉”云端路由 (Cloud Oracle Router)
- **机制：** AI 作为总控中枢，当发现本地隐状态“缺乏外部常识判断”时，自动利用极简精确的关键词发包给外部大厂 API（如 Gemini 3.1）。
- **用户体验：** 巨头 API 沦为一个“后台搜索员”。AwareLiquid 接收整理外来冰冷的资料事实后，化为己用，再通过符合用户潜意识配置的格式反馈。

## 4. Key Metrics & Benchmarks (技术约束与数据红线)
* **内存占用：** 理论极限必须维持在 $O(1)$，常态单进程占用（包含权重与推理堆栈）限制在 < 500MB 以适配移动端 NPU/CPU。
* **抗干扰精确度：** 当挂载超过 128K 或无限字符的历史流后，核心逻辑提取（如“大海捞针”盲测）维持 $>99\%$。
* **提速目标 (Sparse Resonance)：** 简单交互（社交寒暄、简单调用）在低算力下需实现瞬间休眠机制，提速超过 13%且毫秒级响应。

## 5. Reasoning UX 差异化 (vs Gemini 3.1)

知识广度不打——拆给云端 API 当事实硬盘。**推理体验**靠三件 Gemini 结构上做不到的事:

| # | 差异点 | 实现位置 | Gemini 为何做不到 |
|---|---|---|---|
| D1 | 跨 session 持续推理状态 | Layer 1 · Capsule v2 (belief + open_q + evidence) | 云端 query 无状态,response 结束即蒸发 |
| D2 | 按语义熵 + Φ 信号的细粒度路由 | Layer 2 · Deliberation Router (三级) | thinking budget 是一刀切 |
| D3 | 可回放、可归因的 Φ-trace | Layer 3 · Verifiable Trace + UI 时间线 | 黑盒 thinking summary,无持续状态可归因 |

详见 [ARCHITECTURE.md](../../ARCHITECTURE.md)。

## 6. Phased Roadmap (路线图)

- **Phase 1 (Q3 2026) · Backbone (P0)**: 跑通 1.1B Qwen/TinyLlama adapter。`train_llama_mt_adapter.py` → 产出可生成连贯长文的 L0 骨干。**前置依赖,不可跳。**
- **Phase 2 (Q3 2026) · Φ-Trace 接线 (P1)**: 把 `phi_iit.py` 接入 streaming loop,`ui.html` 渲染推理时间线。差异化护城河,80% 代码已写好。
- **Phase 3 (Q4 2026) · Deliberation Router (P2)**: Entropy router 升级为 semantic entropy + 真 cloud API。事实缺口/推理困难分别路由。
- **Phase 4 (Q4 2026) · Capsule v2 + Daemon**: belief/open_q/evidence 三段结构 + 本地常驻进程。
- **Phase 5 (2027+) · Meta-Learning Plane**: Capsule 聚类生成用户专属先验。RAG-on-state。

## 7. Competitive Advantage Summary (总结)
我们通过制造一个强有力的本地底层流状态机（AwareLiquid），改变了与大厂的零和博弈。我们把大厂（Gemini/GPT）降维成了 Awareness Network 下的一个“存储硬盘”，而把“智慧和意识”留在用户的手中。