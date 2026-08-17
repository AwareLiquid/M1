# AwareLiquid 商业闭环 MVP：系统级 PRD 与 TODO 清单

## 1. 核心目标 (Objective)
当前 `mt_lnn` 核心引擎的算法闭环（$O(1)$ 推理、稀疏共振）已在本地验证成功。
本 PRD 旨在补齐产品形态上“端云联动”和“隐私继承”的缺失闭环，真正实现 Investor Deck 中承诺的 **Hybrid Edge-State Intelligence System（混合边缘状态智能系统）**。

最终交付物：一个名为 `demo_mvp_loop.py` 的全新入口脚本，能够向投资人完整演示“**状态保存 -> 遇见盲区 -> 本地截断 -> 调取云端资料 -> 融入本地状态 -> 持续对话**”的全过程商业故事。

---

## 2. 缺失模块 PRD (Product Requirements Document)

### 2.1 状态胶囊的持久化与继承 (State Capsule I/O)
* **用户场景**：用户在设备 A 上聊了三个月，积攒了丰富的个人习惯。换到新设备 B 时，只需导入一个极小的文件，新设备瞬间懂他。
* **功能定义**：
  * 将 `MT-LNN` 的隐状态张量（$h_{prev}$，约 4.1KB）序列化为本地文件（如 `.capsule` 或 `.safetensors`）。
  * 支持从磁盘免算力、零延迟地 `load_capsule()` 恢复上下文。
* **验收标准**：证明 100 轮对话后的状态，可以通过写入和读取 4.1KB 的文件实现 100% 的接续输出，不再需要重新输入历史 Token。

### 2.2 认知盲区与幻觉阻断 (Confidence & Deviation Detection)
* **用户场景**：当被问到生僻的行业术语或实时新闻时，AwareLiquid 不会像传统模型那样“胡编乱造”，而是能“意识到自己不懂”。
* **功能定义**：
  * 在 token 采样环节 (`streaming_inference`) 引入**隐层置信度/熵值 (Entropy/Confidence) 监测**。
  * 如果连续 N 个 token 的预测分布极度平缓（即熵值过高，置信度极低），触发 `INTERRUPT`（截断）信号。
* **验收标准**：面对设定的刁钻知识问题，程序能够自动打断输出，并在控制台打印 `[WARN] 认知置信度低于阈值，准备请求外脑...`

### 2.3 云端打工节点路由 (Cloud Oracle Router)
* **用户场景**：被阻断后，本地 AwareLiquid 自动生成精准的搜索词发给云端巨头 API，拿回生肉数据后，消化成自己语气的回答丢给用户。
* **功能定义**：
  * 构建一个 `router.py`，负责发起对外请求。
  * **MVP 方案**：可先用 Mock 数据字典，或接入简单的搜索 API（如 Wikipedia API、或简单的假大模型响应），证明“收编云端事实为己用”的链路。
  * 将云端返回的“客观知识流”在后台以 `Quiet Mode`（静默模式）塞入 AwareLiquid 引擎，只更新状态但不输出。
  * 状态更新后，再次向用户回答。
* **验收标准**：日志清晰展示：`截断 -> 发送极简 Query 到云端 -> 获得【外部事实】 -> 静默吸收 -> 用本地风格流畅回答`。

---

## 3. 工程执行 TODO (Action Items)

### 阶段一：实现记忆持久化 (State Capsule)
- [x] **创建文件** `mt_lnn/capsule.py`。
- [x] 实现 `save_capsule(state_dict, filepath)`：提取包含 $h_{prev}$ 及其它 $O(1)$ 必须参数，使用 `torch.save` 或 `safetensors` 保存。
- [x] 实现 `load_capsule(filepath)`：反序列化并还原到运行时模型，接管模型当前状态。
- [x] **单元测试**：写一个简短的 `test_capsule.py` 验证一小段对话保存后，重新加载继续对话的逻辑连贯性。（已在 `demo_mvp_loop.py` 闭环测试）

### 阶段二：实现置信度探针与截断 (Entropy Monitor)
- [x] **修改核心推理** `mt_lnn/streaming.py` 或 `app.py` 的 generate 逻辑。
- [x] 在 `next_id` 的 `logits` 挑选阶段，增加 `compute_entropy(logits)` 方法。
- [x] 设定硬性阈值（例: `ENTROPY_THRESHOLD = 0.85`）。
- [x] 如果触发阈值，捕获 `BlindSpotException` 等自定义异常跳出生成循环。

### 阶段三：云端外挂路由 (Cloud Oracle Router)
- [x] **创建文件** `mt_lnn/router.py`。
- [x] 实现 `call_awareness_cloud(query: str) -> str` 接口（初版可写一个字典 mock，如检测到“量子力学”，返回固定的专业解释文本）。
- [x] 实现“静默吸收吸收层”：将返回文本喂给 `streaming_inference`，但不 yield 字符向屏幕输出，纯粹为了让模型刷新 $h_{prev}$ 隐状态。

### 阶段四：集成与演示 (Demo MVP Loop)
- [x] **编写大闭环演示脚本** `demo_mvp_loop.py`。
- [x] 流程脚本包含：
  1. 初始化 AwareLiquid。
  2. 用户输入包含盲区的问题。
  3. 触发断点 -> 调用 Router -> 吸收事实。
  4. 最终回答。
  5. `save_capsule()` 存入当前工作目录体验。
- [x] 在命令行跑通，确保证明逻辑畅通，用于后续向投资人和评审做“技术变现”的 Demo。

---

## 4. 下一阶段 TODO — Reasoning UX 超越 Gemini 3.1 (2026-05-28 新增)

> 战略详见 [ARCHITECTURE.md](../../ARCHITECTURE.md)。MVP 闭环已通过,以下任务把 demo 推向"真能用、真比 Gemini 强"的产品形态。

### Phase 5 · Backbone 升级 (P0,前置依赖,不可跳)
- [ ] 在云 GPU 上跑通 `train_llama_mt_adapter.py`,产出 1.1B Qwen/TinyLlama + MT-LNN adapter checkpoint
- [ ] 把 `streaming.py` 的 ASCII vocab=200 替换为真 tokenizer (Qwen/Llama BPE)
- [ ] 验证: WikiText-103 PPL 接近原 backbone (退化 < 10%)
- [ ] 验证: 长上下文 needle-in-haystack 8K 准确率 > 95%
- [ ] **验收**: `demo_mvp_loop.py` 能用 1.1B 模型对自然语言问题输出连贯回答

### Phase 6 · Φ-Trace 接线 (P1,差异化护城河)
- [x] `phi_hat.py` 提供 Kraskov kNN Φ̂ 估计器 (`compute_phi_hat(hidden, K, k_nn)`)
- [x] `demo_awareliquid_v2.py` 每 N 步通过 forward hook 采样 last-layer hidden 算 Φ̂,写入 ReasoningTrace JSONL
- [x] `reasoning_trace.py` 记录 (token_id, entropy, route, phi) 四元组,可回放
- [ ] `ui.html` 推理时间线视图 (留 UI 迭代)
- [x] **验收**: `python demo_awareliquid_v2.py --phi_every 8` 跑通真 LM + Φ̂ + trace

### Phase 7 · Deliberation Router (P2,Layer 2 升级)
- [x] `deliberation.py` 新增 `semantic_entropy(samples)`: 对 N 条候选续写做语义熵 (cluster-based)
- [x] 路由分支: low → 直出 / mid → self-critique / high+fact_gap → cloud
- [x] `lexical_fact_gap()` 检测: 与 `capsule.evidence_log` 做词级重叠 (embedding 检索留 Phase 9)
- [x] `cloud_client.py`: env-var 驱动的 `build_oracle_client()`,真 API (Gemini/OpenAI) lazy-import,缺 key 优雅降级 Mock
- [x] **验收**: 离线模式下 router 不崩 (graceful degrade 到 self-critique);测试覆盖 20 个用例 ✅

### Phase 8 · Capsule v2 + Daemon
- [x] `session_state.py`: `HFSessionState{session_id, open_questions, evidence_log, history}` 序列化
- [x] `open_questions` 入栈: entropy 触发 cloud 时把当前子问题压入 (在 `demo_awareliquid_v2.py`)
- [x] `evidence_log` 写入: 每次 cloud inject 记录 (source, query, fact_len, ts)
- [x] `awareliquid_daemon.py` stdlib HTTP daemon (端口 8765),零新依赖,聚合多 session
- [x] **验收**: `tests/test_session_state.py` + `tests/test_awareliquid_daemon.py` 通过,关机重启可恢复

### Phase 9 · Meta-Learning Plane (2027+)
- [x] `meta_learning.py`: 哈希嵌入 + numpy KMeans 对全量 evidence_log 聚类
- [x] daemon `/meta/clusters?k=N` 暴露聚类结果 (representative_query, sources histogram)
- [ ] 提取共有 belief → "用户专属先验" capsule (留 prefill 阶段)
- [ ] prefill 时优先加载先验 capsule,再叠加当前 session capsule
- [x] **验收**: `tests/test_meta_learning.py` 双 blob 聚类 purity > 0.85

### 跨 Phase 不变量 (Invariants)
- I1: Capsule ≤ 5KB · I2: 端侧延迟 < 50ms · I3: Φ 全程可算 · I4: 离线可用 · I5: Evidence 可审计