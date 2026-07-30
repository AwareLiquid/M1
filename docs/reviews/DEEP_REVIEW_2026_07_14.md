# MT-LNN 深度评审报告（架构 / 性能 / Bug）

> 日期：2026-07-14 · 评审方式：4 路独立并行审查（架构合理性 / 性能证据 / 核心模型代码逐行 / 记忆与服务子系统），关键数值 bug 均经实跑复现验证。
> 范围：E:\M1 主仓库（忽略 E:\M1\M1 镜像与 .venv*）。审查时工作区有未提交改动：config.py / hamiltonian_head.py / model.py（Hamiltonian world model，+229 行）。

---

## 一、总体结论

**架构：核心命题不成立，但内含一个真实资产。** 真正干活的组件是 `FastWeightMemoryV2`（带衰减的线性注意力，GLA/DeltaNet 一脉）——跨窗口召回 0.56 vs 移除后 0.008 是它的功劳。而"微管/液态/GWT/Orch-OR"叙事层要么被模型自己学会关闭（GWTB broadcast_gate 收敛到 0.0009），要么被自家消融证伪（5 个生物模块 PPL 全中性），要么存在数学硬伤（v1 递归记忆天花板约 10 token）。v2 的设计演化实质上是对 v1 的逐步否定，正在收敛到 Mamba/GLA 的标准做法。

**性能：真实但有限，且早期旗舰声明已被项目自己正式撤回。** RESULTS.md（2026-07-11）是一份诚实的"证据对账文件"：−28%~−34% PPL 适配器增益（实为纯 LoRA + 冻结适配器）、Selective Copy ×42（评估 bug，真实 ×1.32）、Orch-OR/意识模块（NULL/INERT）全部撤稿。立得住的只有三个小规模结果，均未在主流基线（Mamba/GPT-2/Pythia）或收敛训练下验证。

**Bug：存在 2 个 critical + 3 个 high。** 未提交的 Hamiltonian head 新代码在推理路径必崩（已实测复现）；服务端 memory 端点漏加锁会污染共享流状态；长会话恢复停留在约第 32 轮的过期快照。

---

## 二、架构分析

### 2.1 核心递归（v1 CfLTC）的数学硬伤

递归形式：`h_t = decay·h_{t-1} + (1-decay)·σ(W_in x_t)`，`decay = exp(-dt/τ)`，`τ ∈ [0.01, 10]`（config.py:32）。

- **记忆上限被 τ_max=10 钉死**：最慢通道 decay ≈ 0.905，100 token 后信息保留 ≈ 4.6e-5，跨 512 窗口梯度因子 ≈ e⁻⁵¹ ≈ 0。v1 原生递归数学上不可能记住几十 token 以外的信息。实测 out-of-window LM 双重 null（TBPTT +0.004 噪声）与此理论预测完全一致。
- **decay 与输入无关**：与 Mamba 最本质的差距（Mamba 的 Δ(x) 让内容决定遗忘速率）。v2 的 `selective_decay` 补了课但默认关闭，v1 原生模型没有。
- **状态被 sigmoid 压到 (0,1) 正象限**：表达力弱于 S4D/Mamba 的线性可正可负状态。
- v2（mt_lnn_v2.py）已把 tau_max 提到 100、加 selective decay + SiLU 门，但原生 125M 训练线（train.py → model.py → mt_lnn_layer.py）仍用 v1 参数化。

### 2.2 各组件评价

| 组件 | 评价 |
|---|---|
| FastWeightMemoryV2 | ✅ 真实资产。chunked 并行化实现干净（实测与逐步参考 1e-7 一致），跨窗召回 + bit-exact 快照/恢复是差异化能力 |
| 并行扫描 parallel_scan.py | ✅ 数学正确（Blelloch、h_init 吸收），但纯 Python 递归 + (B,T,P,S,D) 物化是训练显存劣势主因 |
| MicrotubuleAttention | ⚠️ "极性偏置" = 可学习符号的 ALiBi；bias 物化为 (B,H,T,T) 稠密张量 → 无法走 Flash-Attention，O(T²) 额外激活，4096 OOM 主因之一。RoPE+极性+ALiBi+bilinear 四种位置机制冗余叠加 |
| GWT Bottleneck | ⚠️ 实现质量不错，但模型训练后把 broadcast_gate 学到 0.0009 = 自己投票关闭；48M switch-matrix PPL 中性 |
| GlobalCoherenceLayer | ❌ 有真 bug（跨样本泄漏，见 4.2），且被证伪的机制仍留在默认前向路径 |
| 5 个生物模块 | ❌ 全部 PPL 中性，全开亏 5.6% 吞吐；默认开启的 predictive coding 反而趋负（+0.65 PPL） |

### 2.3 过度工程化（量化）

- 74 个模块 / 约 23,000 行，**生产核心只需 11 个模块**（约 4,000 行支撑全部已证明结果）。
- **23 个模块（约 4,400 行）完全孤儿**（无人 import）：phi_iit、geometry_ops、replay、attractor_ops、session_consolidation、stdp_ops 等。
- `__init__.py` 45 条顶层急加载 import：任一实验模块的依赖缺失会击穿整个包。
- 同一功能多份平行实现：Hebbian ×2、世界模型 ×2、Φ ×3、记忆 ×6（其中 4 份孤儿）。
- config.py 约 60 个开关，两两交互无测试覆盖。
- 建议收敛路径：v2 + ARR + fast-weight（约 15 个模块），孤儿与 phi/quantum/anesthesia 系整体移入 experiments/。

---

## 三、性能评估

### 3.1 立得住的数字（RESULTS.md PROVEN 表，已与原始 JSON 工件逐格对账）

| 指标 | 数字 | 条件与保留意见 |
|---|---|---|
| 125M 从零训练 val PPL | 299.5 vs 436（自建 Transformer），−31% | WikiText-103 仅 2000 步严重欠训练、单种子、无 Mamba/GPT-2 基线、MT-LNN 慢 1.6× |
| 跨窗口联想召回 | 0.56 均值（3 种子 0.43–0.62），对照组 0 | 对照组（注意力/LoRA）按测试构造必然为 0——证明信息走 fast-weight 通道，**不是**对 RMT/Memorizing Transformer/Mamba 等竞品记忆方案的胜利 |
| O 系列 O(1) 推理状态 | 恒定 0.381 MB vs KV-cache 384 MB@128k | 但 O 系列 PPL = 教师 2.15×（蒸馏 token 仅 18M vs 业界 3B+），未达可用 |
| Needle（修复后） | 2048 内 base 与 adapter 平局（非增益） | 每格仅 5 样本 |
| 能力保持 | LAMBADA/ARC-e/HellaSwag/PIQA ±1pt 噪声内 | lm-eval-harness 全量 |
| Hamiltonian 物理头 | 能量漂移 3–6×、rollout 2–24× 优于 MLP | 仅物理轨迹任务，不在 LM 路径 |

### 3.2 已撤回的声明（不可再引用）

- 适配器 −28.5%/−27.7%/−34.4% PPL → PEFT 把 MT 适配器冻结在随机初始化，实际只有 LoRA 在训练；受控归因显示 MT 多花 62.8M 参数只换 −0.064 PPL（噪声内）。
- Selective Copy ×42 → 评估 bug（基线被逐 token 喂丢前缀），真实 ×1.32。
- 长上下文压缩/出窗增益 → 双重 NULL。
- Orch-OR/Φ̂/麻醉验证 → AVP FAILED，Φ̂ 在麻醉下符号反转。
- Cloud-inject +13.3% → 100% 来自 prompt 模板。

### 3.3 可信度评估

- ✅ benchmark 脚本抽查真实跑模型、严格 exact match、带 HONEST GUARD、负面结果主动报告；当前 RESULTS.md 口径可信。
- ⚠️ 但项目历史上出过三次系统性评估事故（×42、needle harness 假象、冻结适配器），2026-06 后才修正。**任何未列入 RESULTS.md PROVEN 表的说法默认不可信。**
- ⚠️ PHASE5B_ANALYSIS.md 和 ABLATIONS.md 开头仍残留已撤回数字，与 RESULTS.md 冲突，应清理。
- ⚠️ 训练显存是负资产：混合架构 12344 MB@2048 vs Transformer 8993 MB，4096 同样 OOM。

---

## 四、Bug 清单（按严重度）

### CRITICAL

**C-1 · Hamiltonian head 在 no_grad 下必崩（未提交新代码，已实测复现）**
- hamiltonian_head.py:311-336 + model.py:667-670。`HamiltonianWorldModelHead.forward` 无条件跑 `ham.step()` → `torch.autograd.grad`，但没包 `torch.enable_grad()`。`model.generate()` 本身 `@torch.no_grad()` → 开启该功能后第一次验证/生成即崩。实测：`element 0 of tensors does not require grad`。
- 修复：forward 内 `with torch.enable_grad():` 包裹（eval 时 detach 输出），或 `compute_loss=False and not training` 时短路。

**C-2 · /v1/memory/write 与 /v1/memory/query 无锁翻转共享 adapter 流状态**
- serve/server_hf.py:813-841 + llama_adapter.py:553-576。commit 129fdd4 只给 3 个生成端点加了 `_gen_lock`，memory 端点漏了。`adapter_streaming_paused` 全局翻转 `stream_enabled` 并快照回写：与生成并发 → 生成中途退化为无状态 FFN + 流状态被回滚成旧快照（FW_SESSION_STORE=1 时脏状态还会被持久化）；两个 memory 端点彼此交错 → `stream_enabled` 永久卡 False 直至重启。
- 修复：两个端点加 `Depends(_gen_lock)`（两行改动）。

### HIGH

**H-1 · 长会话恢复到过期快照（约第 32 轮）**
- fast_weight_store.py:134-152 + knowledge_memory.py:272-273。同 session 每轮追加一条 exact-id 行（cosine 并列 1.0），`torch.topk` 并列取最旧索引（已实测），召回窗口 32 之外的新快照永远取不到；`touch=True` 还刷新旧行 access_seq，LRU 反而逐出新快照——bug 自我固化。
- 修复：per-session UPSERT（write 前 delete 同 id 旧行），可一并解决 M-3。

**H-2 · sparse resonance top-k 因果泄漏 + batch 泄漏（已实测复现）**
- mt_lnn_layer.py:161-162。`gate_mean` 在 batch 和全时间窗上求均值选 active τ-scale：未来 → 过去泄漏、样本间泄漏、prefill vs 逐 token 解码不一致（实测 max diff 0.209，dense 模式 8.9e-8）。默认 off，但所有开启该 flag 的实验结论被污染。

**H-3 · serve/server.py 整个文件零锁 + 流式缺 try/finally**
- server.py:687-721 客户端断开时 `GeneratorExit` 使 `save_state` 不执行，session 状态丢失（server_hf 同问题已修，native server 未同步）；392-423 同 session 并发 load→gen→save 丢失更新；629-631 `/v1/sleep` 无锁原地缩放 live 权重。

### MEDIUM（摘要）

- **M-1** world_model.py:213 / model.py:537-546：推理期 `last_pred_error` 永不更新且 buffer 非持久 → checkpoint 加载后"surprise 驱动 LAVI"闭环在部署中不存在。Hamiltonian 新代码复制了同一模式。
- **M-2** model.py:823-834：MTP head k=1 与主 CE 完全重复任务，K=3 时 1/3 MTP 参数冗余，对 spec-decode 无 lookahead 价值。
- **M-3** fast_weight_store.py:80-108：consolidate 按行而非 session 保留，单个高 surprise 长会话挤占全部配额，其它 session 记忆静默消失。
- **M-4** knowledge_memory.py:132-164：打开旧库不校验 key_dim，换基座模型复用 KB_PATH → 全部查询 500，混合维度写入后库永久不可查。
- **M-5** knowledge_memory.py:90-94：`torch.load(weights_only=False)` × "db 可跨端同步"宣传 = pickle 任意代码执行面。
- **M-6** awareliquid_daemon.py + session_state.py:25-29：ThreadingHTTPServer 无锁 load→append→save + 非原子 write_text → 丢数据/文件撕裂后连锁 500。修复：临时文件 + os.replace + per-session 锁。
- **M-7** mt_lnn_layer.py:562-565：GTP 时钟用 x.dtype 建 arange，纯 fp16 下位置 >2048 计数错乱。
- **M-8** model.py:583-651：pad_mask 不传 GWTB/相干层，左 padding/packed sequence 下有效 token attend 到 pad；辅助损失不 mask pad。
- **M-9** global_coherence.py:105-132：collapse gate 对 batch 求标量能量 → 样本 i 输出依赖样本 j（跨样本泄漏）；默认开启的 decay_wm 分支含 for-t 逐 token Python 循环（512 次/前向，训练慢 1.6× 的重大嫌疑）；`decay_rate` 无约束可漂出 (0,1)。

### LOW（简列）

- app.py:87-96 `_top_p` 是全仓库唯一未修的错误 nucleus 实现（丢弃跨越 p 的边界 token），server_hf.py:858-864 已是正确版本。
- model.py:436 全局 init_weights 冲掉 CompetitiveGWTB BidProjector 的零初始化（std=0.02 扰动，非声称的 O(1e-4)）。
- memory.py:98-103 `check_same_thread=False` 共享连接零锁（latent，与 129fdd4 修掉的同型）。
- mt_lnn_v2.py:419-429 流式状态不校验 device/dtype；hamiltonian `_grad` 原地改调用方张量 requires_grad；`gtp_gamma.clamp` 梯度死区；generate() 不接受 attention_mask。

### 验证过无问题的部分

parallel_scan 结合律与增量一致性（9e-8）、FastWeightMemoryV2 分块闭式扫描（1e-7）、mt_attention KV cache 因果性、gwtb causal mask、Hamiltonian 训练路径（leapfrog + create_graph 梯度回流）、129fdd4 的 KB RLock 修复本身、序列化对称性、空库/空查询边界。

---

## 五、优先行动建议

1. **立即**（发布/训练阻断级）：修 C-1（enable_grad 包裹）、C-2（两行加锁）、H-3 的 try/finally。
2. **本周**：H-1/M-3 一并用 per-session UPSERT 解决；给 server.py 引入 _gen_lock；清理 PHASE5B/ABLATIONS 中已撤回数字。
3. **架构层**：a) 决定 v1 原生线的去留——tau_max=10 + 无 selective decay 的训练线继续投入意义有限，应切到 v2 参数化；b) 把 GlobalCoherenceLayer 的跨样本 gate 改 per-sample 或直接移出默认前向；c) attention bias 改 fused（否则 max_seq_len 1024 即上限）；d) `__init__.py` 急加载改懒加载，孤儿 23 模块移 experiments/。
4. **性能叙事**：README 头条的 −31% 应加"budget-limited signal"限定；下一轮实验补 Mamba/GPT-2 基线 + 多种子 + 收敛训练，否则对外声明缺乏支撑。

**一句话定位**：把项目重新定位为"带可持久化 fast-weight 记忆的 GLA/Mamba 变体 + O(1) ARR 蒸馏"，删掉微管/Orch-OR 叙事层，会是一个立得住的小而精的研究库。

---

## 六、修复状态（2026-07-14 同日完成）

第四节列出的 bug 已全部修复（工作区未提交，涉及 17 个文件 +744/−155 行），验证：

- 专项验证脚本 `_verify_fixes_2026_07_14.py`：**19/19 pass**——含 C-1 no_grad forward/generate 不崩、H-2 sparse prefill vs 增量一致（2.1e-06）且 batch 无关（0.00）、MTP 新索引 + 短序列 guard、GWTB 零初始化不变量、pad_mask None-parity（0.00）、WM buffer 持久化 + 旧 checkpoint strict 加载兼容、UPSERT 写 40 次召回第 40 次快照且单行、KB key_dim fail-fast、weights_only round-trip、原子写 round-trip。
- 仓库回归测试（-m "not slow"，排除缺 hypothesis 的 property 测试）：核心相关 **162 passed** + 记忆相关 **30 passed**，0 失败。

修复要点备注：
- **H-2** 的修复方案：sparse 尺度选择从"当前窗口 gate 均值 topk"改为 **`kappa_gate.bias`（静态可学习参数）的 topk**——确定、因果、batch 无关、prefill/decode 一致。这改变了 sparse 实验模式的语义（不再内容自适应选尺度），已有的 sparse 实验结论本就被泄漏污染，需重跑。
- **MTP** 对齐 DeepSeek-V3 约定：head k 预测 t+1+k（原 t+k 与主 CE 重复）。开启 MTP 的旧训练 run 的 mtp_loss 数值不可与新代码对比。
- **world_model / hamiltonian 的 `last_pred_error`** 改 persistent=True，并加 `_load_from_state_dict` 垫片保证旧 checkpoint strict=True 加载不炸。
- **GlobalCoherence** collapse gate 改 per-sample；`decay_rate` 使用时 clamp 到 (1e-4, 1−1e-4)（不改参数化，旧 checkpoint 兼容）。
- **decay_wm 的 for-t Python 循环**（疑似训练 1.6× 减速主因之一）**未**向量化——属性能优化非正确性 bug，留待后续。
- **未修的架构级问题**（R1 tau_max 天花板、R3 attention bias 物化、R6 基线问题）是设计决策，不在本次 bug 修复范围。

---

## 七、GPU 实测验证（2026-07-15，Kaggle P100-16GB，kernel `muningan/mt-lnn-gpu-verify` v5）

**正确性（全绿）**：19 项修复验证套件在 Kaggle 上 19/19；GPU 专项 6/6——forward/backward/generate、hamiltonian no_grad、sparse prefill==增量（1.6e-06）、pad_mask parity（0.00）、fp16 autocast。本地全套件 **1200 passed / 0 failed**。

**GPU 实测中新发现并当场修复的 bug**：
- `global_coherence.py` 4 处 `masked_fill(-1e9)` 在 `autocast('cuda', fp16)` 下溢出 Half（±65504）直接 RuntimeError——存量 bug，说明该代码从未在 fp16 AMP 下跑通过。已改 `torch.finfo(dtype).min`（commit 57b995b）。
- 基准脚本层面确认了一个健壮性隐患：OOM 中断的训练 forward 会把带 autograd 图的 aux 张量（`resonance.last_pred_error`、`_hebb_signal`）残留在模块属性上，钉住整个部分前向的激活（实测 ~15GB 无法通过 `empty_cache` 回收，必须重建模型/触发新前向覆盖）。生产训练循环里做 OOM 恢复时要注意。

**性能基准（129.3M 参数默认配置，fp32）**：

| 场景 | 吞吐 | 峰值显存 |
|---|---|---|
| 训练 B=4 T=512 | 3,068 tok/s | 7,969 MB |
| 训练 B=4 T=1024/2048 | **OOM** | >16 GB |
| 训练 B=1 T=1024 | 1,252 tok/s | 4,818 MB |
| 训练 B=1 T=2048 | 983 tok/s | 10,536 MB |
| 增量 decode（prefill 512 + 128 新） | 28.4 tok/s | **968 MB** |
| （参考）本地 CPU 24 线程 | 训练 213 tok/s / decode 18.6 tok/s | — |

**解读**：
- 训练显存随 T 的超线性增长（B=1: 1024→2048 显存 2.2×、吞吐 -21%）与 R3（O(T²) bias 物化）+ R1（5 组 per-scale 状态物化）的评审判断一致。T=2048 在 16GB 上勉强可训（B=1），T=4096 无望——**不改 attention bias 实现，长上下文训练就是天花板**。
- decode 968 MB 的常驻足迹健康（混合架构带 KV cache），但 **28.4 tok/s 的解码速度太慢**（仅为本地 CPU 的 1.5 倍）——瓶颈不是算力而是每 token 的 Python/调度开销（decay_wm 逐 token 循环、每步 KV 重拼、13 个小 einsum），需要 kernel 融合或 CUDA graph 才有质变。
- Kaggle 运行注意事项：P100 是 sm_60，Kaggle 预装 torch 只支持 sm_70+，kernel 里必须先装 `torch==2.4.1 --index-url .../cu121`。Phase 5 在该镜像上有 16 个依赖性失败（本地全过，torch 降级引发的环境噪音），不影响结论。
