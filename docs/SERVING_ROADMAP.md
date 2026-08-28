# SERVING_ROADMAP — O / M 两条线进主流推理栈的路线与代价

> 决策文档，不是施工文档。目标是回答三个问题：**今天我们能做到哪一步**、**要
> 真进 vLLM 还差什么**、**每一项的依赖与工作量是多少**。
> 数字口径：2026-08。所有对外部项目现状的引用都给了出处，便于日后复检。

---

## 0. 一句话结论

| 产品线 | 本分支之前 | 本分支之后 | 到"生产级 vLLM 原生"还差 |
| --- | --- | --- | --- |
| **M-series**（冻结 HF 基座 + MT adapter） | 只有 `serve/server_hf.py` 能从 checkpoint `args` 重建图 | `adapter_export` 产出 Hub 就绪目录；`recipes.load_mt_adapter_dir` 一句加载。LoRA 部分是标准 PEFT 目录 | 近零 —— 基座本来就受支持，adapter 走 PEFT/自定义加载即可 |
| **O-series**（原生 `MTLNNModel`） | 只有自建 serve + ONNX workaround（`dynamo=False`） | `MTLNNForCausalLM` = 标准 `PreTrainedModel` + `config.json` | **38–68 人日 + GPU 验证**（自定义循环状态后端，见 §3.4） |

**最短价值路径**：本分支本身就把 O-series 送进了 vLLM 的
**Transformers Modeling Backend 兜底**（§3.1）—— 零建模代码、当天可用，
代价是没有 PagedAttention 级别的性能。真正的原生支持是另一件事，别混为一谈。

---

## 1. 现状盘点

| 能力 | M-series | O-series | 落点 |
| --- | --- | --- | --- |
| 自研 HTTP 服务 | ✅ `serve/server_hf.py` | ✅ `serve/server.py` | 已有 |
| HF 标准序列化（`from_pretrained`） | ⚠️ 基座有、adapter 无 | ❌ → ✅ 本分支 | `mt_lnn/hf_model.py` |
| Hub 就绪分发目录 | ❌ → ✅ 本分支 | ❌ → ✅ 本分支 | `mt_lnn/adapter_export.py` |
| ONNX | 同基座 | ⚠️ 有导出、有约束 | §5 |
| GGUF / llama.cpp | 同基座 | ❌ 仓库内零代码 | §5 |
| vLLM / SGLang | ✅ 基座原生支持 | ❌ | §3 |

"HF 原生"这一步不产生任何新的推理能力 —— 它买的是**生态入场券**：
Auto 类分发、Hub、`--model-impl transformers` 兜底、以及后面所有工作的地基。

---

## 2. 参照系：国内模型的 day-0 支持现状（2026-08）

这不是"别人做得到所以我们也要做到"的攀比，而是**工作量标定**：下面这些模型
的 day-0 支持都不是"发一个 PR 加个 modeling 文件"就完事的。

* **vLLM 0.27.1**（2026-08 稳定版，两周一个 release）近期 day-0 覆盖
  Qwen3.8-2.4T-A95B、Kimi K3、GLM-5.2、Nemotron 3.5 Lightning
  （[vllm.ai](https://vllm.ai)、
  [vLLM 版本记录](https://rightaichoice.com/tools/vllm)）。
* **Kimi K3**（Moonshot，2.8T MoE）：KDA 线性注意力（固定大小循环状态）与全
  注意力混合。vLLM 为此建了**混合 KV cache 管理器**——paged KV block 与紧凑
  循环状态块统一调度，并且在循环状态之上重做了 hybrid prefix caching（在物理
  块内登记状态快照、扩展前先复制）。官方原话：这套机制"如今惠及所有与 Kimi K3
  类似的混合模型"（[vLLM × Kimi K3](https://docs.vllm.ai) 技术博客）。
* **Qwen3.8**（2026-08-12，2.4T/95B active，92 层 = 69 GDN 线性注意力 + 23
  GQA 全注意力）：SGLang 的 day-0 要同时处理**三种状态**——全注意力 KV、GDN
  循环状态、GDN 卷积窗口；MTP 投机解码要 `ReplaySSM` 重放循环输入；PD 分离要
  在 prefill/decode 之间搬运 KV + 循环状态 + 卷积窗口 + 草稿 KV + 隐状态 +
  top-k 元数据（[SGLang × Qwen3.8 day-0](https://www.aib.vote/en/news/sglang-miles-qwen3-8-day0-support)）。
* **DeepSeek-V4**：FP4 原生专家权重 + 混合稀疏注意力，SGLang 为此做了
  ShadowRadix（跨异构 KV 池的前缀缓存）与 in-graph metadata（把混合注意力的
  元数据准备融进 CUDA graph）（[LMSYS 博客](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)）。
* **SGLang** v0.5.x 的 day-0 清单同样覆盖 Llama 4 / Qwen3 / DeepSeek V4 /
  GLM-5 / Kimi K3，量化支持 FP8 / AWQ / MXFP4 / NVFP4 / GPTQ /
  compressed-tensors / W8A8。

**结论**：线性/循环注意力今天的待遇已经从"没人支持"变成"一等公民，但要为一个
新架构单独写状态管理 + 前缀缓存 + 投机解码适配"。我们面对的是同一个量级的工
作，不是配置一下就行。

---

## 3. O-series 进 vLLM：三条路径，我们落在哪

vLLM 拿到一个 `config.json` 后按 `architectures` 查注册表，三种结果
（[注册表与插件机制](https://docs.vllm.ai/en/latest/contributing/model-registration.html)、
[插件系统](https://docs.vllm.ai/en/latest/design/plugin_system.html)）：

| # | 路径 | 触发条件 | 性能 | 我们的成本 |
| --- | --- | --- | --- | --- |
| 1 | **Transformers Modeling Backend 兜底** | 注册表里查不到，或显式 `--model-impl transformers` | 兼容性优先，官方定位"与专用实现相差不大"，但 **拿不到循环状态的 paged 管理 / 前缀缓存** | **≈0**（本分支已完成前置条件） |
| 2 | **树外插件** | `vllm.general_plugins` entry point + `ModelRegistry.register_model(...)` | 可做专用优化 | 中（不用 fork vLLM） |
| 3 | **内置实现** | 加进 `_VLLM_MODELS` 并合入上游 | 最好 | 高（要走上游 review） |

### 3.1 路径 1：Transformers 兜底（本分支直接解锁）

vLLM 的 Transformers 后端会包装任意 `PreTrainedModel`，并接上它自己的 paged
KV cache、TP/PP、量化与权重加载
（[后端实现](https://deepwiki.org/vllm-project/vllm/5.3-transformers-modeling-backend)）。
**它驱动的是 `forward`，不是 `generate`** —— 采样由 vLLM 自己做，所以我们在
`hf_model.py` 里覆写 `generate`（委托原生解码器）并不构成阻塞。

两个必须在真机验证的坑（Phase B 之后）：

1. **meta device 建图**。该后端用 `AutoModel.from_config` 在 meta device 上先
   建结构、后灌权重。我们的 RoPE 表（`inv_freq` / `cos_table` / `sin_table`）
   与 GWTB 的 causal mask 都是 **non-persistent buffer 且在 `__init__` 里算出
   来的**，不在 state_dict 里，加载时不会被回填。若 vLLM 没有物化它们，就会
   得到"meta tensor"运行时错误。修法（届时按代价排序）：改成 persistent
   buffer / 改成 lazy 构造 / 声明不支持 meta init。
   *这也是 `MTLNNForCausalLM.from_pretrained` 默认 `low_cpu_mem_usage=False`
   的原因，同一类问题。*
2. **`AttentionInterface` 未接入**。要吃到 HF 侧的优化注意力后端，模型需要用
   `ALL_ATTENTION_FUNCTIONS` 并设 `_supports_attention_backend = True`
   （[Transformers as a backend](https://huggingface.co/docs/transformers/v4.56.0/transformers_as_backend)）。
   我们用的是自建 `MicrotubuleAttention`，所以兜底路径下**正确性有、注意力性能
   特性没有**。

### 3.2 路径 2/3：原生实现 —— 我们属于最难的那一类

vLLM 把"带状态的层"分三种情况
（[新模型接入指南](https://docs.vllm.ai/en/latest/contributing/model/basic.html)）：

1. 纯 Mamba-1/2（无注意力）：`IsAttentionFree` + `MambaMixer/MambaMixer2`；
2. Mamba + 注意力混合：`IsHybrid`；
3. **自定义 mamba-like 层**（线性注意力、ShortConv 等）：继承 `MambaBase`，
   自己实现状态管理。

O-series 的液态核心（LTC 连续时间常数 + 多尺度共振 + 可选 selective decay /
DeltaProduct / Householder 转移）**不是 Mamba2 mixer**，属于第 3 类。要做的
事情在 vLLM 文档里是逐条列明的：

* 继承 `MambaBase`，实现 `get_state_dtype` / `get_state_shape` / `mamba_type` /
  `get_attn_backend`；
* 实现对应的 attention metadata 类（参照 `LinearAttentionMetadata`）；
* 更新 `MAMBA_TYPE_TO_BACKEND_MAP` 与 `MambaAttentionBackendEnum`；
* 用 `direct_register_custom_op` 把层调用包成自定义算子，并登记到
  `vllm/config/compilation.py` 的 `_attention_ops`，否则 torch.compile 与
  piecewise CUDA graph 不支持；
* 线性层换成 `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear`
  / `VocabParallelEmbedding` / `ParallelLMHead`；
* 实现 `load_weights`（含 `stacked_params_mapping` 合并权重）。

若 O-series 保留 `attention_layers`（混合形态），还要额外继承 `IsHybrid`，并
面对 §2 里 Kimi K3 / Qwen3.8 遇到的同一件事：**循环状态上的 prefix caching**。

### 3.3 自定义 attention backend vs recurrent state 后端 —— 现状判断

| | 自定义 attention backend | recurrent state（mamba-like）后端 |
| --- | --- | --- |
| vLLM 成熟度 | 一等公民：`Attention` + backend 注册表，FlashAttention / FlashInfer / MLA 等 | 已体系化：`MambaBase` 协议 + 状态后端注册表，但**每个新状态类型都要单独写一遍** |
| 我们的位置 | `MicrotubuleAttention` 是自定义注意力，接入需要按 backend 接口重写 | 液态核心是自定义循环状态 —— 这才是主要工作量 |
| 判断 | 若 O-series 走 `attention_layers=()`（纯循环），这一项**完全不需要做** | **必须做**，且是唯一无法绕开的深水区 |

**因此架构上有一条省一大笔钱的路：把 O-series 的对外形态固定为"纯循环 +
无 GWTB"（`attention_layers=()`, `use_gwtb=False`），则不需要碰自定义
attention backend，也不需要 Kimi K3 那套混合 cache。** 反过来，一旦要保留
注意力层，工程量立刻翻倍。这条应该在架构分支里钉死，而不是留给 serving 阶段。

### 3.4 若要 vLLM 原生支持，必须新增的依赖与工作量

依赖（**都是新依赖，仓库现在一个都没有**）：

| 依赖 | 用途 | 备注 |
| --- | --- | --- |
| `vllm >= 0.27` | 引擎本体 | 需要 CUDA/ROCm 工具链；Kimi K3 的 day-0 甚至一度只提供 Docker 镜像（依赖 FlashInfer 预发布版） |
| `triton` | 自定义循环状态 kernel | vLLM 的 mamba-like 路径大量使用 |
| `flash-attn` / `flashinfer`（按 backend 选） | 注意力后端（**仅当保留注意力层**） | 版本与 vLLM 强绑定 |
| GPU 机器 | P1/P3/P4 阶段的验证 | 循环状态 kernel 与 CUDA graph 无法在 CPU 上验证 |

工作量（人日，**含 GPU 验证**，按上文依赖链顺序累加）：

| 阶段 | 内容 | 人日 | 是否需要 GPU |
| --- | --- | --- | --- |
| P0 | 插件骨架：`vllm.general_plugins` + `ModelRegistry.register_model` + config 适配 + 最小可跑 | 3–5 | 否 |
| P1 | `MambaBase` 子类：状态 shape/dtype、metadata 类、backend 枚举注册 | 10–15 | **是** |
| P2 | 并行层替换 + `load_weights` + TP 正确性验证 | 5–8 | **是** |
| P3 | `direct_register_custom_op` + `_attention_ops` + torch.compile / piecewise CUDA graph | 5–10 | **是** |
| P4 | **仅当保留注意力层**：混合 cache + 循环状态上的 prefix caching | 10–20 | **是** |
| P5 | 量化接入（FP8 / int8，对接 vLLM linear_method 或 compressed-tensors） | 5–10 | **是** |
| P6 | 上游合入（若走内置路径）+ 回归套件 | 5–10 | 否 |
| | **合计** | **38–68**（纯循环口径）／**48–88**（保留注意力） | |

这些是**估计**，标定依据是 §2 里 Kimi K3 / Qwen3.8 的公开工程量描述，不是我
们自己跑过的数字。真要启动，第一步应该是 P0 的 3–5 人日 —— 它同时是"值不值
得继续"的探针。

---

## 4. M-series：PEFT 的现成路径与我们的缺口

**现成的部分**：基座是 Qwen/Llama/GLM，vLLM 与 SGLang 原生支持；多 LoRA
批服务是 vLLM 的一等功能。训练时若开了 LoRA，那一半导出的
`peft_lora/` 是标准 PEFT 目录，`PeftModel.from_pretrained` 直接吃，社区工具链
全部兼容。

**缺口（必须说清楚）**：`MTResidualAdapter` 不是 LoRA。它是注入进 decoder
layer 的自定义残差模块（N protofilaments × 多时间尺度共振 + 可选 fast-weight
memory），PEFT 的 LoRA / AdaLoRA / IA3 都表达不了它。硬塞进 `peft_type` 只会
得到一份 PEFT 读不懂、我们也加载不回来的目录。

所以 `adapter_export.py` 的取舍是：

* LoRA 一半 → PEFT 原生子目录（标准路径）；
* MT 一半 → 我们自己的 `mt_adapter.pt` + `adapter_config.json["mt"]` 图规格，
  由 `mt_lnn.recipes.load_mt_adapter_dir()` 加载（约 20 行，内部复用
  `llama_adapter.attach_*` 的同款重建语义，键名逐字对齐
  `serve/server_hf.py` 读的 checkpoint args）。

**这是非标模块必须付的税，写清楚比假装它是 LoRA 更有用。**

服务侧后果：只要 MT adapter 生效，vLLM 的原生多 LoRA 就用不上（vLLM 只认识
LoRA）；MT adapter 要走自建 serve 或等 vLLM 支持自定义 adapter 类型。**在
M-series 上，适配的性价比排序是：先确保 LoRA-only 变体能跑满 PEFT 生态，
MT adapter 作为自建 serve 的差异化能力。**

---

## 5. ONNX / GGUF 的定位

**ONNX：存在，但是 workaround 级别，不是 serving 路线。**

* 现状：`export_onnx_for_netron.py`（可视化）、
  `benchmarks/check_onnx_webgpu_feasibility.py`（可行性门禁）、
  `benchmarks/export_o1_for_browser.py` / `benchmarks/streaming_edge_profile.py`。
* 硬约束（都是实测出来的，不是猜测）：
  1. **必须 `dynamo=False`** —— torch ≥ 2.9 的默认 dynamo 导出器会拒绝本模型
     （attention head-merge 产生非连续张量，其分解坚持走 `aten.view`）；
  2. **序列长度被烘焙进图** —— 只能在固定窗口上导出，浏览器侧 pad/截断；
  3. 数值一致性已验证（ORT vs PyTorch max diff **3.58e-07**）。
* 定位：**浏览器 / 端侧 demo 的下发格式**（48M 投影 ≈ 66 MB int8，可下发），
  不是服务端 serving 路线。而且它靠的是 legacy TorchScript 导出器 —— 这条
  路径在 torch 的未来版本里是会被拿掉的，属于技术债。

**GGUF / llama.cpp：仓库内零代码，是绿地不是既有链路。**

之前没有任何 GGUF 相关工作（全仓搜索无命中）。llama.cpp 要求算子在 GGML 里
有实现，液态核心的并行扫描 + 可学习时间常数 + 数据依赖门控都没有。定位：**端
侧 CPU 部署的候选，但要先回答"循环状态在 GGML 里怎么写"，工作量不亚于 vLLM
路径的一半，而生态收益小得多。建议不做，除非端侧 CPU 变成硬需求。**

---

## 6. 建议的决策顺序（以及明确不做的）

1. **做（本分支已完成）**：HF 原生打包。它是后面每一条路径的公共前置条件，
   而且立刻兑现 `--model-impl transformers` 兜底。
2. **做**：真机验证 §3.1 的两个坑（meta device buffer、`AttentionInterface`）。
   便宜、决定路径 1 到底能不能用。
3. **做**：P0 探针（3–5 人日，插件骨架）。跑通后再决定是否投入 P1–P3。
4. **钉死架构口径**：O-series 对外到底是不是"纯循环 + 无 GWTB"。这个决定值
   10–20 人日（P4 整段）。
5. **不做**：GGUF / llama.cpp（生态收益 < 工作量）。
6. **不做**：为了"看起来标准"把 MT adapter 伪装成 PEFT 类型。
7. **不做**：ONNX 作为服务端路线（legacy 导出器 + 定长约束）。

---

## 7. Phase B 待回填的数字

本分支的 `benchmarks/hf_load_bench.py` 会在合并后跑一次，回填下面的
`TBD`，出处是 `benchmarks/results/hf_load_bench.json`：

| 指标 | HF 原生路径 | 自研 serve 路径 | 值 |
| --- | --- | --- | --- |
| 冷加载（墙钟，含解释器 + import） | ✅ | ✅ | TBD |
| 冷加载（仅加载，进程内口径） | ✅ | ✅ | TBD |
| 依赖足迹（`sys.modules` 增量） | +transformers | +mt_lnn only | TBD |
| 首 token 延迟（batch=1、8 token、greedy） | ✅ | ✅ | TBD |
| 两条路径 logits 最大绝对差 | — | — | **应为 0.0**（数学零改动的证据） |

口径局限随 JSON 一起落盘（脚本内 `CAVEATS` 常量）：CPU、随机小权重、含解释器
启动、无 CUDA graph / 无连续批处理。这些数字**只能用来比较两条加载路径**，
不能外推成真实吞吐。
