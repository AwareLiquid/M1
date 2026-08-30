# HF_NATIVE_LANDING — 目标·现状·关键决策·已知问题（交接文档）

> 接手人入口：读这一篇 + `docs/SERVING_ROADMAP.md` 即可接手。
> 代码分支 `iter/hf-native-serving`，worktree `../M1-hf-serving`（无 .venv 是设计，勿装）。

## 0. 一句话

入场券的模具已造好（HF 原生加载 + adapter 分发 + 基准脚本 + vLLM 路线文档），
**但还没有盖钢印**——所有验证仍是纸面的，位等价与三行加载冒烟属于 Phase B。

## 1. 目标与动机

**目标**：pip 装完三行代码加载推理。两条产品线各自的标准形态：

- O-series（自研架构）→ `MTLNNForCausalLM.from_pretrained(path)`，trust_remote_code 类
- M-series（冻结基座+适配器）→ `recipes.load_mt_adapter_dir(base, dir)`，Hub 就绪目录

**为什么做**：
1. 行业基线：国产开源模型（DeepSeek/Qwen/GLM/Kimi/MiniCPM）发布即带 FP8 权重
   + vLLM/SGLang day-0 支持——没有生态入口的模型没人采用
2. 仓库现状：只有自研 `serve/` 与 ONNX workaround（且需 `dynamo=False`），
   生态适配是最大工程欠账
3. 杠杆排序：HF 原生加载 = 天级工作量、立刻解锁"能被加载/能被分发"；
   vLLM = 38-88 人日（SERVING_ROADMAP §3）。先拿最便宜的入场券

**非目标（红线）**：不改模型数学/训练；不做 KV 账目；不改 serve/server.py；
不新增第三方依赖；不上传任何 Hub 产物（任何阶段外发留给人工）。

## 2. 现状盘点（截至 2026-08-29）

| 交付物 | 位置 | 状态 |
|---|---|---|
| O-series HF 模型类 | `mt_lnn/hf_model.py`（411 行） | 已 commit，未执行 |
| M-series adapter 布局 | `mt_lnn/adapter_export.py` + `recipes.load_mt_adapter_dir` | 已 commit，未执行 |
| 加载基准脚本 | `benchmarks/hf_load_bench.py`（275 行，只写不跑） | 已 commit |
| vLLM 路线决策文档 | `docs/SERVING_ROADMAP.md`（272 行） | 已 commit，§7 待回填 |
| 随机小权重测试 | `tests/test_hf_model.py`(5) + `tests/test_adapter_export.py`(2) | 已写，未跑 |

Commit 清单（本 PR 自有增量，6 个）：98592f8 模型类 → 1997ca2 adapter 布局 →
ee634fa 基准脚本 → d0ff544 路线文档 → 1c8a5e6 transformers 5 实测修正 →
5330ded+ 本交接文档。

Phase B 四项**全部未发生**：① 分支未合并（不在 main、不在任何工作分支）；
② `benchmarks/results/hf_load_bench.json` 不存在；③ SERVING_ROADMAP §7
"Phase B 待回填的数字"仍空着；④ ≤200 字结论 + README 三行验证版未产出。

协议合规证据：worktree 内 `__pycache__` 仅 11 个变更文件、时间戳同一秒、
无 `.pytest_cache`、无连带 import 产物 → 只有 py_compile 批量语法检查；
语法检查 11/11 通过；`serve/` 与 requirements 零改动。

## 3. 关键决策记录（接手必读，按影响排序）

- **D1 委托外壳，数学零改动**：`PreTrainedModel` 外壳的 forward/generate 全部
  委托既有 `MTLNNModel`，不复制任何数值路径。红线"位等价"的工程表达。
- **D2 不走 GenerationMixin.generate**：ModelCacheStruct 是循环状态，
  HF DynamicCache 表达不了。直接复用 model.py 既有 generate；代价是依赖
  HF 标准 generate/cache 的生态集成（assisted decoding 等）不可用。
- **D3 按实装 transformers 5.16.1 适配，而非 lock 的 4.57.3**：按 .venv 实际
  安装的源码逐条核对出三处 4.x 写法全错（`_tied_weights_keys` 列表→字典、
  `low_cpu_mem_usage` 已移除、`post_init()` 会重置权重）。核心修法是
  **buffer 重算协议**：5.x 用 meta device 建图，non-persistent buffer
  （RoPE 表/注意力掩码/GWTB causal）会被 `torch.empty_like` 灌成垃圾值，
  **静默算错**——`get_init_context` 摘掉 meta 上下文 + `_init_weights`
  回调 `reset_non_persistent_buffers()`，两道缺一不可，并有专用守卫测试。
- **D4 刻意不导出到 `mt_lnn/__init__.py`**：不让 `import mt_lnn` 无条件
  拉起 transformers（依赖足迹，见基准脚本的 sys.modules 口径）。
- **D5 MT adapter 不伪装成 PEFT 类型**：MT 残差适配器不是 LoRA，硬塞
  peft_type 只会得到双方都读不懂的目录。双轨：LoRA 部分走 peft 原生
  子目录，MT 部分走自有权重文件 + 图规格——非标模块必须付的税。
- **D6 图规格键名三路径共享**：`adapter_config.json["mt"]` 键名逐字照抄
  server_hf.py 从 checkpoint args 读的那套（也照抄训练 argparse）——
  训练/服务/分发共享同一份图重建规格。挂载顺序严格：先 MT 再 PEFT
  再灌权重（训练产物 key 带 `base_model.model.` 前缀，反序挂不上）。
- **D7 tiny 随机权重构造器是测试的关键设计件**：vocab 64 / 2 层 /
  d_model 104，毫秒级；位等价断言用 `torch.equal`（非 allclose），
  沿用仓库 bit-exact 口径。随机权重对"序列化忠实性"是最强检验，
  不需要真实训练权重。
- **D8 基准诚实口径**：冷加载用独立子进程（分离"启动+import"固定成本）；
  依赖足迹用 sys.modules 差分；5 条口径局限写死 CAVEATS 常量并随 JSON
  落盘，防止数字被误读。
- **D9 vLLM 路线**：Transformers Modeling Backend 兜底（零建模代码但拿不到
  循环状态 paged 管理）/ 树外插件 / 内置实现三条路径；架构建议：把
  O-series 对外固定为纯循环 + 无 GWTB 可省 10-20 人日；GGUF 零代码是
  绿地，建议不做。
- **D10 Phase A/B 协议本身**：worktree 零执行（py_compile 除外）——因为
  worktree 无 .venv 与缓存是设计，验证必须在主 worktree 真环境。
  失败回 worktree 修复再请求合并，是内建的迭代环。

## 4. 已知问题与风险（按严重度）

1. **[阻塞] Phase B 未执行**：一切"三行代码可用"目前是设计断言，非实测事实
2. **[环境坑] transformers 版本账实不符**：requirements.lock 写 4.57.3，
   .venv 实装 5.16.1；按 lock 重装会回到 4.x 语义（三处 API 差异）。
   lock 修复超出本分支范围，但接手人须知
3. **[结构性税] 三行代码 ≠ 零依赖**：自定义架构类要求 mt_lnn 包在场
4. **[生态边界] 不走 GenerationMixin.generate**（见 D2）
5. **[PEFT 边界] MT adapter 非 LoRA**，vanilla peft 读不懂（见 D5）
6. **[维护风险] buffer 重算协议靠纪律**：新模块加 non-persistent buffer
   忘写 `reset_non_persistent_buffers()` → 静默算错；守卫测试只覆盖
   现有 7 类模块
7. **[服务缺口] vLLM 无路径**：兜底拿不到循环状态 paged 管理与连续批处理
8. **[小瑕疵] 测试 7 个**：超"4~6"全局读法上限 1 个（按任务拆分合规）

## 5. 测试策略：用例做门禁，最小流程做验收

被测对象是**加载路径的忠实性**（本分支零模型行为变化），不是模型质量：

1. **门禁 = 位等价用例**（已写好的 7 个）：随机小权重 + `torch.equal`，
   触达每个张量，tiny 模型即可全覆盖。必须全绿才继续
2. **验收 = 最小流程**：hf_load_bench 实跑落盘（两路径 logits 差应=0.0）
   + 真实小 checkpoint 冒烟（随机权重测不出 dtype/格式/布局问题）
   + README 三行示例实跑过才可贴
3. **何时才需要"设计模型流程迭代测试"**：被测对象变成行为/性能时——
   即做 vLLM 后端（吞吐/时延/连续批处理）或量化（FP8 误差预算）之时，
   位等价口径失效。那是后续分支，本任务不需要

## 6. Phase B 执行清单（收到合并指令后，主 worktree）

预估：绿路径机器时间约 15-25 分钟（全量 1432 测试约 5-15 分钟 + 基准
3-5 分钟），含文档写作约 1 小时；若有测试红，每轮修复加 1-3 小时。
最大单点风险：transformers 5.16.1 运行时行为（D3 是照源码核对的，未执行过）。

1. **前置**：主 worktree 本地 `iter/irregular-streaming-edge` 落后其远端
   （e87919e vs 1774c49），先 `git pull --ff-only` 同步基座
2. `git merge iter/hf-native-serving`（merge-tree 实测零冲突；
   与在途未提交文件零重叠）
3. 全量 pytest；红 → 回 worktree 修复 → 再请求合并
4. `python benchmarks/hf_load_bench.py` → `benchmarks/results/hf_load_bench.json`
5. 真实小 checkpoint 冒烟，两条线各一（本地已有 `checkpoints/llama_mt_adapter`）
6. 回填 SERVING_ROADMAP §7；README 三行示例定稿（已验证版）
7. ≤200 字结论（两线示例是否跑通 + vLLM 最小依赖路径）→
   `git worktree remove ../M1-hf-serving`

## 7. 接手上手指南

- **位置**：worktree `../M1-hf-serving`（分支 `iter/hf-native-serving`）；
  主仓库 `/Users/aricredemption/Projects/M1`（有 .venv，Phase B 在这里跑）
- **PR**：见 PR 描述——stacked，base=`iter/irregular-streaming-edge`，
  本迭代自有增量仅 6 commit；基座合 main 后 GitHub 自动 retarget
- **两分钟动线**：`git -C ../M1-hf-serving log --oneline -6` 看 commit →
  读 `mt_lnn/hf_model.py` 文件头（三个已知边界写在头注释）→ 读本文 §3
- **禁忌**：worktree 内零执行（py_compile 除外）、不 pip install、
  不上传 Hub、不动 serve/、不自行合并
