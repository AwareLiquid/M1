# HF_NATIVE_LANDING — Phase A 现状、已知问题与 Phase B 测试计划

> 状态快照：2026-08-29。分支 `iter/hf-native-serving`（worktree `../M1-hf-serving`）。
> Phase A 代码/测试/文档已全部 commit 并封存，等待合并指令进入 Phase B。

## 0. 一句话

入场券的模具已造好（HF 原生加载 + adapter 分发 + 基准脚本 + 路线文档），
**但还没有盖钢印**——所有验证仍是纸面的，位等价与三行加载冒烟属于 Phase B。

## 1. 现状盘点（已核实证据）

| 交付物 | 位置 | 状态 |
|---|---|---|
| O-series HF 模型类 | `mt_lnn/hf_model.py`（411 行） | 已 commit，未执行 |
| M-series adapter 布局 | `mt_lnn/adapter_export.py` + `recipes.load_mt_adapter_dir` | 已 commit，未执行 |
| 加载基准脚本 | `benchmarks/hf_load_bench.py`（275 行，只写不跑） | 已 commit |
| vLLM 路线决策文档 | `docs/SERVING_ROADMAP.md`（272 行） | 已 commit，§7 待回填 |
| 随机小权重测试 | `tests/test_hf_model.py`(5) + `tests/test_adapter_export.py`(2) | 已写，未跑 |

- 分支 5 个任务 commit：98592f8 → 1997ca2 → ee634fa → d0ff544 → 1c8a5e6，
  基于 `iter/irregular-streaming-edge`@e87919e 切出，预期可 fast-forward 合回。
- 协议合规证据：worktree 内 `__pycache__` 仅 11 个变更文件、时间戳同一秒
  （2026-08-29 02:55:22）、无 `.pytest_cache`、无连带 import 产物 → 只有
  py_compile 批量语法检查，零执行验证。`serve/` 与 requirements 零改动。
- 语法检查 11/11 通过（验收时用 ast.parse 重验过）。

## 2. 已知问题与风险（按严重度）

1. **[阻塞] Phase B 未执行**：分支未合并、pytest 未跑、`hf_load_bench.json`
   未落盘。"三行代码"目前是设计断言，不是实测事实。
2. **[环境坑] transformers 版本账实不符**：requirements.lock 写 4.57.3，
   .venv 实装 5.16.1。代码按 5.16.1 源码逐条核对（meta 建图、
   `_tied_weights_keys` 字典化、`low_cpu_mem_usage` 移除三处 API 差异），
   但若按 lock 重装环境会回到 4.x 语义。lock 本身是待修项，超出本分支范围。
3. **[结构性税] 三行代码 ≠ 零依赖**：自定义架构类要求 mt_lnn 包在场，
   trust_remote_code 解决的是"怎么加载"，不是"免安装"。
4. **[生态边界] 不走 GenerationMixin.generate**：ModelCacheStruct 是循环
   状态，DynamicCache 表达不了。依赖 HF 标准 generate/cache 的集成
   （assisted decoding 等）不可用。
5. **[PEFT 边界] MT adapter 不是 LoRA**：vanilla peft 读不懂，必须走
   `recipes.load_mt_adapter_dir`；LoRA 部分才走 peft 原生子目录。
6. **[维护风险] buffer 重算协议靠纪律**：新模块若新增 non-persistent
   buffer 却不实现 `reset_non_persistent_buffers()`，from_pretrained 出来
   的模型会**静默算错**；守卫测试只覆盖现有 7 类模块。
7. **[服务缺口] vLLM 无路径**：兜底 Transformers Backend 拿不到循环状态
   的 paged 管理与连续批处理；真支持需 38-68 人日（纯循环口径），
   见 SERVING_ROADMAP §3/§6。
8. **[小瑕疵] 测试数 7 个**：超"4~6 个"全局读法上限 1 个（按任务拆分
   ——HF 类 5 + adapter 2——读法合规）。

## 3. 测试策略：用例测试 vs 流程测试

**结论：两层都要，顺序严格——用例测试是门禁，最小流程测试是验收；
不需要设计训练/质量迭代流程。**

判断依据是被测对象：本分支零模型行为变化（红线禁止动数学/训练），
被测的是**加载路径的忠实性**。对"忠实性"而言，随机小权重 + 位等价断言
（`torch.equal`，非 allclose）是最强且最便宜的检验——随机权重已触达
每个张量，tiny 模型即可全覆盖，不需要真实训练权重。

### 第一层：测试用例（门禁，必须全绿才继续）

已写好的 7 个即此层：config 往返逐字段 / 权重位等价+绑定保持 /
buffer 位等价（transformers 5 专属守卫）/ 三行加载前向逐位 /
generate == 原生 / adapter 布局契约 / adapter 重载位等价。
口径：transformers 5.16.1 实装版本。

### 第二层：流程级验收（绿门之后）

用例测不到的三件事，用最小流程补上：

- **真实工件**：随机权重测不出真实 checkpoint 的格式/dtype/目录布局问题
  ——取一个小真实 checkpoint 走三行路径 vs 直接构造，位等价 + generate 对比
- **用户旅程**：`hf_load_bench.py` 实跑（冷加载/依赖足迹/首 token +
  两路径 logits 差应为 0.0），JSON 落盘
- **可复制性**：README 三行示例必须实跑过才可贴（Phase B 验收项）

### 何时才需要"设计具体模型流程迭代测试"

当被测对象从"忠实性"变成"行为/性能"时——即进入 SERVING_ROADMAP 的
vLLM 后端（吞吐/时延/数值容差，位等价口径失效）或量化链路（FP8/INT8
误差预算）之时。那属于后续分支，本任务不需要。

## 4. Phase B 执行清单（收到合并指令后）

1. 主 worktree：`git merge iter/hf-native-serving`（预期 fast-forward，
   与在途未提交文件零重叠）
2. 全量 pytest；红 → 回 worktree 修复 → 再请求合并（协议内建迭代环）
3. `python benchmarks/hf_load_bench.py` → `benchmarks/results/hf_load_bench.json`
4. 真实小 checkpoint 冒烟，两条线各一（O: from_pretrained；M: load_mt_adapter_dir）
5. 回填 SERVING_ROADMAP §7 数字；README 三行示例定稿（已验证版）
6. ≤200 字结论（两条线示例是否跑通 + vLLM 最小依赖路径）
7. 全绿后 `git worktree remove ../M1-hf-serving`
