# 潜空间递归（latent recursion）— 测试交接手册（Phase A → Phase B）

> 分支 `iter/latent-recursion` → base `main`。本手册给测试同事：先读 §1 现状与
> 风险，跑 §2.1 本地门（约 10 分钟），Phase B（GPU 判决跑）按 §2.2 runbook。
> 设计文档是 [docs/LATENT_RECURSION.md](LATENT_RECURSION.md)，本手册只管
> "现状/怎么测/出了问题归因给谁"。

## 0. 一句话现状

Task 1（算力账本）已出数、Task 2/3（裁决/前沿）协议预注册且驱动经 Stage-1
实测验证、Task 4（τ 阶梯）开关就绪、Task 5 文档入库——**本 PR 全部代码与
协议落地，判决级实验数据待 GPU（Phase B），零实验声明**。

> **Phase B 回填（2026-08-31，A100 服务器回收前抢救入库）**
> - **Task 2 判决跑 48/48 完成，判决 NULL**：`h_supported=False`，
>   `diagnosis=budget_wall`（全档 chance：stack d1→d8 = 0.0668→0.0675 非单调，
>   core d8 = 0.0676；gain 0.0007 « 2σ 0.0048）。循环体在 30k 步预算下从未
>   学会 pointer_chase，深度效应免谈；判据未移动。处置按 R2：加 steps 重跑。
> - 事故记录：① core 单模式收尾聚合在行 JSON 落盘**之后** KeyError: 'stack'
>   （不丢数据，纯收尾瑕疵）；② 服务器就地 verdict 是崩溃残留（n_rows=1，
>   不可引用）——入库 verdict 由 48 行原始数据 + 本分支冻结 judge 代码本地重算；
>   ③ 行 JSON `git_rev=unknown`（容器内无 git），`device=cuda` 如实保留。
> - **Task 3（anytime_frontier）未起跑**：A100 probe = CoT 0.154 s/step（×4
>   粒度）+ latent 0.204 s/step ≈ 6.8 h/seed ≈ 41 h 串行，回收窗口装不下。
>   resume 语义无损，仍为本 PR 唯一 Phase B 待办（§2.3 第 4 条"两实验齐全"
>   尚未满足，PR 不宜判全绿合并）。

## 0.5 分支谱系与迭代时间线（新接手者先读）

- 2026-08-29 00:17 从 `iter/event-stream-liquid` 的 HEAD 建分支（当时该
  worktree 被多个并行会话共享，先后有 event-stream/parametric-memory 的
  提交误落在本分支名下）。
- 分支手术：外部提交以 `rescue/event-dvs`、`rescue/pm-bench` 两个 tag 保
  全后，分支重置到 `main`（cad9372），本线工作迁入独立 worktree
  `M1-latrec`（根治共享 worktree 的 checkout 竞争）。
- 早期孤儿提交 4238b88（= Task 1 内容）残留在 `iter/irregular-streaming-edge`
  历史里（PR #2 的 base），内容与本 PR 首个提交相同，合并时 git 自动去重，
  无需处理。
- 2026-08-29 03:00 rebase 到最新 `main`（48706a6，KV 线 PR #4/#6/#7 已并，
  零冲突），62 项触碰路径回归全绿。

## 1. 现状与已知问题（测试前必读）

### 1.1 已交付（commit 清单，rebase 后哈希）

| commit | 内容 |
|---|---|
| 6f1c5d2 | Task 1 账本 `benchmarks/compute_accounting.py` + 6 手算值单测 + 产物 JSON |
| 1fd2ebc | Task 2 裁决驱动 `benchmarks/latent_recursion.py`（预注册判决 + resume + 日志）+ 7 单测 |
| 90d11e8 | fix: 启动即崩的 `args.modes` AttributeError + namespace 完整性回归测试 |
| f73c7fa | fix: resume 必须匹配 steps（防低预算行让正式跑静默跳过）+ Stage-1 产物 |
| a6fa835 | Task 4 `liquid_step_ladder`（config/model/layer 三处 + 纯函数）+ 6 契约测试 |
| 7d29fcf | Task 3 `benchmarks/anytime_frontier.py` + `gen_pointer_chase_cot` 生成器 + 5 单测 |
| 118968e | Task 5 文档（LATENT_RECURSION.md / BENCHMARKS / RESULTS / README） |

### 1.2 核心事实与已知风险（诚实清单）

**已出的唯一硬数字（解析事实，非实验声明）**：probe 口径（T=54）下
潜空间整块循环 N=8 = 157 MFLOPs vs 生成 8 个 CoT token = 33 MFLOPs
（**4.8×**）；潜空间优势在内存（0 KV 增长 + 4.1 KB 恒定状态 vs 104.8 KB
KV 峰值）。"省 token"叙事在 FLOPs 口径**不成立**——这是本分支最重要的
先验，Task 3 预期被压制，压制即如实记录。

| # | 风险 | 归因/处置 |
|---|---|---|
| R1 | 跨设备数字漂移：MPS/CPU/T4 的训练数值不逐位一致 | 协议靠 seeds 不靠设备钉死；行 JSON 记录 device 字段；跨设备**不混表** |
| R2 | grok 预算墙：mix pointer_chase d8 可能 >30k 步才 grok（ROADMAP 参照 ~28k） | verdict 会自诊 `budget_wall`——这是**信息不是 bug**；处置 = 加 steps 重跑（resume 会跳过已完成同预算行，新预算全量跑），**禁止降判据** |
| R3 | 双峰掷硬币区：某档半数 seeds grok 半数 chance | verdict 自诊 `bimodal_grokking_zone` 并拒绝引用；处置 = 加 seeds 重跑该档 |
| R4 | `tests/test_multimodal_clip.py::test_multimodal_endpoint_rejects_bad_base64` 失败 | **先于本分支存在**（缺 PIL 的环境问题，已在基线 commit cad9372 复验确认）；与本 PR 无关，勿记在本 PR 头上 |
| R5 | 共享机器计时噪声 ±20% | 本机有多会话并行抢核；`--probe` 的 ETA 只做量级参考 |
| R6 | CoT 臂评估无 KV cache（baseline 不支持），贪心自回归 = 每 token 全量重前向 | 正确但慢；GPU 上可忽略，CPU 上是已知优化点（不阻塞） |
| R7 | Task 2/3 判决数据未跑 | 本 PR 的预期状态（Phase B），见 §2.2 |
| R8 | resume 粒度是**配置级**（训练内无 checkpoint）：中断即重跑当前配置 | 长配置（stack d8 ≈ 19.5h @MPS）必须跑在 uptime 超过它的设备上；若只有 Kaggle（12h 会话），需先给 `train_model` 加 intra-config checkpoint（本 PR 范围外，`scaling_comparison.py` 的 `--ckpt_every` 模式可抄） |

### 1.3 红线（违反任一即返工）

1. 单 seed / 双峰区结论禁引；判负标准已写死在
   `latent_recursion.judge_decision`，**不得事后移动**；
2. FLOPs 账本口径 2026-08-29 冻结（`compute_accounting.py` docstring），
   不得中途更换；
3. 配对符号检验 3 seeds 最小 p=0.25 > 0.05——**可引用判决必须 6 seeds**
   （`test_latent_recursion_driver.py` 锁死此协议事实）；
4. 不改 `serve/server.py`、不新增依赖；`reasoning_depth.py` 既有签名不动。

### 1.4 关键决策记录（为什么是现在这个样子）

- **裁决对象是 stack 不是 core**：HANDOFF §2.5 教训"只循环液体核心 ≠ 思考，
  组合查找在注意力里"；core 只作阴性对照臂。
- **6 seeds**：见红线 3（数学下限，不是拍脑袋）。
- **潜空间臂用 poisson 深度采样训练**：均匀随机深度已被 §4.5 round-1 证伪
  （教会模型无视迭代）；poisson 是 Geiping 式修复且保持 anytime 兼容。
- **CoT 口径 = 确定性中间符号**（粒度 g 链在 g,2g,...,k 落节点，链尾即答案，
  贪心自回归、错一步链全错）：不用 reason 风格提示——小词表任务上自由文本
  CoT 无定义，确定性符号可解析判分。
- **压制规则平局从严**（flops≤ 且 acc≥ 即算压制）：潜空间叙事本就弱势，
  判据从严才立得住。
- **base 选 main**：PR #2 的 base（`iter/irregular-streaming-edge`）是 KV 线
  WIP；本线自包含，与 main 零冲突（merge-tree 已验证）。

### 1.5 尚未覆盖的盲区（接手人应知）

- τ 阶梯（Task 4）**零训练数据**——按任务书降级条款"开关就绪不出数"；
  Task 2 判决出来后它才有出数资格。
- `tfm_latent_N` 账本行是**纯解析**（Huginn 坐标对齐用），本仓库没有训练
  权重绑定 transformer 循环的模型——它是坐标，不是实验臂。
- CoT 臂每个粒度单独训一个模型，**不是 anytime**（链长由 g 钉死）；只有
  潜空间臂是 anytime 语义。前沿表读数时记得这个不对称。
- 单任务族（pointer_chase）：结论外推到 mod_chain/S5 未做。

### 1.6 文档地图

| 想知道什么 | 去哪读 |
|---|---|
| 设计、机制对比表（Coconut/Huginn/本仓库三列）、全部协议 | `docs/LATENT_RECURSION.md` |
| 数字与状态（对外的） | `BENCHMARKS.md` "Latent recursion" 节 + `RESULTS.md` PREREGISTERED 节 |
| 账本口径（冻结） | `benchmarks/compute_accounting.py` docstring |
| 判决数学 | `benchmarks/latent_recursion.py::judge_decision` + 其单测 |
| 怎么测 / Phase B runbook | 本手册 §2 |

## 2. 怎么测试

### 2.0 fresh clone 路径（测试同事从这里开始）

```bash
git clone https://github.com/everest-an/M1.git && cd M1
git checkout iter/latent-recursion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 或 requirements.lock
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### 2.1 本地门（约 10 分钟，全 CPU）

```bash
# 1) 本分支的门：24 用例（4 文件，协议数学锁 + 位等价 + resume 语义），<1min
python -m pytest tests/test_compute_accounting.py \
                 tests/test_latent_recursion_driver.py \
                 tests/test_anytime_frontier.py \
                 tests/test_liquid_step_ladder.py -v

# 2) 触碰路径回归：62 用例（含 core_iterations N=1 位等价——
#    ladder kwargs 线程不得破坏 stock 路径），<10s
python -m pytest tests/test_core_iterations.py tests/test_deep_supervision.py \
                 tests/test_model.py tests/test_selective_decay.py \
                 tests/test_signed_decay.py tests/test_attention_thinning.py -q

# 3) 账本复现（对照 benchmarks/results/compute_accounting.json，秒级）
python benchmarks/compute_accounting.py

# 4) 任务生成器自测（CoT 链黄金回放）
python benchmarks/reasoning_tasks.py --selftest

# 5) 冒烟（可选，~3min CPU）：30 步全管线 + verdict + 前沿表
python benchmarks/latent_recursion.py --steps 30 --seeds 0 --device cpu \
    --out_dir /tmp/lr_smoke
python benchmarks/anytime_frontier.py --steps 30 --seeds 0 --device cpu \
    --out_dir /tmp/af_smoke

# 6) 全量回归（可选，~3min）：期望 1351+44 过，仅 R4 那一个先在失败
python -m pytest tests/ -q --deselect \
    tests/test_multimodal_clip.py::test_multimodal_endpoint_rejects_bad_base64
```

### 2.2 远程全量（Phase B runbook — GPU 判决跑）

**先测速再决定切分**（任何设备）：

```bash
python benchmarks/latent_recursion.py --probe --device <cuda|mps> --probe_steps 20
python benchmarks/anytime_frontier.py  --probe --device <cuda|mps> --probe_steps 15
```

实测参照（MPS，本机多会话抢核，±20%）：Task 2 全协议（2×4×6 配置 × 30k
步）≈ 400h；Task 3（潜空间 1 + CoT 4 臂 × 6 seeds）≈ 120h。**Kaggle T4 未见
实测，先跑 probe 折算**。切分算术（按 seed 或按配置切会话均可，resume 语义
保证跨会话拼接——行 JSON 的 steps 匹配才跳过）：MPS 单配置耗时 ≈
core {5.5, 4.5, 8.3, 16}h / stack {2.6, 5.0, 8.1, 19.5}h（d1→d8）——
**注意 R8：stack d8 ≈ 19.5h 超过 Kaggle 12h 会话上限**，见风险表处置。

```bash
# 判决跑（Task 2）——每配置完成即落盘，可随时中断续跑
python benchmarks/latent_recursion.py --steps 30000 --seeds 0 1 2 3 4 5
#   → benchmarks/results/latent_recursion_{core,stack}_d{1,2,4,8}_s{0..5}.json
#     + latent_recursion_verdict.json（自动判：h_supported / diagnosis 四选一）

# 前沿跑（Task 3）
python benchmarks/anytime_frontier.py --steps 30000 --seeds 0 1 2 3 4 5
#   → anytime_{latent,cot_g*}_s{0..5}.json + anytime_frontier.json（含压制判定）
```

**判读（不移动门）**：verdict JSON 的 `h_components`（monotonic /
gain ≥ 2σ）与 `gate.reasons`（E0 门槛）；前沿 JSON 的
`latent_points_dominated`。budget_wall → 加 steps；bimodal → 加 seeds；
两个脚本 resume 重跑即可，已完成配置零浪费。

### 2.3 通过标准（"全绿"的定义）

1. §2.1 第 1/2/3/4 步全过（24 + 62 用例 + 账本逐字节可复现 + 自测 OK）；
2. §2.1 第 5 步冒烟产出结构合法的 verdict/frontier JSON（全 chance 是
   预期，`diagnosis=budget_wall` 是**正确**自诊）；
3. 全量回归仅 R4 一个先在失败（基线归因已给）；
4. Phase B 产物 JSON 齐全（8 档 × 6 seeds × 两实验）且 verdict 不含
   "reasons" 空缺。

## 3. 关键值：测什么、记什么、落在哪

| 产物 | 落点 | 记什么 |
|---|---|---|
| 算力账本 | `benchmarks/results/compute_accounting.json` | 5 路径 × N 的 MACs/FLOPs/权重流量/KV 峰值/状态字节（口径冻结） |
| Task 2 行 | `benchmarks/results/latent_recursion_{mode}_d{d}_s{s}.json` | acc_by_k / mean_acc / wall_s / s_per_step / device / git_rev |
| Task 2 判决 | `benchmarks/results/latent_recursion_verdict.json` | h_supported / h_components / tiers / gate / diagnosis |
| Task 3 行+前沿 | `benchmarks/results/anytime_*.json` | acc_by_k 或 acc_by_depth / mean_acc / frontier points + 压制判定 |
| 训练日志 | `benchmarks/results/latent_recursion_*.log`、`anytime_*.log` | 时间戳 start/done + step/loss tee |

## 4. 改动明细与设计理由（审阅对照用）

| 文件 | 改动 | 设计理由 |
|---|---|---|
| `mt_lnn/config.py` | +`liquid_step_ladder`/`liquid_step_kappa` + 校验 | 默认 False 全路径不进新分支；开启时 kappa>0 才校验（关闭时不烦人） |
| `mt_lnn/mt_lnn_layer.py` | resonance `ladder`/`ladder_kappa` 字段 + `ladder_scale_bias` 纯函数 + blend logits 偏置 | 偏置加在 softmax 前、dynamic_kappa 前——与既有门控正交；τ 数学 fp32（与 decay 同纪律） |
| `mt_lnn/model.py` | block `ladder_base` kwarg + core 循环 `ladder_iter=ladder_base+1+i` + stack 循环 `ladder_base=pass×core_iters` | 纯 kwargs 默认 0——**不改任何既有调用方**；第 0 次迭代无偏置 → ON+N=1 位等价（测试钉死） |
| `benchmarks/reasoning_tasks.py` | +`gen_pointer_chase_cot` + `_cot_hop_marks` + selftest 扩展 | 增量扩展（任务书红线 2）；确定性中间符号口径写死在 docstring |
| `benchmarks/latent_recursion.py` | 新文件 | 复用 `build_mtlnn/train_model/evaluate_per_k/make_mix_generator` + E0 协议件；只做编排与判决 |
| `benchmarks/anytime_frontier.py` | 新文件 | CoT 训练循环镜像 `train_model` 优化器卫生（β2=0.95/clip 1.0），唯一差异 = 标签覆盖整链 |
| 测试 ×4 文件 | 24 用例 | 只锁不变量/边界/易回归点：协议数学、位等价、resume 语义、账本对账 |

## 5. 附录：PR 状态

见 GitHub PR（base `main`，head `iter/latent-recursion`）。Phase A 零实验
声明；Phase B 数据回填后 RESULTS.md 的 PREREGISTERED 节转 Proven/Null。

**回填状态（2026-08-31）**：Task 2 已转 **Null（budget_wall）**——RESULTS.md
PREREGISTERED 行已更新，48 行 + 日志 + 重算 verdict 已入库
`benchmarks/results/`；Task 3 仍 PENDING（见 §0 回填块）。
