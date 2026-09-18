# M1 多级算力架构 — 层级、路由与工程约定（B-24）

> 2026-09-16 · 社区扫描（SkyPilot/dstack/Modal 谱系）+ L008/L010 制度化落地。
> 路由器：`scripts/tiered_run.py`（v0，零新依赖）。

## 一、层级表

| 层 | 目标机 | 窗口 | 代价 | 适用 | 铁律 |
|---|---|---|---|---|---|
| T0 | 本机 M2 Max | 无窗 | 免费 | est≤1h 快速校验/探针/判读 | L008：>1h 禁止本机 |
| T1 | Kaggle CPU（4 核） | **12h 硬窗** | 免费（不烧配额） | CPU 批量长跑（如 tau-ladder） | L010：4 lane × OMP 钉死；按配置原子落盘 resume-safe |
| T2 | Kaggle T4 | 9h 窗 | ~30h/周配额 | 小型 GPU（T4 16GB 装得下） | 配额耗尽自动降级 T3 |
| T3 | 服务器 A100 40GB（SSH） | 无窗（常驻） | 已部署 /root/M1_run | 大 VRAM / 长跑 / 正式实验 | rsync 收割；nohup+日志落盘 |

## 二、路由规则（tiered_run.py --tier auto）

- device=cpu：est≤1h → T0；>1h → T1
- device=gpu：require_a100 或 est>9h → T3；否则 T2
- 任何层级可用 `--tier` 显式覆盖；T1/T2 缺 kernel 目录、launcher 未钉死
  OMP 线程 → 拒绝执行（L010 检查项）

## 三、各层 runbook

- **T0**：`tiered_run.py --cmd "..." --est-hours 0.2` 直接执行并计时。
- **T1/T2**：先建 `kaggle_kernels/<name>/{kernel-metadata.json, run_*.py}`，
  路由器校验后 `kaggle kernels push`；截断后同轮
  `kaggle kernels status` + `kernels output` 收割，重推即 resume 续跑
  （tau-ladder 三会话实例：V12→V14→V15）。
- **T3**：cmd 经 base64 传入（引号安全），`nohup bash .job.sh` 落日志；
  收割 `rsync -av m1-a100:/root/M1_run/benchmarks/results/ benchmarks/results/`。

## 四、社区谱系（为什么自建）

SkyPilot（跨云套利 + on-prem）/ dstack（on-prem+云控制面）/ Modal
（serverless、贵、锁单云）是社区三主流；个人常见配置 = "训练 SkyPilot +
推理 Modal"。**三者均不覆盖 Kaggle 免费档**，而 Kaggle CPU/T4 是 M1 的
T1/T2 主力 → 自建轻路由器（零依赖、~150 行）而不是引入 SkyPilot 依赖。
若未来接入竞价云（RunPod/Vast），再评估 SkyPilot。

## 五、已验证

- 四层 dry-run 路由 ✓；L010 负例拦截（无钉死 launcher 拒绝）✓；
- T0 实跑 ✓；T3 base64 引号安全 ✓。
- 首个真实用例：B-2 tau-ladder（T1，V14 3-lane 教训 → V15 4-lane 钉死版）。

## 六、跨会话/跨主机 checkpoint 契约（L010/L011 闭环）

训练状态 = **按配置原子的结果 JSON**（tau_probe_*.json）。汇聚点 = Kaggle
Dataset `aricredemption/m1-tau-ladder-ckpt`：

- **harvest（本地，有凭证）**：拉取 kernel 输出 JSON → 并入
  `benchmarks/results/tau_ladder/`（git 存档）→ 自动
  `datasets create/version` 推送 checkpoint（失败仅警告降级，不阻塞）。
- **launch（任意主机）**：metadata.dataset_sources 挂载 checkpoint 数据集；
  launcher 启动时把挂载里的历史 JSON 并入 working/results → probe 的
  resume 跳过生效 → **跨会话/跨主机不重跑已完成配置**。
- **降级链**：checkpoint 数据集缺失/DNS 故障 → launch 自动剥离挂载引用
  （本会话无 resume，其余照常）；harvest 推送失败 → 警告，git 状态安全。
- 已验证：403 缺失检测 ✓、剥离降级 ✓、base64 T3 ✓。

## 七、多机训练的边界说明

B-2 类探针实验的并行是**配置级**（18 配置互相独立），不是梯度同步级——
多主机 = 多主机各领一批配置 + checkpoint 汇聚，无需 NCCL/torchrun。
需要真梯度同步的大训练（M2 2B 线）走 T3 A100 单机多卡（fsdp 已在 #57
骨架内），属另一条线。

## 八、实测吞吐与分配策略（2026-09-17，用户质询驱动）

**本机实测**（M2 Max：12 核 CPU 8P+4E + 38 核 GPU + 32GB 统一内存）：
合成训练步（3×Linear256+GELU，batch128）——CPU 3.39 ms/step、**MPS 2.40
ms/step（快 1.4×）**；MPS 可用 ✓（torch 2.13）。对照：Kaggle CPU 容器实测
~667 ms/step（V14，3 进程争抢）——**本机 MPS 比 Kaggle CPU 快 ~280×**。
（注：合成基准非探针原模型，指示性非精确。）

**关键结论：Kaggle CPU（T1）是四层中最慢的一层**，其唯一优势 = 挂机不占
本机 + 免费。本机 MPS 被严重低估。

### 分配策略（按 workload 类型）

| workload | 路由 | 理由 |
|---|---|---|
| PMB 类秒级评测/判读 | T0 本机 CPU | 出门就是浪费 |
| B-2' 硬化探针单配置校验（10K gate） | **T0 本机 MPS（~40min）** | 上大批量算力前必须先验证任务落在深度敏感带 |
| 硬化探针 18 配置矩阵 | **T2 Kaggle T4（30h 周配额全空）** | 预计一个 9h 会话跑完，不占 A100 |
| 2B fsdp / 大 VRAM 正式跑 | T3 A100 | 唯一合身层 |
| 纯 CPU 挂机型长跑 | T1 Kaggle CPU | 仅当不占本机是硬需求 |
| TPU 20h | **暂不使用** | MT-LNN 自定义结构 XLA 适配成本高；每周刷新不累积，闲置即可 |

### L008 修订提案（待用户裁定）

MPS 训练任务与 L008 原始痛点（CPU 时间片 + swap 污染）不同：GPU 侧计算
几乎不占 CPU 时间片。提案：**MPS 任务上限放宽至 2h**（探针 30K 步 ~2h
可全本机完成）；CPU 任务维持 1h 红线。裁定后更新 L008 与本表。

## 九、Kaggle CPU 与 TPU 的充分利用评估（2026-09-17）

### Kaggle CPU：**能用好，杠杆是并发数**（已落地 `kaggle_kernels/cpu_sweep/`）

单机（4 核容器）确实慢（~667ms/step vs 本机 MPS 2.4ms），**但 CPU 会话可以
多 kernel 并发**（数十核聚合，免费无配额）。适用画像 = **大量小而独立的
CPU 友好任务**：

- **任务-难度校准扫描**（B-2' 前置）：硬化 τ 阶梯需要先找到"depth=1 学不
  会、depth=4 能学会"的任务点——difficulty {8,16,32,64} × 多变体 × gate
  探针（10K 步，~1.8h/条）≈ 40 条 = 72 lane-hours，5 个并发 kernel 一个
  窗口跑完。这类"找工作点"扫描正是 CPU 池的最佳用途。
- **批量评测/验证/复现**：矩阵级秒级任务本地更快，但数百条 accumulate 后
  （如全任务 × 全系统 × 全 seed 的回归验证），挂到 Kaggle 过夜跑不占本机。
- **落地**：`kaggle_kernels/cpu_sweep/`（通用 sweep 模板：4 lane 工作窃取 +
  按输出文件 resume-safe + L010 钉死），sweep.json 换队列即复用。

### Kaggle TPU：**当前组合下诚实的答案是不能**，20h/周维持闲置

- **模型规模不匹配**：MT-LNN 探针级模型（~10⁵-10⁶ 参数）单步计算量太小，
  TPU 的优势在超大 batch 矩阵吞吐——小子模型上 per-step 调度开销反而主导，
  大概率比 GPU/CPU 更慢。
- **移植成本不匹配**：torch_xla 需要静态 shape + XLA 兼容算子，MT-LNN 的
  液体核自定义结构适配成本以周计；M2 2B 线已有 A100 fsdp 骨架（#57），
  TPU 是第二套并行方案的重复建设。
- **重启条件（两条任一满足再评估）**：① 2B 预训练线 A100 成为瓶颈且需要
  第二条大算力 lane；② 出现天然的 XLA 友好工作负载（大 batch 静态 shape
  训练）。在此之前 20h/周闲置是**正确的浪费**——用错地方才是真浪费。

## 十、TPU lane 激活实测增补（2026-09-18，ADJ-012）

**§九 "TPU 闲置=正确浪费" 被本节取代**：其重启条件②（天然 XLA 友好负载）
成立，移植成本已付清。实测（`m1-tpu-feasibility-probe` v2 + 本地等价验证）：

| 项 | 实测值 |
|---|---|
| TPU v5e 单核 medium d4 | **0.25 s/step**（机队最便宜重批 lane；10K 步/配置 ≈ 42min） |
| XLA 编译税 | ~70s/配置（36s+34s），故 **<5K 步配置禁上 TPU** |
| GPU∥TPU 并发 | ✓ 可同跑（修正旧"一会话"民俗）；真限制=**TPU 批会话 1 个/账号** |
| 等价性 | train_xla 与 bundle train_model 逐行镜像，CPU 回放 loss 逐位一致 |

**路由增补**：固定深度 ≥5K 步 → TPU；动态深度/每步换形状 → eager lane
（T4/本机，XLA 逐形状重编译）；两者超会话窗 → §六 接力契约（kernel_sources
挂载上游 output + resume-skip，B-2″ T4→TPU 链实证 9/9 skip）。
8 核利用审计排队中（`m1-tpu-core-audit`），判 TRUE_PARALLEL 前 TPU 按单核核算。

**跨级复用 R1-R4**（防"训了又训"，正文见 ADJ-012）：R1 权重不跨级（渐进
训练作 treatment 须预注册）；R2 结果 JSON 跨级 resume-skip（§六契约的规则化）；
R3 探真产出=配置清单+标定数非模型；R4 判决 seed 与探真 seed 不同源
（催生事件 L012）。判据一句话：复用为"少跑配置"→复用；为"同配置少训几步"→不复用。

**本机口径**：双 Mac = **一条可变先验门**（手边一台，M1 Pro 吞吐未标定，
计划 200 步 d4 medium 同探针定标）；任何机器可作控制面客户端，状态不落本机。
