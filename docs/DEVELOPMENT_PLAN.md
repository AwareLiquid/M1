# MT-LNN 研发计划（基于 2026-08 GPU 证据）

> 制定日期：2026-08-15 · 来源：ABLATIONS.md 入档 + benchmarks/results/*.jsonl 全量分析
> 接续入口：先读本文，再读 HANDOFF.md 与 docs/ROADMAP_M2.md

## 战略总览

**放弃**：原生 LM 质量赛道（实测输 modern Transformer 11.3%，追平需 10 年生态工程）
**主攻**：三条 Transformer 结构上无法进入的资产

| 资产 | 现状证据 | 计划中的角色 |
|---|---|---|
| ① selective_decay 电路表达力 | d16 parity 3/3 vs 0/3（唯一干净分离） | **论文核心主张**（需从玩具桥接真实任务） |
| ② O(1) 推理状态 | 0.381MB 恒定，8063×@1M | 效率曲线主张 + 端侧产品 |
| ③ 跨窗口/会话记忆 | 0.56 vs 0.000，bit-exact 快照 | 产品差异化（记忆型会话） |

**论文三主张（重构后）**：S1 电路表达力分离（selective vs stock）、S2 O(1) 状态效率、S3 跨会话持久记忆。PPL 只作为"训练稳定性"佐证，不作为质量主张。

---

## P0 阶段：证据加固与任务桥接（第 1–2 周，本地 RTX 5060 8GB）

### E0 · 实验协议 wrapper（第一天，所有实验的前提）
固定好配方、强制 ≥3 seeds、结果落 jsonl 前自动跑配对检验、groking 双峰检测（acc 分布 bimodality 检查）。目标：**从今以后不再产出"单 seed 结论"**。

### E1 · parity 分离加固实验 🔴 最高优先级
- **假设**：selective_decay 在 k≥16 对 stock 的分离在多 seed 下稳健（p<0.05）
- **协议**：`benchmarks/reasoning_depth.py`，隔离纯 LNN 核心（attention_layers=[]），k=16/32/64 × 3 难度，6 seeds，6000–8000 steps，lr=3e-4，beta2=0.999/clip=0（好配方），每 arm 独立进程
- **验证**：每难度 sel vs stock 的 grok 率 Fisher 检验 + 配对 sign-test
- **决策门 G1**：k=16 分离不稳健（p>0.05）→ parity 路线降级，主打 S2/S3；稳健 → 进入 E2
- **产出**：reasoning_depth.jsonl 新行 + ABLATIONS.md 入档

### E2 · 真实算法任务桥接（把玩具变成能力）
- **任务池**：动态计数（running count，需输入依赖转移）/ 状态跟踪（变量赋值后查询，防作弊）/ 流式聚合（running sum/max）
- **协议**：selective / stock / vanilla Transformer 三方，3 seeds，好配方，全难度扫描
- **决策门 G2**：任一任务分离 → selective_decay 晋升论文 Figure 1；全 null → parity 仅作理论证据
- **产出**：benchmarks/reasoning_tasks.py 扩展 + 新 jsonl

### E3 · consciousness-m1-v2 分支对比（免费信息）
- 拉取分支，diff 其 parity 实现 vs main；确认其 k、预算、init 与 selective_decay 的关系
- **产出**：实现差异报告。若其 3/3 在大 k → 实现细节是 E1/E2 的优化线索

---

## P1 阶段：预算敏感实验（第 2–4 周，Kaggle P100 / AutoDL A100）

### E4 · k=32 长预算裁决（Kaggle 队列 D）
- 12K steps，3 seeds/arm，P100-cu118（已解锁）
- 决策门：sel grok>0 而 stock=0 → 理论主张确认；双方都 0 → 记录"预算墙"

### E5 · 长度外推加预算（最有叙事价值）
- 依据：sel s2 的 L48=0.749（1.5×训练长度）是 6 runs 唯一正向信号
- 协议：parity_lengthgen.py，curriculum L=1→32→64，100K steps，6 seeds，好配方
- 决策门 G3：仍无干净外推 → 正式归档 null；≥2/6 sel 外推到 2× → 晋升 S2 核心证据

### E6 · 文本收益重测（最后机会，配方修正后）
- 依据：text_ab v4 的 8000 步 PPL≈470 说明配方根本没训起来，null 结论无效
- 协议：好配方 + 125M（非 10M）+ ≥20K steps + 3 seeds；先 2000 步 sanity check（PPL<150 才投长预算）
- 决策门 G4：仍 null → 永久归档"文本 LM 无收益"；正 → 超预期资产

---

## P2 阶段：论文与产品（第 3–6 周，与 P1 并行）

| 工作包 | 内容 | 依赖 |
|---|---|---|
| W1 论文重构 | 三主张框架；负结果章节作为可信度资产 | E1-E6 |
| W2 效率曲线基准 | 每 token 成本 × 每字节状态的 Pareto 前沿（官网已有成本定位图开端） | E2 + S2 |
| W3 端侧闭环 | 1.27MB int8 + 5KB 恒定状态 + WebGPU 浏览器推理，绑定真实流式场景 demo | 独立可先行 |
| W4 开源传播 | E0 协议 wrapper 作为"诚实实验协议"卖点开源 | E0 |

---

## 里程碑与决策门总表

| 时间 | 里程碑 | 通过标准 | 失败预案 |
|---|---|---|---|
| W1 末 | M1: E1 加固完成 | k=16 分离 p<0.05 | 转 S2/S3 主打 |
| W2 末 | M2: E2 桥接完成 | ≥1 真实任务分离 | parity 降为理论证据 |
| W3 末 | M3: E4+E5 出结果 | 大 k grok 或长度外推其一 | 归档 null，论文走"诚实负结果+O(1)" |
| W4 末 | M4: E6 定论 | 文本收益二选一定案 | 明写 null |
| W6 末 | M5: 论文 v1 + 效率曲线 | 三主张各有一张图 | — |

## 资源分配

- **本地 8GB**：E1、E2、E3、E0（parity 单 run 500–800s，6 seeds ≈ 1.5h）
- **Kaggle P100**：E4（队列已排）、E6 长预算
- **AutoDL A100**：E5（100K steps 唯一重负载）
- **算力原则**：<3h 本地，>3h 上云（HANDOFF 教训：本地不跑 20K 级长跑）

## 风险清单（每条都有预案）

1. grokking 双峰 → E0 强制多种子 + bimodality 检测，双峰区数据不引用
2. 配方否决 → 所有实验行内记录 beta2/clip，好配方为默认
3. 单种子幻象 → 引用前 ≥3 seeds + 配对检验（M1 硬门槛）
4. 预算墙 → curriculum 冷启动（E5 已验思路）
5. Windows 工具链（torch+pyarrow DLL 冲突） → 数据用 .txt，每 arm 独立进程
6. 分支漂移 → E3 只做 diff 分析，不合并不盲搬数

## 执行顺序（按依赖）

1. E0（解锁可信度）→ 2. E1（本地）→ 3. E3（本地 diff）→ 4. E4（Kaggle 推进）
