# notes/overnight/backlog.md — 夜间迭代方向池

> 状态机：CANDIDATE(方向+外部扫描来源) → PREREGISTERED(判负标准+最便宜实验方案)
> → RUNNING → DONE(结果 JSON 路径) / DEAD(判负触发，注明证据)。
> 判负方向标 DEAD 带证据，绝不在同方向重开（§4.6 方向版止损）。
> 价值排序：前沿学术化（学界会引用吗）> 被验证过的商业模式（有人用真金白银验证过吗）；
> 两档皆无不为凑数登记。
> **预注册判据模板（2026-09-16 修订，B-21/B-22 缺口驱动）**：判负标准必须
> **双侧写死**——①正向改善阈值（→DONE/立项）；②**显式负向变化分支**
> （指标下降 ≥ 阈值同样触发判定路径，负变化不得落在字面区间外）；
> ③中间带（→边界登记）。跨时点对照必须同跑（L009，
> kb/lessons/L009-same-run-cross-time-comparison.md）。
> GPU 通道状态（大实验前置条件，状态变化在此登记，提示词不写死）：
> 2026-09-09 阻塞——Kaggle Secrets 未配 GITHUB_TOKEN 且 kernel 凭据修复 PR 未合并；
> 两项齐备后把本行改为"可用"。
> 初始化：2026-09-09 首航（手动触发，非窗口时段）。

---

## PREREGISTERED（可直接执行）

### B-1 · text_selective 配对实验的预算-增益曲线（32K 步判别点）· DEAD（2026-09-10 服务器 GPU 执行）
- **假设**：selective_decay 相对 stock 的 PPL 增益随训练预算放大（2000→8000 步
  配对全优的趋势是否延续到 32K）；若 32K 仍不显著，"预算放大增益"方向判 DEAD，
  selective_decay 的文本价值退回"小预算探针现象"。
- **判负标准（预注册）**：32K 步 3-seed 配对中 selective 胜 ≤1/3 且 sign-test
  p > 0.05 → 标 DEAD（证据 = 32K 行 JSON）；反之转 DONE 并升格 E 队列候选。
- **执行（2026-09-10）**：用户定调"本机只跑 ≤1h"→ 迁 **服务器 A100 GPU**（政策
  解锁）：M1 部署 `/root/M1_run` + wikitext 语料；6 组合(sel×seed)×32K 步，
  ~29min/组合（GPU 99%），全程 ~3h。
- **首手结果（val_ppl @32K，3-seed 配对）**：
  - stock: 388.854 / 389.360 / 383.085 → **387.100 ± 3.486**
  - selective: 386.168 / 390.248 / 382.040 → **386.152 ± 4.104**
  - 配对差(sel−stock): −2.69 / +0.89 / −1.05 → **2/3 selective 优，mean −0.95**
  - **sign-test 单尾 p = 0.5（>>0.05，不显著）**
- **判定**：**DEAD**——按预注册假设"32K 仍不显著 → 判负"，p=0.5 触发。
  **边界注**：判负标准字面条件"selective 胜 ≤1/3"未满足（实为 2/3 胜），
  即 selective 略正向但**统计不显著**；8K（胜 1/3, p=1.0）→32K（胜 2/3, p=0.5）
  胜率上升但增益未达显著，方向仍判负。
- **结论（可引用）**：selective_decay 的文本 PPL 增益不随预算放大到统计显著；
  其价值不在文本 PPL（而在长度外推等机制）。
- **证据**：服务器 `/root/M1_run/benchmarks/results/text_selective_ab.jsonl`（6 行）
  + `/root/b1_gpu.log`；GPU 结果与本地 CPU（388.0/389.6）复现一致。
- **来源**：HANDOFF §8 v4 文本续训计划；ITERATION_PRINCIPLES P1。

### B-2 · tau-ladder 探针 · DEAD（flat/天花板，A100 通道 2026-09-16 结案；见下方结案记录）
- **结案与通道更替（2026-09-16 21:15 发现）**：另一会话按用户指令将本方向
  通道由 Kaggle 换为 A100，18/18 配置当日完成并判负——结案记录见下方
  "B-2 结案记录"（本文件 61 行起）。本条目保留 Kaggle 通道的过程记录与
  教训（V14 僵尸 12h + metadata 缺挂载 → L010/L011；V16 4 lane 优化版
  22:53 自然截断，其产出与 A100 证据重复，不重复入库）。
- **当前状态（2026-09-16 02:05 CST 核查）**：V14（kernel `m1-b2-tau-ladder-cpu`）
  RUNNING——9/15 15:04 UTC 起跑，进行中（kaggle CLI 实查）。
- **状态（2026-09-14 23:0x）**：V12 RUNNING，6/18 配置完成（全 OFF 臂基线 acc≈1.0），剩余 12 个 τ 阶梯 ON 臂配置继续推进。
- **进展**：Save & Run All 强制重启后首次成功执行（v5-v11 均卡队列未实际运行），6 个 OFF 臂基线已产出（acc 0.9995-1.0）。
- **待跑 12 个**：ladder=ON × depth{1,2,4} × seeds{0,1,2} = 9 个 + OFF 臂剩余 3 个。
- **resume 保障**：每配置原子写 JSON，12h 上限截断后重推自动跳过。
- **前置阻塞**：#53 未合并 + Kaggle Secrets 未配 GITHUB_TOKEN（用户人工步骤）。
- **判负标准（预注册）**：18 配置中 parity 域配对（d2/d4/d8 × 3 seeds）在
  30K 步预算内 acc 全部 < 0.25 → tau-ladder 判定任务不可行，ADJ-001 的
  替代判决任务线标 DEAD；learnability gate（10K 步 acc>0.2）作为会话内止损。
- **最便宜实验**：分片推送（单实例 ≤2 臂，ADJ-004），T4 ~4h/30k 步趟。
- **验收门**：kernel 输出 JSON 与本地复算逐字节一致 + publishable() 纪律。
- **来源**：ADJ-001（budget_wall 脱出）、kb/threads/T001、kaggle_kernels/tau_ladder_probe/。

### B-2 结案记录 · tau-ladder 探针 · DEAD（flat / 天花板，2026-09-16 A100 通道）
- **执行**：18 配置（2 ladder × 3 depths × 3 seeds）× 30K 步；**通道由用户指令从
  Kaggle 换为 A100 服务器**（协议/判据不变，仅换算力通道，4 workers CPU+GPU 并行）。
- **结果**：18/18 配置 mean_acc ≈ 1.0——OFF d1/d2/d4: 0.9997/0.9998/0.9998；
  ON d1/d2/d4: 0.9997/0.9999/0.9998。**全饱和**。
- **判决**（judge_verdict，预注册判据写死）：h_supported=false，delta_gain=0.0
  （阈值 2σ=0.0002），diagnosis=**flat**，配对符号检验 p=1.0。
- **诚实标注（重要）**：此 flat 是**天花板效应**，非"τ 阶梯无效"的干净判负——
  parity d16 在 depth=1 即达 1.0，深度维度无区分度，τ 阶梯无增益可测（与 M2
  mod_chain 天花板同因）。**该协议无法检验 H004**；需"depth=1 学不会、depth=4
  能学会"的硬化任务（更高 difficulty 或 pointer_chase 类查表域，判据同款）重测。
- **证据**：`benchmarks/results/tau_ladder/`（18 配置 JSON + verdict JSON + README）。
- **后续**：B-2' 难度硬化 τ 阶梯探针（新条目，预注册判据同款）。

### B-2' · 难度硬化 τ 阶梯探针（H004 可检版）· 筛选判负（depth-flat，2026-09-17 A100）
- **预注册（overnight/loop，合并自本分支）**：硬化方向 = 更高 difficulty 或
  pointer_chase 类查表域，目标 = "depth=1 学不会（gate<0.55）、depth=4 能
  学会"；判据 judge_verdict 同款（depth 增益 ≥ 2σ 且 ON/OFF 分离 →
  h_supported；flat/全墙 → 族判负，§4.6 不重开）。方法升级：staircase 二分
  校准替代盲扫（Faith & Fate 曲线 / grokking 相变 / 心理物理阶梯三依据）。
- **校准（T2 T4，20K 步，seed 0）**：k=32 处曾现 ON 深度增益 +0.054 的
  单 seed 信号（staircase 有效性初证）。
- **筛选结果（A100 通道，2 ladder × 2 depths × 1 seed，30K 步）**：
  - OFF d1=0.9183 / OFF d4=0.9160（gain −0.0023）；ON d1=0.9183 / ON d4=0.8994（gain −0.0189）
  - verdict：**flat**——d32 打破了天花板（acc≈0.90 而非 1.0），但 **depth 1 = depth 4**，
    parity 任务在任何难度下都不奖励深度，τ 阶梯无增益可测。
- **信号对账（L009 精神）**：校准的 +0.054 单 seed 信号未在 A100 30K 筛选
  （−0.019，1 seed）中存活——不同预算/硬件的跨时点对照不可比，筛选判负
  以 30K A100 为准。
- **结论（累计）**：d16 天花板 + d32 深度平坦 → **parity 家族无法检验 H004**。
  剩余判别任务 = pointer_chase 类查表（M2 已验证的判别器），但需 medium 规模（17.7M）
  才可靠（probe 规模高方差）——**medium × 18 配置 × 30K 步是较大 GPU 投入，待用户拍板**。
- **证据**：`benchmarks/results/tau_ladder/d32_screen/`（4 配置 + verdict，服务器）；
  `benchmarks/results/tau_staircase/`（T2 校准 12 配置）。

## CANDIDATE（待外部扫描后转正）

### B-3 · 跨会话 fast-weight 记忆 vs 检索基线的同任务对打（P1 主轴升级）· CANDIDATE（受限）
- **扫描结论（2026-09-09 二航）**：LongMemEval（ICLR'25，500 题 5 能力）与 PMB
  的映射：IE≈T1、KU≈T2、MSR/Abstention 为 PMB 缺口（t4/t5 候选）、遗忘曲线为
  PMB 独有。LongMemEval_S ≈115K tokens/50 sessions，LLM 生成会话，非单夜 CPU
  可落地；其完整 harness 集成属大工程。
- **阻塞**：对打实验的 fastweight 臂需要 GPU 微调的 adapter checkpoint
  （README:127-129 明示 interface skeleton）——GPU 通道解锁前无法执行。
- **升级路径**：GPU 解锁后 → fastweight 臂跑 T1/T2/T3 vs rag（即 B-4 判负实验）
  → 若存活再考虑 LongMemEval_S 子集对打。
- **来源**：ITERATION_PRINCIPLES 附录 B 弱点 1；本次扫描（上方映射）。

### B-4 · persistent memory Zoology 基准族（主轴 b）· PREREGISTERED（范围修正 2026-09-09 二航）
- **⚠️ 首航登记修正（L001 实例）**：首航把本方向登记为"PMB 加 D5 遗忘曲线任务"，
  二航盘点发现 **PMB 已有 T1/T2/T3 三个任务族**（跨会话保持网格 / 流式更新召回
  / 遗忘曲线全曲线，benchmarks/persistent_memory/README.md）——正是 Zoology 生态位
  需要的三件套，D5 是重复造轮子。修正后范围：加固 + 扩种子/系统跑通既有任务族
  + 开源定位，不新增任务。
- **外部扫描证据**：Titans Revisited（arXiv:2510.09551）只测 MLM/时序/推荐，
  缺口仍开放；LongMemEval（arXiv:2410.10813, ICLR'25）五能力中 Information
  Extraction ≈ T1、Knowledge Updates ≈ T2，Multi-Session Reasoning 与 Abstention
  是 PMB 缺口（未来 t4/t5 候选），**遗忘曲线全曲线是 PMB 有而 LongMemEval 没有
  的维度**（M1 的差异化贡献点）。
- **区分度预注册判负标准**：none/rag 不分离（rag ≈ none）或 oracle 无窗口塌陷
  现象 → "PMB 可作 Zoology 基准"判 DEAD。**二航实测：判负未触发**（none 0.0 /
  rag t1 难格 0.839 / oracle 超窗塌陷 1.0→0.0，见 B-6）——基准有区分度，方向存活。
- **最便宜实验**：fastweight 臂补跑（GPU adapter 产物到位后）对打 rag 参照
  （T1/T2/T3 全网格，3-seed 配对 sign-test）；判负标准 = fastweight 不优于 rag。
- **来源**：ITERATION_PRINCIPLES 附录 B START ④；两次扫描（上方证据）。

  github.com/yifanzhang-pro/fast-weight-attention）提出 Falcon-1/2/3（NLMS/逐列/
  滑窗 mini-batch 的一阶 fast-weight 更新）与内积变体，含数值稳定的正衰减重归一化；
  实验 = 语言建模 + 变长数字加法长度外推。同域热度：TTT Done Right (2505.23884)、
  In-Place TTT (2604.06169)、Elastic TTT (2604.07350)——**fast-weight 更新规则是
  2025-2026 活跃前沿**。
- **重叠度判定**：Falcon 系研究**序列内**的读后写更新规则；"per-user/per-session
  fast-weight state"仅为其应用展望，**未做跨会话持久化（bit 级序列化/恢复）语义，
  未做跨会话基准**——登记时主轴 (a) 缺口开放，但窗口收窄（3 篇/14 个月）。
- **判负标准（预注册）**：Falcon-1（NLMS，d=256，B-6 同网格）在 T1 难格 (64,16)
  与 T2 上 3-seed 均值 recall ≤ rag(hash) 0.839 - 0.10 = 0.739 → DEAD。
- **执行与判定**：#55（另一会话）实现 falcon_nlms 写规则并入 parametric_memory.py
  （seed 0：recall 0.0 vs sum 规则 0.406——单 seed screening）；夜班补齐 s1/s2 +
  sum 对照 s1/s2（各 0.5s CPU，不违反让路条款）：**falcon_nlms 3-seed 难格
  recall = [0.0, 0.0, 0.0]，均值 0.000 ≤ 0.739 → 正式 DEAD**。sum 规则 3-seed
  均值 0.385（同样低于检索式 0.839，但作为参数化系统内部对照有效）。
- **证据路径**：benchmarks/results/pmb_falcon/falcon_nlms_t1_s{0,1,2}.json +
  sum_t1_s{0,1,2}.json；实现入 mt_lnn/parametric_memory.py（#55）。
- **学术结论（可引用）**：*"The Falcon-1 NLMS update rule, competitive on
  in-sequence LM tasks, achieves 0.0 recall on cross-session retention at
  O(d²) state — update-rule strength does not transfer to persistence
  semantics"*——更新规则强度 ≠ 持久化语义，反向支撑主轴 (a) 的定位：
  缺口在持久化机制而非更新规则。
- **来源**：本次扫描；对照主轴 (a)；价值档 1。

### B-5 · 首航观察衍生：探索性脚本无 argparse 保护
- **观察**：`benchmarks/text_selective_ab.py` 无 argparse，`--help` 会直接开训
  （试航实测，已截停）。同类历史脚本可能不止一个。
- **候选动作**：夜间班次如需复用历史脚本，先读源码确认入口（L006 的执行细则）；
  是否系统性加 argparse 属于 mechanism 轨小改动，可作为白班候选，夜间不做
  （避免动无关文件）。
- **来源**：L006（复用历史脚本前先读隐性约束）——本条是 L 工件触发条件的实例。
- **白班执行（2026-09-10）**：扫描 benchmarks/ + scripts/ 顶层 → 9 个无 argparse
  但 `__main__` 直接执行：analyze_irregular_line / check_onnx_webgpu_feasibility /
  compare_baselines / memory_recall_validation / probe_m1_mmlu20 / pscan_probe2 /
  run_benchmark / streaming_edge_profile / text_selective_ab。text_selective_ab.py
  已加参数化入口（--sel/--seed，PR #56，B-1 并行执行所需），仍为 sys.argv 手解析
  （--help 仍会开跑）；全量 argparse 改造降级低优先 mechanism 项——L006"复用前读
  源码确认入口"纪律为强制项，不依赖此改造。

## DONE（本池启用后完成）

### B-6 · PMB 参照系统重跑入库（首航发现的 L001 欠账清偿）· DONE（2026-09-09 二航）
- **起因**：ITERATION_PRINCIPLES §3 E2 行声称"rag/hash 难格 0.78 已出数"，
  但 benchmarks/results/ 与仓内均无可定位的落盘文件（L001：二手数字不可引用）。
- **执行（2026-09-09 二航）**：重跑 none/rag/oracle × seeds{0,1,2} × task all，
  9 个 JSON 入 `benchmarks/results/pmb/`，scripts/validate_results.py PASS。
- **首手参照数（pmb-v0 默认网格）**：none 全 0（sanity 地板）；rag（hash encoder）
  t1 难格 (64,16) **0.839**、t2 recall 1.0/stale 0.0、t3 曲线 [1.0, 0.875, 1.0, 1.0]
  （0.875 为 16 题抽样噪声，检索式本不应遗忘）；oracle 窗内 1.0、**超窗塌陷
  1.0→0.0**（设计性数据点：固定窗口硬墙 = M1 差异化主轴的实证）。
- **对 0.78 声明的对账**：重测 0.839 ≠ 0.78，原 0.78 出处不可考（疑为不同 seed 集
  或网格配置），**以本次入库 9 文件为权威参照**，HANDOFF 的 0.78 表述待用户裁定
  是否订正（晨报待判定项）。
- **判定**：DONE（区分度判负标准未触发，参照数一手入库）。

## DEAD（判负归档，带证据）

### B-7 · Falcon 系更新规则作 PMB 参赛系统 · DEAD（2026-09-09 白班 s0 screening + 夜班 3-seed 判负落地）
- **预注册判负触发**：Falcon-1 NLMS（d=256）PMB T1 难格 (64,16) recall **0.0**
  << rag(hash) 0.839 − 0.10（门槛 0.739）。按预注册任一格点不满足即 DEAD，
  T2 不再执行。
- **首手证据**（`benchmarks/results/pmb_falcon/falcon_nlms_t1_s0.json`）：
  T1 全网格 recall 0.0–0.5；难格 (64,16)=0.0，n=64 任意 k ≤0.02（回归式 F 在
  64 facts 时读出折中，不指向任一 value）。
- **写规则对照**（`sum_t1_s0.json`）：sum 规则 n=4/16 全 1.0，难格 0.406
  ——叠加式外积优于回归式（难格仍 < rag 0.839）。
- **夜班 3-seed 补齐（正式判定落地）**：s1/s2 补跑 + sum 对照 s1/s2（各 0.5s CPU，
  遵守让路条款）→ falcon_nlms 难格 **[0.0, 0.0, 0.0]，均值 0.000 ≤ 0.739**，
  满足 ≥3 seeds 判定纪律；sum 3-seed 均值 0.385（作参数化内部对照有效）。
- **结论**（预注册的"输"结论，可引用）：更新规则强（Falcon：LM 外推）
  ≠ 跨会话持久化语义强。NLMS 回归式记忆适配序列内 read-after-write，不适配
  跨会话精确事实绑定；sum/delta 叠加式是跨会话保持的候选。
- **实现落库**：parametric_memory.py 加 falcon_nlms 规则；PMB 加
  ParametricMemorySystem + `--update-rule`（可测 sum/delta/falcon_nlms）。

---

## 统计（每班更新）

- 初始化（2026-09-09 首航）：PREREGISTERED 2 / CANDIDATE 3 / DONE 0 / DEAD 0
- 首航外部扫描后：PREREGISTERED 3（B-4 转正，带 arXiv:2510.09551 扫描证据）
  / CANDIDATE 2 / DONE 0 / DEAD 0
- 二航（2026-09-09 16:28）：B-4 范围修正（D5 重复，PMB 已有 T1-T3）、B-6 完成
  （PMB 参照 9 JSON 入库）、B-3 扫描后挂起（等 GPU adapter）→
  **PREREGISTERED 2 / CANDIDATE 2 / DONE 1 / DEAD 0**
- 三航（2026-09-09 17:1x，价值对齐后首轮）：学术扫描命中 Falcon 系
  （arXiv:2608.27763，4 周前）→ **B-7 登记 PREREGISTERED**（Falcon-1 作 PMB
  参赛系统，首次跨会话持久化基准，学术价值档 1，CPU 可跑）→
  **PREREGISTERED 3 / CANDIDATE 2 / DONE 1 / DEAD 0**
- 白班（2026-09-09）：**B-7 执行判负 DEAD**（falcon_nlms PMB T1 难格 recall 0.0
  << rag 0.739 门槛；sum 对照难格 0.406）→
- 夜班（2026-09-09 23:00）：B-7 3-seed 补齐（[0,0,0] 均值 0.000 ≤ 0.739）→
  正式 DEAD 落地；让路条款生效（系统 506%>50%），新训练降级为扫描/分析 →
  **PREREGISTERED 2 / CANDIDATE 2 / DONE 1 / DEAD 1**
- 白班（2026-09-10）：**B-1 32K 判别点在服务器 A100 GPU 执行判负 DEAD**
  （selective 2/3 配对胜但 sign-test p=0.5 不显著；8K→32K 增益未放大）→
  **PREREGISTERED 1 / CANDIDATE 2 / DONE 1 / DEAD 2**

---

## 计算发现（非实验条目）

### 2b 拆分训练可行性 · 2026-09-10 服务器 A100
- **结论**：2b（2.06B，M2 preset 2080×34×16/4）可在 **40GB A100 + 现有 40GB 磁盘**
  训练，无需扩容——推翻初判"需 80GB 单卡 + 100GB 磁盘"。
- **手段**：8bit 优化器（优化器状态 33GB→~4GB）+ bf16 参数检查点（9.1GB→3.9GB）
  + **lr 1e-4**（3e-4 会发散到 loss 14.7）。
- **证据**：1000 步 val PPL 7.00；2000 步 val PPL 5.99（服务器 `/root/t2b_long.log`）。
- **注意**：bf16 检查点不可精确 resume（optimizer 状态未存）；要 fp32 可 resume
  则需扩盘（~100GB）。
- **来源**：DEEP_INTEGRATION_PLAN 1b 的 2b 训练准备；本会话服务器执行。

### B-8 · PMB 扩 t4/t5 任务族（Multi-Session Reasoning + Abstention）· PREREGISTERED（学术价值档 1）
- **依据（二航扫描）**：LongMemEval 五能力映射中，IE≈T1、KU≈T2 已覆盖，
  **MSR（跨会话聚合/比较推理）与 Abstention（对未知信息拒答）是 PMB 缺口**；
  补齐后 PMB 覆盖 LongMemEval 4/5 能力 + 独有遗忘曲线 T3——"Zoology 基准"
  的能力矩阵才完整（附录 B START ④：做 persistent memory 领域的 Zoology）。
- **任务设计草案**：
  - t4 跨会话聚合：K 个会话各写入 1 个同属值（如同一实体的不同属性计数/求和），
    问题要求跨会话聚合（"总共出现过几个不同的值"）；变量 K ∈ {2, 8, 16}；
  - t5 拒答：问题引用**从未写入**的键，正确行为 = 不返回任何 gold 串；
    指标 = 拒答准确率 + 幻答率（返回了不存在的值）。
- **预注册判负标准**：防枚举截断（4000 chars）下，若 t4/t5 无区分度
  （none 系统 recall > 0 或 rag 与 oracle 差距 < 0.3）→ 任务设计无效，
  回炉重设计（不算方向 DEAD，是任务族工程返工）。
- **最便宜实验**：tasks.py 加 t4/t5 生成器 + systems.py 兼容验证（none/rag/hash
  × 3 seeds，CPU 全网格 ~分钟级）；fastweight 臂待 GPU（B-4 判负实验同批）。
- **来源**：LongMemEval 五能力映射（二航）；附录 B 主轴 (b)；价值档 1。

### B-8 结案记录 · PMB 扩 t4/t5 任务族 · DONE（2026-09-11 上午，含回炉注记；预注册条目见上方 B-8）
- **实现**：tasks.py 加 `generate_t4`（跨会话 JOIN：rating 会话 → 干扰 → ID 会话，
  问题给 rating 问 ID，答案必须跨会话拼接）+ `generate_t5`（8 已知题 + 4 拒答题
  混合，拒答失败 = scored 前缀出现任意 6 位码）；run_pmb.py 加 abstain 评分分支、
  run_t4/run_t5 驱动器、TASK_RUNNERS 注册（t1-t5 全集）。
- **参照数（3 systems × 3 seeds，18 JSON 入 benchmarks/results/pmb_t45/）**：
  - t4 难格判读：none 全 0（地板✓）；rag (4,0)=1.0 但 **(4,4)=0.0、(8,4)=0.0**；
    oracle (k=0)=1.0 → **(k=4)=0.0**——JOIN 任务上检索式与固定窗口双双结构性失败
    （rating 锚点在 A 会话、答案在 B 会话，topk 只拉到一头；超窗则两头全丢）；
  - t5 拒答：**rag 幻答率 100%**（未知人名查询必拉回干扰 code）、oracle 100%
    （截断窗内必有码）、none 0%（空返回 artifact）——拒答不可能性对检索式的
    结构性暴露。
- **预注册判据裁定**：t4 的 none=0 满足；但 "rag-oracle 差距 <0.3" 字面条件在
  (4,0) 格触发（两者同为 1.0）——**按预注册记"部分回炉条件触发"**，解读：
  区分度存在于 (M,k) 维度而非系统维度，任务设计本身有效（k=4 的塌陷正是要
  度量的现象），**回炉裁定（2026-09-12 自主，用户授权免决策）**：字面触发但
  实质为窗口语义（k=4 时 rag/oracle 同塌于超窗，非任务失败）→ 判据修正为
  "k≥4 且窗口内（k=0）rag 正常时 rag-oracle 差距<0.3 → 回炉"，本轮 (4,0)
  rag=1.0 正常 → 判据未触发，任务族存活不回炉。
- **学术价值（可引用）**：JOIN 与拒答是检索式记忆的两个结构性失败模式，
  fastweight 的 O(1) 状态在这两格上有天然的假设检验位置（待 GPU adapter）。
- **判定**：DONE（任务族实现 + 18 证据 JSON 入库；回炉注记待裁定）。

## 统计追加（2026-09-11 上午轮）
- B-8 DONE → **PREREGISTERED 2（B-2/B-4）/ CANDIDATE 1（B-3）/ DONE 2 / DEAD 2**

### B-9 · PMB 编码器敏感度对照（rag hash vs e5）+ t4/t5 任务矩阵 · DONE（2026-09-11 3-seed 判定）
- **判定（对照预注册三标准，3-seed 实测）**：
  - (a) t1 难格 (64,16)：e5 = **1.0**（hash 0.839）→ ✅ 达标，且**语义编码器连精确
    码检索都更好**（问句措辞与事实句的语义匹配优于词袋）；
  - (b) t4 JOIN：e5 全格 ≥ 0.5 ✅，难格 (8,4) **0.5 vs hash 0.0** → ✅ 达标——
    **hash 检索的 JOIN 全灭被证实为编码器缺陷**（rating 锚点问句的语义检索能同时
    拉到 IDs 会话），B-8 判读据此修正：JOIN 失败 = hash 检索盲区为主 + 聚合残余待验；
  - (c) t5 幻答率：e5 = **1.0** → ❌ 未达标（hash 同为 1.0）——**拒答失败与编码器
    无关，是 evidence 模式的结构性极限**：只要上下文被干扰会话填满，任何检索式
    系统都会返回码。拒答需要决策层（generative 模式的"未知"输出或检索置信阈值），
    登记为 t5 的后续方向（t5b：abstention 的 generative/阈值协议）。
- **证据路径**：benchmarks/results/pmb_e5/pmb_rag_e5_s{0,1,2}.json（对照
  benchmarks/results/pmb_t45/pmb_t4_rag_s*.json / pmb_t5_rag_s*.json）。
- **学术结论（可引用）**：*"Encoder quality dominates cross-session JOIN
  retrieval (0.0→0.5-0.75 from hash to e5) while abstention failure remains
  encoder-invariant (100% false-answer for both) — evidence-mode memory
  benchmarks need an explicit abstention protocol."*
- **来源**：B-8 判读后续；价值档：学术 1 + 工程 1。
- **假设**：检索式系统在 PMB 上的判读随编码器质量变化——e5（语义向量）相对
  hash（词袋精确匹配）在语义变体问句上应有优势，但在精确 6 位码检索上优势
  有限；t4 JOIN 任务的失败（rag 难格 0.0）若源于检索而非聚合，e5 应能部分
  恢复（仍 ≤ oracle 窗内上界）。
- **预注册判读标准**：对照三档（hash 基线 26/27 行已入库）——
  (a) t1 难格 (64,16)：e5 ≥ 0.839±0.1 视为"编码器不伤精确检索"；
  (b) t4 (4,0)：e5 ≥ 0.5（hash=1.0）且 t4 难格 e5 > hash=0.0 → "语义编码器
  改善 JOIN"候选成立（升格论文素材）；
  (c) t5 幻答率：e5 < 1.0 → "语义检索具备部分拒答能力"成立。
- **执行**：run_pmb.py --task all --system rag --encoder e5 × seeds{0,1,2}
  （需下载 multilingual-e5-small ~120MB，首次运行含下载时间，预计 20-40min，
  超 1h 政策边界——分批跑，先单 seed 确认可行再铺满 3 seeds）。
- **来源**：B-8 判读的后续（检索器维度）；价值档：学术 1 + 工程 1。
- B-9 判定（2026-09-11 05:3x-09:4x）：e5 3-seed 对照完成，(a)(b) 达标 (c) 未达标
  （编码器不变性发现）→ **B-9 DONE**，t5b（拒答协议）浮现 →
  **PREREGISTERED 2（B-2/B-4）/ CANDIDATE 2（B-3/t5b）/ DONE 3 / DEAD 2**

### B-10 · PMB vs 2026 新基准对照与差异化定位 · DONE（2026-09-11，docs/PMB_VS_2026_BENCHMARKS.md）
- **判定：差异化成立（判负未触发）**——遗忘曲线全曲线与 bit 级持久化机械强制
  两项核心差异化在 4 个对照对象中均未被覆盖。
- **最关键发现**：StateMemBench（arXiv:2608.19652）与 PMB T2 同题且闭池三态
  评分更细 → 定位为"引用并对照"（不重复造）；AgentMemBench 是检索质量线
  （LoCoMo/MultiDoc2Dial），与 PMB 的程序生成+精确评分正交。
- **行动项**：README 加 Related work 引用 StateMemBench；开源节奏提前至 2-4 周；
  详见 docs/PMB_VS_2026_BENCHMARKS.md（对照矩阵 + 4 项行动排序）。
- **增补（2026-09-16 05:5x，扫描轮 3 文档化）**：对照总览表加 Memora
  （2604.20006，三差异化存活确认）/ FwPKM（2601.00671，归系统侧）两行 +
  新增"扫描轮 3 增补"节（行业定位强化 + PMB 侧同日新证据索引），详见文档
  第六节。
- **扫描证据（2026-09-11）**：Zoology 生态位正被快速占领——
  ① AgentMemBench（arXiv:2608.00009，6 月）：5 种记忆管理策略统一可复现基准；
  ② MemoryArena（Stanford）：多会话 Memory-Agent-Environment 循环评测 gym；
  ③ "Can Agent Memory Systems Track Evolving State?"（arXiv:2608.19652）：
     演化状态跟踪——与 PMB T2 同题；
  ④ mem0《State of AI Agent Memory 2026》：LoCoMo/LongMemEval/BEAM + 21 框架
     集成的行业报告。
- **PMB 的存活差异化**（须尽快写成定位文档）：
  ① 遗忘曲线全曲线（T3）——新基准均未覆盖；
  ② **bit 级持久化语义的机械强制**（snapshot/restore 逐会话落盘，进程内存=0 分）——
     独有；
  ③ 跨会话 JOIN（t4）与拒答（t5）——AgentMemBench 未报。
- **最便宜实验（CPU）**：拉 AgentMemBench/2608.19652 的任务定义，映射到 PMB
  T1-T5 矩阵，产出《对照与差异》文档（学术引用素材 + 潜在集成路线）。
- **判负/回炉标准**：若对方已覆盖遗忘曲线全曲线 + bit 级持久化 → 差异化不存在，
  B-4 的"Zoology 基准"叙事降级为"工程参考实现"。
- **来源**：本次扫描；主轴 (b)；价值档 1。

### B-11 · memory-clarification boundary 映射 PMB（T2+T5 交叉）· CANDIDATE（学术价值档 2）
- **扫描证据**：arXiv:2608.19564 "Remember, Verify, or Ask?"——持久记忆的错误
  更新会静默扭曲后续行为，研究"何时澄清"的边界。与 PMB 的 T2（stale 语义）
  + T5（拒答）天然交叉：PMB 可加 "verify-then-commit" 任务变体。
- **预注册判据**：待读原文后写死。
- **来源**：本次扫描；价值档 2。

### B-12 · MoNe 分段神经记忆作为 PMB 参赛系统 · CANDIDATE（学术价值档 1，挂 GPU）
- **扫描证据**：arXiv:2608.17616（3 周前）——MoNe：分段 test-time 学习的
  fast-weight 神经记忆，层局部梯度更新，推理时分段读。
- **与 B-7 的区别**：MoNe 是分段化神经记忆（非单条更新规则），可能规避
  Falcon-1 的跨会话失败模式——但"分段 ≠ 跨会话持久化"的假设待检验。
- **执行依赖**：公开代码 + GPU（同 B-4 fastweight 臂批次）。
- **来源**：本次扫描；主轴 (a) 竞品追踪；价值档 1。

### t5b v0 · 检索分数阈值拒答 · DONE（2026-09-11 上午，机制轨）
- **实现**：RAGSystem 加 opt-in `score_threshold`（默认 None=行为不变），
  run_pmb 加 `--abstain-threshold` 透传；make_system 工厂接入。
- **校准+测试**：s0 扫 4 阈值 → 0.5 为完美工作点（幻答 1.0→0.0，
  recall_known 1.0 不动）；s1/s2 盲测 **幻答均值 0.083（基线 1.0）**，
  recall_known 1.0 全保。
- **判定**：DONE——幻答率 1.0→0.083（-92%），零误拒。拒答失败的
  "结构性极限"实为**缺一个 8.8 秒能写完的决策层**；e5 与阈值正交
  （可叠加，待验证）。
- **学术结论（可引用）**：*"A single retrieval-confidence threshold
  eliminates 92% of hallucinated answers on unknown-key questions at
  zero cost to known-question recall — abstention failure in
  evidence-mode memory benchmarks is a protocol gap, not a
  capability limit."*
- **遗留**：阈值 0.5 是 hash 编码器的标定值，e5 需另行标定；正式
  paper 表格需将校准/测试分离写明（s0 校准 → s1/s2 测试）。

### B-4' · 微型 fastweight（梯度写）PMB 判负实验 · DEAD 落地（2026-09-11 上午）
- **实现**：MicroFastWeightSystem——线性联结记忆（d=104）+ 固定随机投影 +
  每事实 8 步 SGD 梯度写（经典 fast-weight 规则的灾难性干扰文献原型）；
  generative 模式，snapshot 位级字节化（W+码表+值矩阵 ~400KB）。
- **3-seed 全网格结果（vs 参照）**：
  - t1 难格 (64,16)：**0.0**（连 (4,0) 也只有 0.167——8 步 SGD × 64 事实的
    灾难性干扰把早期绑定完全覆写）；
  - t4 JOIN 难格 (8,4)：**0.0**；
  - t5：known recall 0.083、幻答率 1.0。
- **判定（预注册标准）**：t4 难格 0.0 ≤ 0.5 → **DEAD**——“梯度写在跨会话
  事实绑定上同样失败”，与 B-7（闭式规则失败）合并为完整结论：
  **无论闭式（sum/delta/falcon_nlms）还是梯度（SGD），fast-weight 写在
  跨会话持久化基准上全灭；检索式（rag hash 0.839）保持唯一有效路径，
  e5 语义编码器 1.0 封顶。**
- **证据路径**：benchmarks/results/pmb_t45/pmb_{t1,t4,t5}_microfw_s{0,1,2}.json（9 JSON）。
- **学术结论（可引用）**：*"Gradient-trained fast weights exhibit the
  catastrophic interference their literature predicts: 64-fact retention
  drops to 0.0, and cross-session JOIN to 0.0 — closing the 'maybe gradient
  writing helps where closed-form fails' hypothesis."*

## ⚠️ 关键勘验（2026-09-11 12:1x）：_TOKEN_RE 数字 token 缺失 → 全部 generative 判负为编码器伪影

- **发现**：`HashedBoW._TOKEN_RE = [a-z']+` 只匹配字母——纯数字 gold（6 位码）
  编码为**零向量**。generative 模式（parametric/falcon/micro_fw/micro_rls）的
  value 表示全零 → 之前所有"判负"测的是"hash 编码器无法表示数字"这一评测器
  伪影，**不是写规则的能力**。rag/oracle 不受影响（返回原句文本，harness 层
  substring 匹配）。
- **修复**：正则加 `[0-9]`（1 字符改动），全部 generative 系统重跑。
- **修复后 3-seed 真实判读（t1 难格 64,16 / t4 JOIN 8,4）**：
  - micro_rls(RLS)：0.005 / **0.0**
  - micro_fw(SGD)：0.016 / **0.0**
  - param(sum)：**0.385** / **0.0**
  - param(falcon)：0.0 / —
  - rag(hash) 对照：0.839 / 0.0
- **结论升级（三条，均可引用）**：
  1. **sum 写规则是跨会话事实绑定的最佳参数化方案**（难格 0.385，远超 RLS/SGD
     的 0.005-0.016——叠加式外积的干扰抵抗力被定量证实）；
  2. **但所有参数化写规则仍不及检索式**（sum 0.385 << rag 0.839）——
     "parametric < retrieval"的排序在数字 token 修复后**保持成立**；
  3. **t4 JOIN 对所有系统（含 e5 检索 0.625）仍是未解格**——跨会话拼接是
     真缺口而非伪影，B-4' 的定位文档结论不变。
- **教训（L 候选）**：评测器必须先自证能表示被测目标（L001"评测器先自证"的
  评测器版）；一个字符类 `[a-z']` vs `[a-z0-9']` 的差异曾把三个方向的判负
  全部变成伪影。

### B-13 · t2 任务族补全 + delta 主场假设检验 · DONE（2026-09-11）
- **执行**：t2（流式更新）全系统补跑——这是全矩阵唯一没跑过任何系统的任务族。
- **结果（3-seed）**：
  - **param_sum 1.0 / param_delta 1.0（stale 0.0）**——叠加与减法写在流式更新上
    均满分（闭式写对"覆写旧值"语义天然适配）；
  - param_falcon_nlms **0.0**（回归式三连败确认：t1 难格/t4 JOIN/t2 全灭）；
  - rag(hash) **0.917**、stale 0.042——检索式在 T2 上接近满分但**有 stale 残留**
    （hash 词袋对"最新值"的时序语义不敏感，14/15 命中）；
  - oracle 0.0/stale 1.0——固定窗口在更新流上结构性失败（设计性数据点）。
- **科学结论（可引用，改写 B-13 判读）**：*"Write-rule × task-semantics
  matching, not a single best rule: additive writes dominate streaming-update
  tasks (1.0 vs rag 0.917 with stale residue), while delta writes lead
  hard-cell retention (0.24) — and regression rules fail everywhere. The
  oracle's structural collapse on update streams confirms fixed-window
  memory cannot track state."*
- **矩阵状态**：5 任务族 × 5 系统（param×3/rag/oracle/none）全网格补齐。

### B-13 补全（2026-09-11 夜班）：param(sum) t1 网格补跑 — 参数化最优易主
- **修复后 param(sum) t1 缺口补跑（3-seed）**：(4,0)=**1.0**、(64,16)难格=**0.385**
  → 超越 param(delta) 的 0.24，**参数化最优易主回 sum**。
- **最终写规则排序（t1 难格）**：sum 0.385 >> delta 0.24 > falcon_nlms 0.0 ≈
  micro_rls 0.005 ≈ micro_fw 0.016 —— 叠加式外积（sum）在 64 事实规模下
  干扰抵抗力最强；delta 的减法覆写在难格上反而弱于 sum（与 t2 满分对照：
  delta 的覆写优势只在显式 update 语义任务上成立）。
- **矩阵最终态**：5 任务族 × 6 系统全网格（judge_t45.py 常驻可复算）。
  最优组合：T1/T2/T5=param(sum)（1.0/0.385）、t4 JOIN=无系统可解（rag hash
  0.0、e5 0.625 部分恢复）、T1 难格=rag(e5) 1.0 封顶。

### B-13' · T2 全系统矩阵收尾（micro×6 复算入库 + none×3 新格）· DONE（2026-09-16 02:1x，预注册→执行→验收同轮闭环）
- **起因（欠账清偿，L001 纪律）**：B-13 声称"5 任务族 × 6 系统全网格"，但 T2 格
  实际缺 3 系统：micro_fw / micro_rls（9-12 凌晨有未入库产物，无一手 commit）与
  none（从未跑过）。**未入库的数字不可引用**——本轮复算替换为一手证据并补齐 none。
  另：pmb_e5/pmb_t2_rag_e5_s{0,1,2}.json 为未入库冗余副本（已核对与 B-9 入库的
  task-all 文件 t2 格逐值一致：recall 1.0/stale 0.0 ×3 seeds）→ 验收后删除。
- **判负/回炉标准（预注册）**：
  1. **确定性门**：同 seed 复算的 recall / stale_answer_rate / state_bytes 必须与
     9-12 产物逐值一致（harness 全确定性，无容差）；不一致 → 复现性 bug，回炉不判定；
  2. **none 地板门**：none t2 recall = 0.0（3/3 seeds）；>0 → 防枚举地板破裂，
     harness bug，回炉；
  3. **结构门**：9 JSON 全字段完整（benchmark/system/seed/config/results.t2 九字段）。
- **最便宜实验**：9 个单任务 CPU 配置（--task t2，秒级/个），不触 GPU、不触 e5 下载。
- **预注册预期（可判负的实质假设）**：覆写式写（SGD）在流式更新上部分有效
  （9-12 产物示 0.75-0.875：灾难性干扰的 recency 偏置恰为"新值胜出"语义所用），
  回归式（RLS）与闭式回归（falcon_nlms）仍失败（0.0）；none=0。
  若复算推翻（如 SGD 归零）→ "写规则×任务语义匹配"叙事需改写为
  "仅叠加式适配流式更新"，仍可结案但结论降级。
- **来源**：B-13 全矩阵声明的收尾；价值档：学术 1（可引用矩阵完整性）+ 工程 1（欠账清偿）。
- **执行（2026-09-16 02:1x）**：9 单任务配置 8.9s CPU
  （`PYTHONPATH=. .venv/bin/python benchmarks/persistent_memory/run_pmb.py --task t2
  --system {micro_fw,micro_rls,none} --seed {0,1,2}`）。
- **验收（预注册三门全过）**：确定性 6/6（recall/stale/state_bytes 与 9-12 产物
  **逐字节一致**，含 s2 stale=0.125 的孤例）；none 地板 3/3 = 0.0；结构 9/9 全字段。
- **T2 矩阵终态（9 系统 × 3 seeds，3-seed 均值 recall/stale_ans）**：
  param_sum **1.000**/0.000 · param_delta **1.000**/0.000 · rag(e5) **1.000**/0.000 ·
  rag(hash) 0.917/0.042 · **micro_fw 0.792**/0.000 · param(falcon) 0.000 ·
  micro_rls 0.000 · oracle 0.000/**1.000**（设计性塌陷）· none 0.000。
- **科学结论（可引用）**：*"On streaming-update retention, additive outer-product
  writes are perfect (1.0, zero stale), SGD overwrite writes reach 0.79 — their
  catastrophic-interference recency bias is exactly the inductive bias that
  update semantics reward — while regression writes (RLS, NLMS) collapse to 0.0.
  Retrieval with a semantic encoder matches additive writes (1.0) without stale
  residue; a lexical encoder retains 8% stale answers."*——写规则×任务语义匹配
  论点补全中间谱系：预注册预期（SGD 覆写部分有效 0.75-0.875）被复算精确证实，
  无叙事降级。
- **冗余副本处置**：pmb_e5/pmb_t2_rag_e5_s{0,1,2}.json 已删（与 B-9 入库
  task-all 文件 t2 格逐值一致，三重核对后清理）。
- **判定**：**DONE**——T2 成为 PMB 首个 9 系统全覆盖任务族，全部一手入库。

### B-14 · T3 遗忘曲线 × 5 参数化/微型系统 + falcon t4 补格（矩阵真收尾）· DONE（2026-09-16 02:4x，含"叠加写无遗忘"证伪升格）
- **缺口**：T3 是 PMB 头号差异化主张（遗忘曲线全曲线），但只测过 none/rag(hash)/
  oracle（B-6）+ rag(e5)（task-all 文件）4 系统——param_sum/delta/falcon_nlms/
  micro_fw/micro_rls **从未跑过 T3**。另 param_falcon 缺 t4 格（3 配置）。
- **预注册判负/预期**：none 地板=0，oracle 窗内 1.0/超窗塌陷。实质假设：
  叠加式写（sum/delta）无衰减项 → T3 曲线应平坦（干扰不随 gap 增长）；
  micro_fw（SGD 覆写）曲线受 recency 主导；falcon/RLS 全零。
  若 sum 曲线呈非平坦衰减 → "叠加写无遗忘"叙事证伪（新发现，升格）；
  若全部参数化系统 T3 平坦 → "T3 对参数化系统无区分度"边界注记（任务族
  边界，非 DEAD——检索式/窗口式已有区分度）。
- **最便宜实验**：18 个单任务 CPU 配置（5 系统 × T3 × 3 seeds + falcon t4 × 3），秒级/个。
- **来源**：本轮任务×系统覆盖盘点；价值档：学术 1（差异化主张的证据完整性）。
- **执行（2026-09-16 02:4x）**：18 配置 92s CPU（t3 × {param_sum/delta/falcon_nlms,
  micro_fw, micro_rls} × 3 seeds + falcon t4 × 3）。
- **验收**：结构 18/18 全字段；3-seed 一致性 sd ≤ 0.078；参照一致（rag(hash)
  平坦 [1.0, 0.875, 1.0, 1.0] 与 B-6 入库吻合）。
- **T3 曲线终态（3-seed 均值 recall，gaps=[0,2,8,32]，gap g = 16+8g 条累计事实）**：
  - rag(hash) 1.0/0.875/1.0/1.0 · rag(e5) 全 1.0（平坦免疫，参照）；
  - **param_sum 1.0/1.0/1.0/0.021 · param_delta 1.0/1.0/1.0/0.0**——平坦直至
    gap=32（272 条 ≈ d=256）**容量断崖**；
  - **micro_fw 0.458/0.354/0.042/0.0**——单调级进遗忘（唯一心理曲线形态）；
  - falcon_nlms 0.062→0 · micro_rls ≈0（地板）；falcon t4 全格 0.0（五任务族全败闭环）。
- **判定（预注册"证伪→升格"分支触发）**：**DONE**——"叠加写无衰减"叙事被证伪：
  真相是**阶跃式容量断崖**而非平坦。升级发现（可引用）：
  *"On the forgetting-curve task, system classes separate into distinct
  signatures: retrieval is immune (flat 1.0), additive outer-product writes
  hold a step-function capacity cliff (perfect at 80 stored facts, ≈0 at 272 ≈
  state dim 256), and gradient overwrite shows graded recency decay
  (0.46→0.35→0.04→0.0) — the first parametric forgetting curves on PMB."*
- **T3 差异化主张强化**：全曲线任务族现在对全部三类系统（检索/叠加写/梯度写）
  给出可分离的曲线形态——"Zoology 基准"核心证据链补全。
- **对 B-17 的锐化**：断崖位置 ≈ 状态维度 d 的预测已生效——d-sweep
  {256,1024,4096} 应把断崖外推 ~4×/16×；B-17 判负标准不变，预期更尖锐。

### B-15 · Kalman Delta 写规则作为 PMB 参赛系统（不确定性感知写）· DEAD（2026-09-16 04:0x，首测判负 + 机制诊断）
- **扫描证据（扫描轮 3）**：arXiv:2609.07816（Kalman Delta Networks，2026-09 新出）
  ——把 delta rule 重构为线性-高斯 SSM 的 innovation 更新：每笔写按 Kalman 增益
  加权（累积证据 × 观测可靠性），保留协方差不确定性递推；delta rule 是其
  "各向同性代理协方差"特例。对角/各向同性变体为**闭式 Möbius 递推**（numpy
  可实现，无 GPU 依赖）。原文只在 750M/1.3B 序列内 LM 预训练上评测（PPL/
  下游均值）——**跨会话持久化从未被测**（与 B-7 Falcon-1 同型缺口）。
- **与现有结果的接口**：写规则谱系已入库（sum 难格 0.385 / delta t2 满分）；
  KDN 的"不确定性随无写入增长"是 T3 遗忘曲线的**首个机制性预测**——
  不确定性门控写 × T3 全曲线天然交叉（B-14 先行补齐 sum/delta 的 T3 基线）。
- **实现**：parametric_memory.py 加 `kalman_delta` 规则（先 isotropic 后 diagonal），
  PMB `--update-rule` 接入；t1 全网格 + t2 + T3 × 3 seeds（CPU，预计分钟级）。
- **预注册判负标准**：kalman_delta t1 难格 (64,16) 3-seed 均值 ≤ sum 0.385+0.05
  **且** T3 曲线不比 sum 平坦 → "不确定性门控对跨会话保持无增益" DEAD；
  任一优于（难格 > 0.435 或 T3 显著更平）→ DONE 升格论文素材
  （"KDN 跨会话首测" claim）。
- **来源**：arXiv:2609.07816；对照 B-7/B-13' 谱系；价值档 1。
- **实现（已入库）**：`mt_lnn/parametric_memory.py` 加 `kalman_delta`（isotropic
  KDN：增益 = u/(u·||k||²+obs_noise) 随累积证据收缩，会话边界按 process_noise
  再膨胀；超参 v0 默认 u0=1/r=1e-2/q=1e-3 未调；u 仅 kalman 会话入快照，
  其余规则快照字节不变）+ systems.py 会话钩子 + run_pmb CLI；test_pmb 9/9 过。
- **执行**：t1/t2/t3 × 3 seeds @ d=256，59s CPU。
- **结果（3-seed 均值）**：t1 难格 **0.000**（判负线 0.435；全格 (4,0)=0.5
  其余 ~0）· t2 **0.042**/stale 0.083（对照 sum/delta 1.0）· T3 [0, 0.021, 0, 0]
  （不比 sum 的 [1,1,1,0.021] 平）→ **预注册双条件同时触发 DEAD**。
- **机制诊断（可引用，本轮真产出）**：各向同性标量协方差把"对同一固定函数的
  证据累积"套在开放世界绑定流上——增益按调和级数 ~1/n 塌缩，第 ~10 条事实后
  写通道实质关闭，晚期绑定不可读出。*"Isotropic evidence-weighted gating is
  an anti-pattern for open-world binding memory: certainty accumulates across
  distinct bindings and shuts the write channel (hard-cell recall 0.0 vs
  fixed-eta NLMS 0.0 and additive sum 0.385 at equal d=256). Uncertainty must
  be localized per binding, not global."*——KDN 原文的 LM 预训练场景（函数固定）
  与跨会话记忆场景（函数持续扩张）的适配边界由此标定。
- **止损注记（§4.6）**：per-binding/对角不确定性或新键重置属同方向变体，
  不重开；本条目 DEAD 即最终态，实现保留作阴性对照。
- **判定**：**DEAD**（首测 claim 落地："KDN 式证据加权在跨会话持久化上首测
  判负，且给出机制解释"）。

### B-17 · sum 写规则容量 scaling 探针（d-sweep）· DONE（2026-09-16 02:5x，判定大幅超阈值）
- **依据**：arXiv:2603.26554（谱优化器联想记忆容量 scaling，2026）+
  Cabannes et al. 2310.02984（外积联想记忆标度律经典）——容量随状态维度 d
  的标度有理论。sum 规则 t1 难格 0.385 若是**容量瓶颈**，d ∈ {256,1024,4096}
  应单调上移；若不动 → 0.385 是干扰/读出瓶颈而非容量。
- **预注册判负标准**：d=4096 难格 recall 相对 d=256 上移 < 0.05 →
  "容量瓶颈"假设 DEAD（结论改判干扰瓶颈）；上移 ≥0.05 → DONE
  （sum 0.385 的第一性解释 + state_bytes 代价同表诚实报告）。
- **最便宜实验**：parametric d-sweep × t1 难格 × 3 seeds，秒级。
- **来源**：扫描轮 3；价值档 1。
- **执行**：新驱动 `benchmarks/persistent_memory/dsweep_t1_hard.py`（仅难格
  (64,16)，episode/评分路径与 run_pmb 全网格完全一致；d=256 输出与已入库
  0.385 逐值吻合——确定性门 PASS）；d ∈ {256,1024,4096} × 3 seeds，198s CPU。
- **结果表（难格 recall / state_bytes / answer_latency）**：
  d=256 → **0.385** / 794KB / 1.4ms（参照吻合）
  d=1024 → **0.849** / 7.35MB / 5.0ms
  d=4096 → **0.995** / 96.5MB / 26ms
  对照：rag(hash) 同格 0.839 @ 2.27MB；rag(e5) 1.0。
- **判定（预注册阈值 lift ≥ +0.05）**：**DONE**——lift = **+0.609**，
  **容量瓶颈确认**。
- **叙事修正（矩阵级，可引用）**：*"The hard-cell deficit of additive
  outer-product writes is a state-capacity wall, not a mechanism failure:
  recall rises 0.385→0.849→0.995 across d=256→1024→4096 (96MB state, 26ms),
  reaching retrieval parity (e5 1.0) at ~42× the state cost of hash
  retrieval (2.27MB @ 0.839)."*——B-13 的"参数化 < 检索"排序须加容量限定词：
  等召回下检索的状态效率优势成立（~42×），但**写机制本身不是瓶颈**；
  B-14 的 T3 断崖（272 条 ≈ d=256 崩塌）由此获得第一性解释，跨任务族自洽。
- **浮现**：容量修复是否跨写规则普适？→ B-18（delta/falcon 对照臂，同驱动）。

### B-18 · 写规则 × 容量交互对照（delta/falcon d-sweep 控制臂）· DONE（2026-09-16 03:3x，分类学闭环）
- **问题**：B-17 确认 sum 的难格亏损是容量墙。容量修复是否跨写规则普适？
  - **delta**（减法覆写，d=256 难格 0.24）：d=4096 也逼近 1.0 → 容量定律跨
    规则推广（"容量是唯一的墙"）；仍低 → **叠加式 sum 与容量唯一正交**，
    写规则排序叙事再升级；
  - **falcon_nlms**（回归式，0.0）：预期对 d 不敏感（机制性失败）——阴性对照臂。
- **预注册判读标准**：delta d=4096 均值 ≥ d=256 + 0.05 → "容量定律跨写规则
  推广"；< 0.05 → "sum 独享容量增益"；falcon 任一 d > 0.1 → B-7 回归式
  判负结论需复核。
- **最便宜实验**：同一驱动 `--update-rule {delta,falcon_nlms}` × d {256,4096}
  × 3 seeds = 12 配置，预计 ~3-4 分钟 CPU。
- **来源**：B-17 浮现；价值档 1。
- **执行（2026-09-16 03:3x）**：12 配置 ~6 分钟 CPU；确定性门双臂 PASS
  （delta d=256 [0.25,0.1875,0.2812] 与 falcon [0,0,0] 均与已入库全网格逐值一致）。
- **结果（难格 3-seed 均值）**：delta **0.240 → 0.990**（d=256→4096，lift
  +0.750 ≥ +0.05）；falcon_nlms **0.000 → 0.005**（max 0.0156 << 0.1 复核阈值）。
- **判定**：**DONE**——**容量定律跨叠加族写规则推广**：sum 0.385→0.995 与
  delta 0.240→0.990 均被容量救活；回归式 falcon 对任何容量免疫（B-7 结论
  稳固，阴性对照通过）。
- **写规则分类学（B-7/B-13/B-14/B-17/B-18 合并结论，可引用）**：*"Write rules
  partition into capacity-limited families — additive (sum) and subtractive
  overwrite (delta) outer-product writes recover from ≈0.24-0.39 to ≈0.99
  hard-cell recall when state grows d=256→4096 — and capacity-invariant
  failures — regression-style writes (NLMS, RLS) stay at 0.0 at every
  capacity. Cross-session persistence cost is state capacity, not write
  mechanics; retrieval remains ~42× more state-efficient at parity."*
- **浮现 B-19（CANDIDATE）**：第三家族梯度写（micro_fw SGD）的容量臂待测。

### B-19 · 梯度写家族的容量臂（micro_fw/micro_rls d-sweep）· DONE（2026-09-16 04:4x，三族分类学确立）
- **问题**：2×2 分类学缺梯度写角——micro_fw（SGD 覆写，d=256 难格 0.016）与
  micro_rls（RLS 回归，0.005）在 d=4096 下是否被救活？预测：RLS 随回归式免疫
  （预测 0 附近）；SGD 覆写介于两者（覆写 recency 在保持任务上是干扰源，
  容量未必是它的墙）。
- **判读标准（预注册草案）**：micro_fw d=4096 > 0.05 → 梯度写也属容量受限族
  （分类学收敛为"线性闭式 vs 回归式"两族）；≤ 0.05 → 梯度写失败同为机制性。
- **依赖**：驱动加 `--system` 透传（micro_fw/micro_rls 接受 d_mem=dim）；12 配置 ~6 分钟。
- **来源**：B-18 浮现；价值档 1（分类学完备性）。
- **转正注记（2026-09-16 04:2x）**：判据即为上方预注册草案，先于执行 commit；
  确定性门 = micro_fw/micro_rls d=256 难格须与已入库 pmb_t1_{microfw,microrls}
  逐值一致。
- **执行（2026-09-16 04:4x）**：12 配置（2 系统 × d {256,4096} × 3 seeds），
  ~20 分钟 CPU（d=4096 micro_fw 单配置 ~3.4 分钟）。
- **确定性门（修正后 PASS）**：初次对检引用了修复前族 `pmb_t1_microfw_*`
  （8892f40，难格 [0,0,0]）→ FAIL；核对档案发现修复后权威族
  `pmb_t1_micro_fw_*`（eb01dfc，[0, 0.0156, 0.0312] ≈ 勘验段 0.016）与新跑
  d=256 **逐值一致** → 门 PASS。judge_t45.py 映射已同步改为修复后族（原映射
  指向废止档案，防后续误读）。
- **结果（难格 3-seed 均值）**：micro_fw **0.016 → 0.010**（d 256→4096，
  state 1.5MB→187MB）——平坦，**容量不救梯度写**（阈值 >0.05 未触发）；
  micro_rls 0.005 → 0.016（平坦，见审计注记）。
- **判定：DONE——三族分类学确立**（叠加/梯度/回归，可引用）：
  *"Write rules partition into three signatures on cross-session retention:
  additive outer-product writes are capacity-limited but capacity-scalable
  (sum 0.385→0.995, delta 0.240→0.990 at d=4096); gradient writes (SGD) fail
  mechanistically at every capacity (≈0.01 flat from 1.5MB to 187MB state);
  regression writes (NLMS, RLS) are likewise capacity-invariant. Only the
  additive family rides state capacity to retrieval parity."*
- **⚠️ 工程审计注记 → 已结案（2026-09-16 05:3x 审计）**：micro_rls 快照存完整
  pairs 列表（键值字符串对），restore 用种子确定性投影重解 W——state_bytes
  恒 ~14KB 是**按设计**的诚实体积（pairs 是字符串，O(n·len) 与 d 无关），
  非漏存。**B-19 的 RLS"容量结论不确定"就此解除并升级**：n=272 ≪ d=4096
  （精确求解域，零干扰理论上限）下 recall 仍 ~0.016 → RLS 失败与存储容量
  彻底无关，机制在读出侧（具体归因开放，未立案）。micro_fw 快照同理审计：
  W/codes/vals 入档，P 投影矩阵不入档但种子确定性再生成（功能等价，代码
  注释与实现一致）。
- **下一方向浮现**：B-16 读原文（arXiv:2601.00671）写死判负标准 → 两跳/
  product-key 基线实现；B-2 收割 ~11:04 CST 优先。

## 轮次追加（2026-09-16 03:3x）
- B-18 DONE → **PREREGISTERED 3（B-2 RUNNING / B-4 挂 GPU / B-15）/ CANDIDATE 4
  （B-12 / B-16 / B-19 / argparse）/ DONE 9 / DEAD 3 / 映射结案 1**
- **下一方向**：**B-15 kalman_delta 实现**（扫描轮 3 转正的主方向，分钟级 CPU）
  → B-19（驱动小扩展）；B-2 V14 收割窗口 ~11:04 CST 优先插入。

## 轮次追加（2026-09-16 04:0x）
- B-15 首测 **DEAD**（预注册双条件触发，机制诊断入库）→
  **PREREGISTERED 2（B-2 RUNNING / B-4 挂 GPU）/ CANDIDATE 4（B-12 / B-16 /
  B-19 / argparse）/ DONE 9 / DEAD 4 / 映射结案 1**
- **下一方向**：**B-19 梯度写容量臂**（驱动加 --system 透传，~6 分钟）——
  分类学最后一块；B-2 V14 收割窗口 ~11:04 CST 优先插入。

## 轮次追加（2026-09-16 04:4x）
- B-19 DONE（三族分类学：叠加=容量受限可救 / 梯度=机制性 / 回归=机制性；
  judge_t45 映射修正指向修复后档案族；RLS 快照审计小项登记）→
  **PREREGISTERED 2（B-2 RUNNING / B-4 挂 GPU）/ CANDIDATE 5（B-12 / B-16 /
  RLS 快照审计 / argparse / +B-19 遗留无）/ DONE 10 / DEAD 4 / 映射结案 1**
- **下一方向**：**B-16**（读 arXiv:2601.00671 写死判负标准 → 两跳检索基线）；
  B-2 V14 收割窗口 ~11:04 CST 优先插入。本地无更便宜可执行项时按新规收工。

## 轮次追加（2026-09-16 05:1x）
- B-16 DONE（两跳检索修复 JOIN：hash 0.0→0.583 达 e5 单跳水平；t5b+16 配对
  论证"evidence-mode 两大失败模式皆为协议缺口"；档案漂移 L 候选登记）→
  **PREREGISTERED 2（B-2 RUNNING / B-4 挂 GPU）/ CANDIDATE 6（B-12 / RLS 快照
  审计 / argparse / judge 加 rag_2hop 行 / 档案漂移 L / FwPKM-B10 更新）/
  DONE 11 / DEAD 4 / 映射结案 1**
- **下一方向**：**B-2 V14 收割**（~11:04 CST 截断后拉取产物按预注册判定）
  优先；次选 = judge 加行/RLS 审计等 mechanism 小项。

## 轮次追加（2026-09-16 05:5x）
- mechanism 双清（judge rag_2hop 行 + RLS 审计结案，05:29 轮）+ B-10 对照文档
  增补完成（Memora/FwPKM 入表，扫描轮 3 文档化收尾）→
  **PREREGISTERED 2（B-2 RUNNING / B-4 挂 GPU）/ CANDIDATE 3（B-12 挂 GPU /
  argparse 白班 / 档案漂移 L 待裁定）/ DONE 11 / DEAD 4 / 映射结案 1**
- **本地池清空确认**：无未立案可执行项 → 后续轮次按新规收工（晨报一行
  池子终态），直至 ~11:04 CST **B-2 V14 收割窗口**（唯一 PREREGISTERED
  主线，届时 `kaggle kernels output` 拉取 + 按预注册 parity 判据判定）。

### B-20 · 会话间"睡眠"巩固变换（sleep consolidation as PMB mechanism）· DEAD（2026-09-16 08:1x，两则机制发现）
- **扫描证据（扫描轮 4，2026-09-16 07:3x）**：SleepGate（arXiv:2603.14517，
  KV 记忆上的学习型睡眠周期：突触缩放/选择性重放/定向遗忘）；sleep-inspired
  replay 防灾难性遗忘（arXiv:2606.08447）；self-consolidation 蒸馏短期记忆
  （arXiv:2606.03979）——"agent 睡眠巩固"为 2026 活跃前沿。**与 PMB 的接口**：
  harness 的会话边界（snapshot 前钩子，B-15 已留 bump_uncertainty 先例）即
  "睡眠窗口"；B-14/B-17 的容量断崖给出精确靶点——巩固变换能否延迟/移动断崖。
- **变体设计（最便宜先行）**：(a) downscale（F←αF，α=0.9；**阴性对照**——
  sum 无 decay 场景下等价于 decay，预期无效，用于检验实验本身有区分度）；
  (b) **prune**（|F_ij| < θ 置零，θ 定为保留 ~10% 非零元——"稀疏化=去干扰"
  假设，呼应 FwPKM 稀疏槽位）。
- **预注册判据**：prune 变体 t1 难格 (64,16) @ d=256 3-seed 均值相比 sum
  0.385 上移 ≥ +0.10，**或** T3 gap=32 从 0.021 上移 ≥ +0.10 → DONE
  （"巩固操作"立项为新机制族，SleepGate 谱系 × PMB 独有断崖靶点）；两指标
  变化均 < +0.05 → DEAD（巩固变换对断崖无效）；中间 → 边界登记。
  downscale 预期 ≈0 变化（阴性对照自检：若它也大幅变化 → 实验实现有 bug）。
- **实现**：ParametricMemory 加 consolidate(mode, param)（snapshot 前调用），
  PMB CLI --consolidate 透传；state_bytes 同时报告非零率（稀疏化诚实入表）。
- **来源**：扫描轮 4；价值档 1。
- **执行（2026-09-16 08:1x）**：prune(keep 10%) / downscale(α=0.9) /
  downscale_z(真阴性对照) × t1 难格+全格 / t3，3 seeds，~5 分钟 CPU；
  test_pmb 9/9 过。
- **结果**：prune 难格 **0.120**（lift **-0.266**）· T3 gap=32 **0.0**
  （-0.021）→ 双指标未达 DONE，且均 <0.05 改善 → 触发 DEAD；downscale
  难格 **0.0**（"阴性对照"爆表）；downscale_z α 剂量曲线 **1.0→0.406 /
  0.99→0.359 / 0.9→0.0**（单调）。
- **阴性对照"失败"的真相（非实现 bug）**：α=1.0 恒等复现基线、效应随 α
  单调且随干扰会话数放大（(·,0) 格全 1.0，(·,16) 格全塌）→ 机制定位：
  **uniform 会话缩放 = 新近度加权 = 遗忘操作**。全局尺度在读出比值
  r=qF/(q·z) 中消去，但跨会话的相对尺度不消去——早会话事实缩 α^16≈0.19
  vs 末会话干扰 α^1≈0.9，读出方向被新近内容淹没。
- **两则机制发现（可引用）**：
  1. *"For linear associative memory with normalized reads, session-boundary
     scalar transforms cannot consolidate: global rescaling cancels in the
     read ratio, while per-session rescaling acts as pure recency forgetting
     (dose-response α=0.99→0.36, α=0.9→0.0). Scalar consolidation is either a
     no-op or a forgetting schedule."*
  2. prune 失败机制：外积态中信号元与干扰元同量级，幅值剪枝无差别摧毁两者
     （SNR 不升）——稀疏化≠去干扰。
- **谱系闭环**：内容感知巩固 = 重放/重解，恰是 micro_rls（存 pairs 重解 W）
  ——而 B-19 审计已证其失败在读出侧。巩固-存储-读出三环节的失败面就此完整
  映射：存储可解（B-17 容量墙）、巩固不可标量化（本轮）、读出侧未解（RLS）。
- **止损注记（§4.6）**：内容感知重放/读出侧归因属新方向，未立案；本条目
  DEAD 即最终态。
- **判定**：**DEAD**。

### B-21 · 迭代吸引子读出（Hopfield pattern completion 迁移检验）· DEAD（2026-09-16 08:4x，含异关联机制诊断）
- **扫描证据（扫描轮 5，2026-09-16 08:3x）**：Hopfield 赛道 2026 新论文均为
  物理实现/容量变体（复值 Hopfield、光子 dense AM、Rotor 高阶），无直接竞品；
  但经典谱系（Hopfield Networks is All You Need / Universal Hopfield Networks）
  的核心是**迭代吸引子读出**——PMB 全部系统均为单步读出。接口：B-19/B-20
  闭环的"读出侧未解"面——d=256 难格 0.385 意味状态里有部分信号，
  迭代补全（r ← F·φ(r) 归一化，φ=tanh(β·)）是经典修法。
- **信息论边界（预注册诚实注记）**：d=4096 时单步已 0.995（无头部空间）；
  d=256 断崖下若状态根本没存下信息，迭代也无从补全——本实验测的是
  "部分信号能否被迭代锐化"。
- **预注册判据**：read_iters=3（β=2）t1 难格 (64,16) @ d=256 3-seed 均值
  相比单步 0.385 上移 ≥ +0.10 → DONE（读出侧迭代修复成立，与 B-17 容量墙
  并列的第二修复轴）；变化 < ±0.05 → DEAD（"读出侧无解"结论加固为
  "迭代修复亦无解"）；中间 → 边界。
- **实现**：ParametricMemory 加 read_iters（recall 内迭代，sum/delta/falcon
  裸读路径通用）；PMB CLI --read-iters 透传。
- **来源**：扫描轮 5；价值档 1。
- **实现（已入库）**：read_iters/read_beta（recall 内迭代 r ← tanh(β·r)·F 归一化）
  + CLI 透传；test_pmb 9/9 过。
- **执行**：t1 全格 × 3 seeds（iter3）+ iter2 剂量点（s0），~40s。
- **结果**：iter3 难格 **0.000**（3/3）· iter2 s0 **0.0**——一次重进入即坍塌。
- **机制诊断（可引用）**：**异关联记忆（k≠v 无关空间）不能直接套自关联
  Hopfield 迭代**——首步读出 r≈v_t 落在值空间，重进入 φ(v_t)·F 的键值重叠
  φ(v_t)·k_j ≈ 0（随机向量正交），信号在首次重进入即消失，剩余全为噪声。
  经典正解 = Kosko 双向联想记忆（F/Fᵀ 交替读出）。
- **预注册缺口（诚实记录）**：判据只写了"+0.10 DONE / ±0.05 DEAD / 中间边界"，
  未覆盖大负变化——-0.385 按意图判 DEAD，负变化区间应显式入判据模板
  （并入档案漂移 L 候选一起修订）。
- **判定**：**DEAD**（朴素迁移判负；正解机制浮现 → B-22 CANDIDATE）。

### B-22 · Kosko 双向交替读出（BAM：F/Fᵀ 异关联迭代）· DEAD（2026-09-16 09:1x，读出侧终局加固）
- **依据**：B-21 机制诊断——异关联迭代必须交替使用正向（k→v，Fᵀ）与反向
  （v→k，F）两个映射；Kosko BAM 谱系。叠加态两向信息都在（F 同时含 k⊗v）。
- **判读标准（草案）**：BAM 交替 2-3 轮 t1 难格 @ d=256 上移 ≥ +0.10 →
  读出侧修复成立（与 B-17 容量墙并列）；< ±0.05 → "读出侧无解"终局加固。
- **来源**：B-21 机制诊断；价值档 1。
- **转正注记（08:5x）**：判据即上方草案，先于执行 commit。实现 =
  `read_protocol="bam"`（read_iters 轮交替：kc = F·tanh(β r) 值→键，
  r = tanh(β kc)·F 键→值，各步归一化——用经实证的 direct 读出映射对，
  非转置臆测）。09:00 B-2 仍 RUNNING。
- **执行（09:1x）**：bam3 × 3 seeds + bam2 s0 剂量点，~40s CPU；test 9/9 过。
- **结果**：难格 **0.000**（3/3；bam2 s0 亦 0.0——第一轮交替即坍塌）；
  全格退化随干扰会话数加重（(4,0)=1.0 但 (64,0)=0.375，(·,16) 全 0）——
  经典 BAM 伪稳态/混合态失败。
- **判定：DEAD——读出侧终局加固**：朴素同映射迭代（B-21）与 BAM 双向交替
  （B-22）两套经典读出修复全部无增益且有害。三环节失败面终版（可引用）：
  *"On a capped additive state, neither naive nor bidirectional Hopfield-style
  re-entry improves hetero-associative recall (0.385→0.0 for both): partial
  signal under interference is not completable by readout — the only
  effective axis is state capacity (B-17: d=4096 → 0.995)."*——存储可解
  （容量墙）/ 巩固不可标量化（B-20）/ 读出侧迭代与双向皆无增益（B-21/B-22）。
- **判据模板缺口第二次命中**（大负变化在字面区间外）→ 判据模板修订升格为
  mechanism 小项（与档案漂移 L 合并，下轮顺手做）。
- **下一方向**：B-2 V14 收割（~11:04 CST）优先。

## 扫描轮 5（2026-09-16 08:3x）：Hopfield/抗干扰赛道
- 2026 新论文（复值 Hopfield / 光子 dense AM / Rotor 高阶）均为物理实现或
  容量变体，无可立案竞品；经典谱系的迭代读出机制 → **B-21 转正**（读出侧
  最后一块的迁移检验）。
- 统计：PREREGISTERED 4（B-2 / B-4 / B-20 DEAD 后为 2 + B-21）/ CANDIDATE 3 /
  DONE 11 / DEAD 5（含 B-20）/ 映射结案 1

### B-23 · Filesystem 简单基线入 PMB（社区争论仲裁实验）· DONE（2026-09-16 09:3x，全任务族击穿 + 社区结论可信复现）
- **社区证据（扫描轮 6，09:2x，社区赛道首扫）**：Letta《Is a Filesystem All
  You Need?》宣称纯文件+grep 在 LoCoMo 上 74.0% 胜 Mem0 68.5%；MemGPT 作者
  公开提醒"别信在线记忆基准"（LoCoMo 局限）；Label Studio 指 LoCoMo 近饱和
  不迁移真实 agentic 场景；r/LocalLLaMA 称厂商对比为"WWE feud"——**社区最热
  痛点 = 记忆基准不可信/无区分度**，恰是 PMB 设计主张（程序生成+精确评分+
  防枚举截断+机械持久化）的靶场。
- **实验**：新增 `FilesystemSystem`（纯文本落盘 + 问题词 grep 式子串检索，
  命中行截断至预算——Letta 式平凡基线的 PMB 化），全任务 T1-T5 × 3 seeds
  与 rag(hash) 同跑同验收。
- **预注册判读（按任务族胜负矩阵）**：(a) 任一任务族 filesystem > rag+0.05
  → 登记"简单基线胜出格"（社区结论在可信 harness 下复现 = 互证，论文素材）；
  (b) 全任务 filesystem ≤ rag → "检索层有增量价值"（mem0 侧辩护成立）；
  (c) 方向性预测两格：t5 幻答率 = 100%（无拒答机制）、t4 ≈ 单跳 rag（同
  JOIN 协议缺陷）——落空即记边界。state_bytes 差距（O(history) vs 向量库）
  同表诚实入档。
- **来源**：社区赛道首扫；价值档：学术 1 + 商业 1（直接回应社区最热争议，
  PMB 定位 = 争议仲裁基准）。
- **实现（已入库）**：`FilesystemSystem`（evidence 模式：全文落盘 + 问题词
  overlap grep 排序返回命中行，无嵌入无索引，state_bytes=原始文本 O(history)）
  + factory/CLI 接线；test_pmb 9/9 过。
- **执行**：--task all × 3 seeds（filesystem 与今日同跑 rag 对拍，~7s）。
- **胜负矩阵（3-seed 均值，filesystem vs 同跑 rag）**：
  t1 难格 **1.000 vs 0.802**（+0.198）· t2 **1.000 vs 0.917**（+0.083）·
  t3 gap32 **1.000 vs 0.875**（+0.125）· **t4 JOIN 难格 1.000 vs 0.083**
  （**+0.917**）· t5 known 1.000/幻答 **1.000**（方向性预测 ✓）·
  t2 stale 0.000 vs 0.042。
- **预注册判据 (a) 多格触发 → DONE**：**社区结论在可信 harness 下全面复现**
  ——verbatim 文件系统在所有任务族上 ≥ 检索式；方向性预测 (c) t5 命中、
  t4 被超越（filesystem 1.0 > 单跳 0.083：grep 读全量天然无 JOIN 协议缺陷）。
- **重排行（同一 (64,16) 集）**：filesystem 177KB vs rag 2.27MB（**≈13× 更小**
  且 recall 更高+JOIN 满分）——"平凡基线又小又强"。
- **学术结论（可引用）**：*"On PMB, a verbatim filesystem with grep solves
  every task family at smaller state than a vector-store retrieval layer —
  including the cross-session JOIN that defeats every memory system
  (1.00 vs 0.08-0.63) — reproducing the community's 'filesystem beats memory
  layers' finding under a program-generated, anti-enumeration harness. PMB
  arbitrates the debate: specialized memory layers must justify themselves
  per task family against the trivial baseline."*
- **连带改判**：t4"无解格"叙事修正——它对"受限读出系统"无解、对"全量读出"
  平凡可解；judge 复算表加 filesystem 行（mechanism 小项，下轮）。
- **判定**：**DONE**。

## 扫描轮 6（2026-09-16 09:2x）：社区赛道首扫（新提示词条款首次使用）
- **社区赛道（命中）**：LoCoMo 基准争议（Letta filesystem 74.0% vs Mem0
  68.5%、MemGPT 作者"别信在线基准"、Label Studio 饱和论、r/LocalLLaMA
  "WWE feud"）→ **B-23 转正 PREREGISTERED**；B-10 文档待补社区争论节
  （下轮与 B-23 实现同批）。
- **生态数字（社区侧，引用时需复核时点）**：Mem0 ~41-62K stars/$24M、
  Graphiti ~28K、Letta ~24K——赛道商业热度实锤（价值档 2 证据）。
- **统计（扫描轮 6 后）**：PREREGISTERED 3（B-2 收割 / B-4 挂 GPU / B-23）/
  CANDIDATE 4（判据模板修订+档案漂移 L / B-12 / argparse）/ DONE 11 /
  DEAD 6 / 映射结案 1
## 轮次追加（2026-09-16 09:1x）
- B-22 执行判负 **DEAD**（读出侧终局加固；判据模板缺口第二次命中 → 修订升格
  mechanism 小项）→
  **PREREGISTERED 2（B-2 收割 / B-4 挂 GPU）/ CANDIDATE 4（判据模板修订+档案
  漂移 L / B-12 挂 GPU / argparse 白班）/ DONE 11 / DEAD 6 / 映射结案 1**
- **下一方向**：**B-2 V14 收割**（~11:04 CST）优先；间隙 = 判据模板修订
  mechanism 小项。

## 轮次追加（2026-09-16 10:0x）
- mechanism 双清：**L009 立案**（kb/lessons/L009-same-run-cross-time-comparison.md，
  跨时点对照必须同跑）+ **判据模板双侧化修订**（backlog 头部，B-21/B-22 两次
  大负变化落在判据字面外的缺口关闭）→ 两项出池 →
  **PREREGISTERED 2（B-2 收割 / B-4 挂 GPU）/ CANDIDATE 2（B-12 挂 GPU /
  argparse 白班）/ DONE 11 / DEAD 6 / 映射结案 1 / L 工件 +1**
- **下一方向**：**B-2 V14 收割**（10:0x 仍 RUNNING，截断 ~11:04 CST）——
  下一或下下轮进入收割流程。
- **B-2 提速诊断（2026-09-16 10:1x，用户质询触发；V15 改 launcher 不改语义）**：
  ① **3 lane 跑 4 核**——深度只有 3 组，第 4 核设计性闲置（-25% 算力）；
  ② **OMP 超订**——launcher 的 `torch.set_num_threads(1)` 只作用于父进程，
  3 个子进程各自默认 4 OMP 线程 → 12 线程挤 4 核（未控变量）；
  ③ 根本原因：18 配置 × 30K 步 × batch128 ≈ 33 core-hours，CPU 容器就是慢
  （GPU 通道阻塞的代价）。
- **V15 启动器已入库**（`kaggle_kernels/tau_ladder_probe/run_tau_probe.py`，
  从 Kaggle version 14 精确拉取源码重建——V14 源码此前从未入库，档案缺口
  顺手关闭）：**4 lane × 单线程**（OMP/MKL/OPENBLAS/NUMEXPR 环境钉死 +
  (depth,seed) 对队列 round-robin 3/2/2/2）。**会话窗数学**：12h 硬窗下
  吞吐 = lane 数 × 单 lane 步速 → 4 lane 预期 ~8.8 配置/会话（V14 3 lane
  ~6.5）——lane 数是硬窗下的吞吐杠杆，config 均衡度次要。
  本地 A/B（200 步）：OMP=4 比 OMP=1 快 24%（35.2s vs 46.5s，本地核多
  不超订）——workload 线程敏感，4 核上钉死 4×单线程最稳（L010 铁律）。
  metadata 同步为 CPU（旧 PR 分支 metadata 是 9/5 的 T4 版）。
  更大杠杆仍是 GPU 通道（#53+Secret，5.5h/配置 → 分钟级）。
## 扫描轮 4（2026-09-16 07:3x）：巩固/LNN 赛道（用户质询触发，配额规则已放宽）
- **巩固赛道（命中）**：SleepGate 2603.14517 / sleep replay 2606.08447 /
  self-consolidation 2606.03979 → **B-20 转正 PREREGISTERED**。
- **LNN 架构赛道（无新进入者）**：LTC 基础架构无实质演进，2026 动态为
  ODE 混合与应用侧——主轴无需更新。
- **Hopfield/抗干扰赛道**：搜索超时，下轮轮换重试。
- **规则更新**：池空收工轮由"当日至多一次扫描"放宽为"每轮可扫一个不同
  赛道（同赛道当日不重扫）"（用户 07:2x 质询驱动）。
- **统计（扫描轮 4 后）**：PREREGISTERED 3（B-2 收割 / B-4 挂 GPU / B-20）/
  CANDIDATE 3（B-12 / argparse / 档案漂移 L）/ DONE 11 / DEAD 4 / 映射结案 1

## 轮次追加（2026-09-16 08:1x）
- B-20 执行判负 **DEAD**（prune 伤 recall -0.27；标量巩固=无操作或遗忘，
  剂量曲线实证；巩固-存储-读出失败面闭环）→
  **PREREGISTERED 2（B-2 收割 / B-4 挂 GPU）/ CANDIDATE 3（B-12 / argparse /
  档案漂移 L）/ DONE 11 / DEAD 5 / 映射结案 1**
- **下一方向**：**B-2 V14 收割**（~11:04 CST）；间隙轮次 = Hopfield 赛道
  补扫（扫描轮 4 超时项）或收工。

### B-16 · t4 JOIN 攻坚：两跳检索基线 · DONE（2026-09-16 05:1x，协议缺口论点第二实例落地）
- **依据**：t4 JOIN 全系统未解（rag hash 0.0 / e5 0.625）；mem0《State of AI
  Agent Memory 2026》把 **cross-session identity resolution** 列为行业最难开放
  问题之一——PMB t4 正是该问题的机械最小化（锚点在 A 会话、答案在 B 会话）。
  扫描候选机制：fast-weight product key memory（arXiv:2601.00671）——**读文结论
  （05:0x）**：序列内稀疏 episodic 层（TTT 式槽位更新），只测 LM 困惑度 +
  NIAH，无跨会话评测、无公开代码 → 归 B-10 对照，不作本轮实现依赖。
- **最便宜实验（本轮执行）**：`RagTwoHopSystem`（继承 RagSystem，同 hash 编码器
  同索引）：hop-1 = 问题检索 top-3 内机械解析评分→人名（解析不到则退化为
  单跳）；hop-2 = 以人名为查询检索 top-2 返回证据。定位 = **协议上限参照**
  （类 oracle 角色）：检验"JOIN 失败是单跳检索协议缺口而非信息/编码器缺口"
  ——与 t5b"拒答失败是协议缺口"同构，构成协议缺口论点的第二个实例。
- **预注册判负/判读标准**：rag_2hop(hash) t4 难格 (8,4) 3-seed 均值——
  **> 0.5** → DONE："JOIN 失败是协议缺口"成立（升格论文素材，与 t5b 并列）；
  **≤ 0.05**（≈rag hash 0.0）→ DEAD："连锚点都检不出"，失败在编码器/信息层
  而非协议；**0.05-0.5** → 边界登记（部分恢复，对照 e5 0.625 语义上限）。
  附带门：t1 全网格与 rag 基线一致（非 JOIN 任务单跳路径不变，sanity），
  state_bytes 同量级。
- **来源**：扫描轮 3；价值档 1（t4 是全矩阵唯一无解格 + 行业最难问题最小化）。
- **实现（已入库）**：`RagTwoHopSystem`（继承 RagSystem）：hop-1 = 问题检索
  top-3 内机械解析评分→人名（解析不到退化单跳）；hop-2 = 人名查询 top-2。
  **实现教训**：首版正则按文档模板写（要求 "out of 5"），而当前 t4 问题模板
  是 "...rating is {r}?"——两跳从未触发、成绩全是单跳回退（与今日 plain rag
  逐格一致暴露）。修正后重跑才得到真两跳数据。**验收句式自查救了一次数值
  误读。**
- **执行**：t4 × 3 seeds @ d=256（0.4s）+ t1 × 3 sanity。
- **结果（3-seed 均值）**：t4 难格 (8,4) **0.000 → 0.583**（[0.375, 0.875, 0.5]，
  阈值 >0.5 触发 DONE）；(8,0) **0.0 → 1.000** 全格满分；hop-1 命中率 58%
  （top-3 检出评分句），残余失败 = hop-1 未命中 + hop-2 干扰会话竞争。
  t1 sanity：与今日同跑 plain rag 全格一致 ✓。
- **学术结论（可引用）**：*"Cross-session JOIN failure is a retrieval-protocol
  gap, not an information gap: a two-hop protocol (rating→person→code) lifts
  hash-encoded retrieval from 0.00 to 0.58 on the hard cell — parity with the
  semantic-encoder single-hop (0.625) — mirroring the abstention finding.
  Both headline failure modes of evidence-mode memory are protocol gaps."*
  ——t5b（拒答=阈值缺口）+ B-16（JOIN=协议缺口）构成配对论证。
- **⚠️ 档案漂移注记（L 候选）**：今日同跑 plain rag 的 t4/t1 数字与 B-8/B-6
  档案存在 1-2 题/格漂移（(4,4) 0.0→0.25）——任务生成器自 9/9-9/11 演进，
  **旧 JSON 不对当前代码逐位可复现**；跨时点对照必须同跑。矩阵历史格数字
  属各自当时版本，引用时注明。
- **判定**：**DONE**——t4 从"全系统无解格"变为"协议可解格"；判读工具 judge
  的 rag 行可加 rag_2hop（下轮顺手）。

## 扫描轮追加（2026-09-11 夜班）
- **B-12 降优先**：MoNe（arXiv:2608.17616）未发现公开代码（Qualcomm AI Research，
  HTML 版无 repo 链接）→ 无实现依赖则无法作为 PMB 参赛系统，保持 CANDIDATE 挂起。
- **新发现**：MemoryBench（arXiv:2510.17281 v5）——LLM 系统记忆与持续学习基准，
  与 AgentMemBench/StateMemBench 同赛道 → 归入 B-10 对照文档的更新范围（不影响
  三差异化存活的判定：遗忘曲线/位级持久化/JOIN 仍无覆盖）。

### t5b-e5 叠加验证 · DONE（2026-09-12 凌晨，编码器×拒答正交性确认）
- **e5 阈值标定**：known top-1 分数 [0.857-0.900] / unknown [0.791-0.817]，
  **完美分离**（工作点 0.84：幻答 1.0→**0.0**，known recall 1.0 不动）。
- **正交性确认**：编码器（hash/e5）× 阈值拒答两维独立——hash 需阈值 0.5、
  e5 需 0.84（各自标定），叠加后双双达成幻答 0 + known 1.0。
- **结论**：拒答修复对任意编码器通用（各编码器单独标定阈值即可）。

## 扫描轮 2（2026-09-12 00:4x）：科研 + 商业双通道
- **科研通道**：Falcon(2608.27763) 已入 B-7 对照；同域无 8/27 后更新论文——
  fast-weight 前沿暂稳，无新竞品。MoNe 无公开代码维持降优先。
- **商业通道**（价值档 2）：mem0/Zep/Letta/LangMem 2026 对照报告确认
  agent memory 赛道有活跃的商业对比生态（niteagent 2026-05 对比文、
  mem0 官方 2026 报告）——PMB 若开源并引用此赛道，定位自然入列。
  无需新增方向，B-10 对照文档已覆盖。
- **结论**：扫描未浮现新的可执行方向 → 本轮以 B-13/t5b 收尾，方向池维持。

### B-11 · MCB memory-clarification boundary 映射 PMB · DONE（2026-09-12 映射结案，非实验方向）
- **MCB 四分法**（arXiv:2608.19564）：persist（持久化）/ use（仅当前上下文）/
  verify（再验证）/ ask（向用户澄清）——LLM 决策层评测（Claude/Qwen 准确率）。
- **与 PMB 的映射**：
  - MCB **persist** ↔ PMB t1/t2（跨会话写入与更新覆写）——已有覆盖；
  - MCB **use** ↔ PMB t3/t4（窗口内使用与跨会话 JOIN）——已有覆盖；
  - MCB **verify/ask** ↔ PMB t5 拒答是**机械基底**（记忆系统正确返回空 =
    LLM 层正确 ask 的前提），但决策层本身不在 PMB 范围内；
- **判定**：**DONE（映射结案）**——MCB 是 LLM 决策层评测，PMB 是记忆本体评测，
  两者正交互补。t5 拒答（阈值 -92%）是 MCB ask 决策的机械基础。
  B-11 不需要独立实验——映射已完成，后续若 PMB 加 generative 模式参评
  （LLM 生成答案），可直接引用 MCB 四分法作为决策分类学。
- **来源**：扫描轮 2（arXiv:2608.19564）；价值档 2（映射结案后降级归档）。

## 统计追加（2026-09-16 凌晨轮）
- B-13' DONE（T2 矩阵 9 系统收尾，9 JSON 一手入库，3 冗余副本清理）→
  **PREREGISTERED 2（B-2 RUNNING / B-4 挂 GPU）/ CANDIDATE 2（B-12 挂 GPU / argparse 工程债）/ DONE 6 / DEAD 3 / 映射结案 1**
- **下一方向（强制浮现）**：① **B-2 V14 收割**——12h 上限预计 9/16 ~03:04 UTC
  （~11:04 CST）截断；届时 `kaggle kernels output aricredemption/m1-b2-tau-ladder-cpu`
  拉取已完成配置 JSON 入库；若 parity 域配对（d2/d4/d8 × 3 seeds）齐 → 按预注册
  判定（30K 步内 acc 全 <0.25 → 任务不可行 DEAD；learnability gate 10K 步 acc>0.2）。
  ② GPU 通道仍阻塞（#53 + Secret）→ B-4/B-12 维持挂起。③ 池内本地可执行项耗尽 →
  下轮若 V14 未到收割点，执行扫描轮浮现新方向。

## 扫描轮 3（2026-09-16 02:4x）：前沿整合评估（用户指令即时触发，当日扫描配额用讫）

- **赛道 1 fast-weight / TTT**：新入视野——**Kalman Delta Networks（2609.07816，
  2026-09）→ B-15 转正**；Rethinking TTT expressivity（2608.21308，理论）、
  End-to-End TTT for Long Context（2512.23675，Nvidia）、Learning to Write
  Context into Memory with TTD（2603.13875，写入门控）、FW Product Key Memory
  （2601.00671）归入 B-10 对照/B-16 候选。MoNe（2608.17616）仍未见公开代码，
  B-12 维持挂起。
- **赛道 2 基准竞品**：**Memora（2604.20006，weeks-months 个性化对话记忆，
  remembering/reasoning/recommending 三任务 + FAMA 失效感知指标）**——
  核对结论：LLM 记忆体（非位级序列化）、无心理遗忘曲线（FAMA 是失效感知
  非衰减曲线）→ **PMB 三差异化（遗忘曲线全曲线/位级持久化机械强制/JOIN+拒答）
  全部存活**，归入 B-10 引用对照更新。mem0《State of AI Agent Memory 2026》
  确认 cross-session identity resolution + temporal reasoning 为行业最难开放
  问题 → t4 JOIN 行业定位强化（B-16 依据）。
- **赛道 3 联想记忆理论**：谱优化器容量 scaling（2603.26554）→ B-17 理论锚；
  Cabannes 2310.02984 外积标度律经典（引用底座）。
- **结论**：池子由空转补——**PREREGISTERED +3（B-14/B-15/B-17），
  CANDIDATE +1（B-16）**。执行序建议：B-14（最便宜、差异化证据）→ B-17
  （秒级）→ B-15（新写规则实现）；B-2 V14 收割窗口 ~11:04 CST 优先插入。
- **统计（扫描轮 3 后）**：PREREGISTERED 5（B-2 RUNNING / B-4 挂 GPU /
  B-14 / B-15 / B-17）/ CANDIDATE 3（B-12 / B-16 / argparse 工程债）/
  DONE 7 / DEAD 3 / 映射结案 1

### B-24 · 多级算力路由器（tiered_run v0）· DONE（2026-09-16 10:4x，社区扫描→工程落地）
- **社区证据（扫描轮 7）**：多级算力编排社区三主流——SkyPilot（跨云套利 +
  on-prem）/ dstack（on-prem+云控制面）/ Modal（serverless、贵、锁单云）；
  个人实践常见"训练 SkyPilot + 推理 Modal"。**三者均不覆盖 Kaggle 免费档**，
  而 Kaggle CPU/T4 是 M1 的 T1/T2 主力 → 自建零依赖轻路由（~150 行）而非引
  SkyPilot 依赖；未来接竞价云（RunPod/Vast）再评估 SkyPilot。
- **实现（已入库）**：`scripts/tiered_run.py`（T0 本机/T1 kaggle-cpu/
  T2 kaggle-t4/T3 a100 四层路由决策 + 各层 launch/harvest recipe +
  **L010 检查项内置**——launcher 未钉死 OMP 拒绝 push）+
  `docs/COMPUTE_TIERS.md`（层级表/路由规则/各层 runbook/社区谱系）。
- **验证**：四层 dry-run ✓；L010 负例拦截（无钉死 launcher 拒绝，rc=2）✓；
  T0 实跑 ✓；T3 base64 引号安全 ✓。
- **首个真实用例**：B-2 tau-ladder（T1 层）V15 推送；T3 首用待 A100 级实验
  （如 2b 训练续跑）。
- **来源**：用户质询（多机器架构社区实践）；价值档：工程 1。
- **⚠️ 台账乱序注记**：本文件节顺序因多次锚点插入已非线性（今晚新增节散布
  于 626-919 行区间，9/11 旧节在尾部）——内容经 grep 全量核对无丢失；
  重排登记为白班 mechanism 项。

## 轮次追加（2026-09-16 11:3x）
- **B-24 运营接线完成**：B-2 的声明式作业规格入库
  （`kaggle_kernels/tau_ladder_probe/job.json`：slug/harvest_dest/
  resume_safe 全字段），tiered_run 的 status/harvest 已在真实 kernel 上
  跑通（含 L010 负例回归）。**今晚 ~22:53 V16 截断后的收割命令**：
  `python scripts/tiered_run.py harvest --job kaggle_kernels/tau_ladder_probe/job.json`
  （产物落 `benchmarks/results/tau_ladder/`，随后按预注册 parity 判据判定）。

## 轮次追加（2026-09-16 11:0x）
- **B-24 增补（checkpoint 契约，多主机训练状态汇聚）**：训练状态 = 按配置
  原子 JSON；汇聚点 = Kaggle Dataset `aricredemption/m1-tau-ladder-ckpt`。
  harvest（本地有凭证）自动推版本；launch（任意主机）经
  metadata.dataset_sources 挂载 + launcher 启动并入 working/results →
  **跨会话/跨主机不重跑已完成配置**。降级链：数据集缺失/DNS 故障 → launch
  剥离挂载、harvest 警告（git 状态安全）。已验证：403 缺失检测、剥离降级、
  引号安全。创建数据集时遇本机 DNS 间歇故障（www.googleapis.com 解析失败），
  今晚 22:53 收割时以真实 JSON 重试建库。
- **边界说明**：B-2 类探针的并行是配置级（18 配置独立），多主机 = 分领配置
  + checkpoint 汇聚，无需 torchrun/NCCL；梯度同步级大训练（M2 2B）走 T3
  A100 单机多卡（#57 fsdp 骨架），另一条线。

## 轮次追加（2026-09-16 22:3x）
- **A100 证据分支合并**（origin/evidence/tau-b2-flat-a100 → overnight/loop，
  零冲突；18 配置 JSON + verdict 入 `benchmarks/results/tau_ladder/`）→
  **B-2 正式 DEAD 结案**（flat/天花板，judge_verdict delta_gain=0，诚实标注
  "非干净判负而是协议无法检验 H004"）；**B-2' 难度硬化重测立项 PREREGISTERED**
  （A100 通道，判据同款）。Kaggle V16 判据冗余，令其 22:53 自然截断，
  产出与 A100 证据重复不入库。→
  **PREREGISTERED 3（B-4 挂 GPU / B-20 已 DEAD 出池 → B-2' / 复核：B-2' 新入）/
  CANDIDATE 2（B-12 挂 GPU / argparse 白班）/ DONE 12（含 B-23/B-24/B-2 结案）/
  DEAD 7（含 B-2）/ 映射结案 1 / L 工件 L009-L011**
- **validate_results 8/8 PASS**（含 A100 证据合并后）。
- **下一方向**：B-2'（A100 通道，判据已写死）；间隙 = 完成性收尾（PR #58
  描述更新，不合 main）。

## 轮次追加（2026-09-16 22:5x）
- **B-2 V16 收割完成（Kaggle 独立复现）**：V16 在 12h 窗完成 **7/18 配置**
  （mean_acc ≈ 1.0，与 A100 判决互证——不同硬件同结论，flat 结案 doubly
  confirmed）。**L010 剂量实证兑现**：单配置 wall 4.70h（OMP 钉死）vs V14
  同容器 5.5h，提速 ~15%（接近本地 A/B 的 24% 上限）。证据谱系分目录：
  `tau_ladder/`=A100 权威集（18+verdict）、`tau_ladder_kaggle_v16/`=Kaggle
  复现集（7 配置 + lane 日志）。
- **checkpoint 数据集建库顺延**：本机 DNS 间歇故障（googleapis 解析失败），
  推送失败已按降级链处理（git 状态安全）；下次 harvest 自动重试建库。
- **⚠️ mechanism 小项登记**：tiered_run harvest 覆盖同名文件事故（已恢复，
  L009 式警觉抓出）——harvest 应检测 dest 同名文件并改道谱系子目录。
- **B-2 系列终态**：B-2 DEAD（flat/天花板，A100+Kaggle 双硬件复现）/
  B-2' 硬化重测 PREREGISTERED / V17 全量 Kaggle 复跑 = 可选（边际价值低，
  默认不做）。**tau-ladder 线关闭**。

## 轮次追加（2026-09-17 21:5x）
- **B-2' staircase 校准完成并收割**（kernel m1-tau-staircase-b-2-calibration，
  T4，344min，seed 0，15 文件入 `benchmarks/results/tau_staircase/`）：
  - k=32：ladderoff d4-d1 = **-0.024**，ladder ON = **+0.054**（0.910→0.964）
    —— **τ 阶梯把深度效应从负转正，H004 预言的形状首次出现**（单 seed 待证）；
  - k=64：两臂中性（±0.02）；k=128：**d4 双臂撞墙**（gate 0.5/chance，mean
    ~0.51）而 d1 能学（0.74）——高难度下深度本身有害，阶梯无关。
- **带内定标：k=32**。预注册完成步 = 18 配置（2 ladder × 3 depths × 3 seeds）
  × 30K 步 @ k=32（A100 通道，judge_verdict 同款 2σ+符号检验）→ H004 正式
  判定。注意：+0.054 为 seed 0 单点，k=128 的 d4-wall 信号（深度有害，
  ladder 无关）一并入档。
- **L011 修正**：运行中 kernel 的 Logs 标签会流式显示日志（11:3x 实测），
  推送后看进展直接开该页。

### H004 正式判定 · k=32 18 配置全矩阵（2026-09-17 A100 通道，4 workers）
- **执行**：2 ladder × 3 depths × 3 seeds × 30K 步 @ k=32，A100 GPU+CPU 并行，
  resume-safe 竞赛（预注册完成步的完整执行）。
- **结果**：h_supported=**false**，诊断 **flat**——gain_off(d4−d1)=−0.0408、
  gain_on=−0.0258、delta_gain=+0.015（阈值 2σ=0.1636）、符号检验 p=1.0（1 胜 2 负）。
- **三层诚实解读**：
  ① 校准的 +0.054 单 seed 信号**未通过 3-seed 2σ 检验**（收缩为 +0.015）——
  预注册判据按设计拦下了 grokking 伪影；
  ② **方向性线索保留**：OFF 臂 3-seed 确认深度在 k=32 有害（−0.041），τ 阶梯
  软化到 −0.026——符号方向与 H004 预言一致，但幅度 ~10× 低于噪声；
  ③ **测试欠 powered**：种子方差 σ≈0.04-0.08（grokking 双峰），2σ 阈值 0.164；
  分辨 +0.05 级效应需 ~10-20 seeds——**这是预算问题，不是机制判负**。
- **H004 正式结论**：**不支持（当前预算）**——"信号方向存在、幅度被方差淹没"。
  若继续：加 seeds（每臂 10+）或降方差（多 lr / 训练至收敛而非 grok 边界）。
- **证据**：`benchmarks/results/tau_ladder_k32/`（18 配置 JSON + verdict + README）。
