# notes/overnight/backlog.md — 夜间迭代方向池

> 状态机：CANDIDATE(方向+外部扫描来源) → PREREGISTERED(判负标准+最便宜实验方案)
> → RUNNING → DONE(结果 JSON 路径) / DEAD(判负触发，注明证据)。
> 判负方向标 DEAD 带证据，绝不在同方向重开（§4.6 方向版止损）。
> 价值排序：前沿学术化（学界会引用吗）> 被验证过的商业模式（有人用真金白银验证过吗）；
> 两档皆无不为凑数登记。
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

### B-2 · tau-ladder 探针恢复推送（GPU，通道解锁后立即执行）
- **前置阻塞**：#53 未合并 + Kaggle Secrets 未配 GITHUB_TOKEN（用户人工步骤）。
- **判负标准（预注册）**：18 配置中 parity 域配对（d2/d4/d8 × 3 seeds）在
  30K 步预算内 acc 全部 < 0.25 → tau-ladder 判定任务不可行，ADJ-001 的
  替代判决任务线标 DEAD；learnability gate（10K 步 acc>0.2）作为会话内止损。
- **最便宜实验**：分片推送（单实例 ≤2 臂，ADJ-004），T4 ~4h/30k 步趟。
- **验收门**：kernel 输出 JSON 与本地复算逐字节一致 + publishable() 纪律。
- **来源**：ADJ-001（budget_wall 脱出）、kb/threads/T001、kaggle_kernels/tau_ladder_probe/。

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
