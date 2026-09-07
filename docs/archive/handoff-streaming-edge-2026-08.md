# 归档：不规则采样证据线叙事（原 HANDOFF §2.9/§2.10，2026-08-29..30）

> 冷热分离迁移（2026-09-06）：两节均已收口（CT 机制 CLOSED、批次2 证据已入库），
> 按"活文档只放当前态"纪律从 HANDOFF 移入。判定态见 RESULTS.md Null 区、
> BENCHMARKS §6/§7/§8；决策见 kb/decisions/ADJ-007/ADJ-008。

## 2.9 机制裁决:CT 叙事关闭(2026-08-29 已跑完,CLOSED)

`mt_lnn_dt`(LiquidDTRegressor, `battery_soh_edge.py`)把逐步衰减改为
`λ_t=exp(-Δt_t/τ)` 走通用 pscan,P/S/blend 结构对齐生产线,共享 MLP 补容量
(参数 50,857,在 mt_lnn/gru 带内,单测强制)。判定 R1(分布内 ≥2/3 个 cv=1
档胜四对手)/R2(训练 Δt 0.05→测试 0.4 外推档全胜)预注册,实测 **R1=0/3、
R2 负**:分布内全不显著(dt0.15 档显著输 gru t=−2.09);外推档探针 0.3005
赢 lstm(+4.41)/特征版 mt_lnn(+8.85) 但输 **gru_d 0.2485(−1.74)**——
可学习衰减比固定结构衰减更抗 Δt 偏移。**机制故事按预注册规则关闭,
全仓库不再讲 CT 叙事。** JSON `benchmarks/results/synth_ct_control_dt.json`
(per-seed/config 回显/git 哈希,evidence PR #23 入库);归档见 RESULTS
Null 区 + BENCHMARKS §6。

**后续**:air 域加固(Dingling 10 seeds + 第二留出站 Gucheng 5 seeds)在
原分支 `iter/process-train` 发车,结果未入 main——按证据轨另行清偿,清偿
前 air 行维持现状。落地叙事收缩为"空气质量类任务实测最优 + 可部署"。

## 2.10 不规则采样证据线:GRU-D 判负 + 机制布线诊断 + 混合配方(2026-08-29..30,批次2清偿自 iter/irregular-streaming-edge)

**结论(全部预注册判定,数字见 RESULTS.md 对应行与 benchmarks/results/*.json)**:

| 项 | 结果 |
|---|---|
| battery 域(NASA, d=78, 10 seeds) | **判负**:GRU-D 吸收测试触发——\|t(mt_lnn,gru_d)\|=1.75<2 且 gru_d 退化 +12.8pp 反好于 mt_lnn +20.8pp;旧 d=65 的 +7.7% 边际在 d=78 未复现。**"架构级不规则采样优势"主张已撤回** |
| air 域(UCI 北京, held-out Dingling, 5 seeds) | **唯一存活 WIN**:60% 丢采样双目标对 gru_d/lstm/gru 全部 \|t\|≥2.5,且是本线唯一显著胜 transformer 的任务(PM2.5 24.70 vs 26.21) |
| synth 域(Van der Pol, 5 seeds) | **判负**:cv=1 全档对 gru_d/lstm/gru 无一显著——对机制叙事是真实反证 |
| 总判定 | 预注册 2/3 规则 → **1/3,整体判负**,已如实入库 |
| 部署画像 | ONNX parity 5.96e-08 / 变 Δt 1.19e-07 / 单核 26.0µs/步 / 状态 3,120B(非独有)/ int8 此规模不缩液态图(1,616 vs 1,480 KiB) |

**根因诊断(代码已验证)**:`mt_lnn/mt_lnn_layer.py:55` 的衰减是
`exp(-config.dt/τ)`,**dt 为构造期固定标量**——三域基准里 Δt 只经 inp 特征进入,
与喂给 RNN 基线同通道;连续时间机制从未接线。旧 pilot(+7.7%)同布线,
本就不是机制性证据(t=2.40/2.63 @ n=10 也在显著边缘)。

**机制裁决实验(2026-08-29,CLOSED)**:`mt_lnn_dt` 探针 R1/R2 双判负,CT 叙事全仓
关闭——判定全文见 §2.9(判定轨 PR #26),证据 PR #23,BENCHMARKS §6,此处不重刊。

**后续(air 域加固,已完成,随本批入库)**:Dingling 10 seeds 优势增强——60% 丢采样
双目标对 gru_d/lstm/gru/transformer 全部显著(含 transformer,PM2.5 +6.51 / TEMP
+5.20);第二留出站 Gucheng 跨站 NULL(全 arch 收敛,|t|<1)——air WIN 确认为单站
结果。落地叙事收缩为"空气质量类任务实测最优 + 可部署"(见
docs/STREAMING_EDGE_LANDING.md)。JSON `benchmarks/results/airquality_irregular_{10seed,gucheng}.json`。

**§2.10 续记(2026-08-29 夜,自适应衰减探针 mt_lnn_ad —— 方向终局)**:
按既定使命执行了第二次机制探针:LiquidADRegressor
(`battery_soh_edge.py`)把 GRU-D 式可学习衰减收编进液态 bank——
λ = g·exp(−Δt/τ) + (1−g)·MLP([逐 proto 摘要;Δt]),逐 (proto,scale)
sigmoid 门控 init 0.5,其余结构与 mt_lnn_dt 逐项一致(单变量归因),
参数 55,731(gru 的 0.75×,单测强制带内)。判定 R2'/R1' 与四分支映射
预注册于运行前(commit e94160c),裁决 CPU 全程 7 arch × 7 档 × 5 seeds,
0 次 UNSTABLE,六个旧 arch 与 §6 那次逐位复现(同基准)。实测**双负**:
R2'(偏移档)探针 **0.3004** 与固定衰减版 **0.3005 统计打平(t=+0.00)**——
可学习通路没有带来任何东西——仍输 gru_d 0.2485(−1.85)与 gru 0.2722
(−1.78),只赢 lstm(+4.79);R1'(分布内 cv=1)**0/3** 档全败,且 dt0.15/
dt0.4 档比纯结构探针差 2~3×。独立复核(analyze_irregular_line.py 从
JSON 的 mean/std/n 重算)与生产者判定 **MATCH**。按预注册映射:
**与 mt_lnn_dt 双负叠加,衰减方向永久归档,不得再立第三探针**。
两个保留发现:①GRU-D 的偏移鲁棒性无法靠"把可学习 λ 贴到 bank 上"
获得——它来自隐藏态逐单元衰减向 running mean + 门交互的整套机制,
不是单一衰减率;②分布内混合路反而放大方差而非自适应。归档:
RESULTS Null 新行 + BENCHMARKS §8;JSON
`benchmarks/results/synth_ct_control_ad.json`(log 同目录,含 per-seed/
config 回显/git 哈希);落地文档无需改动(capability card 本就按
CLOSED 口径)。**本证据线到此收口,不再有后续探针。**

**§2.10 附记(2026-08-29 深夜,判负根因解剖 + 锚点假设消融判负——机制三层关闭)**:
对双负做了可复现解剖(`benchmarks/diagnose_decay_probes.py`,产物
`benchmarks/results/diagnose_decay_probes.{json,log}`):①mt_lnn_ad 门控
训练后仍≈init(0.496/0.499),两路未分化;②λ_mlp 活着(与 Δt 相关 −0.53)
但率自适应零增益;③gru_d γ(Δt_norm=8)≈0.25→75% 落向 running mean。
据此立"锚点假设"(GRU-D 赢在衰减去向)并跑**廉价因果消融闸门**(10 分钟):
GRU-D 去 hbar(hbar≡0,衰减率/门/输入衰减全保留)偏移档 **0.299 vs 完整版
0.306,Welch t=+0.27 无差**——锚点不是差异化来源,**锚点探针不立**,
省下一个 2.5h 探针周期。至此机制方向**三层关闭**:Δt 特征接线(原始) /
衰减率(mt_lnn_dt→mt_lnn_ad 两次预注册探针)/ 不动点锚点(因果消融)。
GRU-D 偏移优势的剩余解释是门控结构(z/r 与衰减解耦的逐步逐单元更新;
bank 的保留与更新共用同一 λ,blend 对时间恒定)——把它塞进 bank 等于
重造 GRU-D,无独立主张价值。**下一个方向是训练配方而非架构**:全体
架构在未见 gap 尺度下 ~10× 退化(BENCHMARKS §6 遗留开放问题),Δt 混合
训练能否修复之——试点 `benchmarks/dt_mixture_pilot.py`(探索性,非
预注册主张)已发车。附:build_dataset 在稀疏档(dt=0.4)按行数计窗口,
实际产出可少于请求(正式裁决 dt0.4 档 2080/665 即此),对所有 arch
一致,不影响已归档对比。


**§2.10 附记二(2026-08-29 深夜,Δt 混合训练试点 + 前沿调研 → 下一步定档)**:
试点(`benchmarks/dt_mixture_pilot.py`,产物 `dt_mixture_pilot.{json,log}`,
探索性非预注册):窄臂(只训 0.05)vs 混合臂(0.05/0.15/0.4 等比)× 5 arch ×
5 seeds,同一评测。**结果:全体架构偏移档修复 4-8×(mt_lnn 0.452→0.057、
mt_lnn_dt 0.308→0.059、gru_d 0.251→0.071,全部 t≥4.8),且架构差异消失
(mt_lnn 反超 gru_d)**;分布内代价小(mt_lnn_dt 反而改善,gru 有 +0.019
绝对的小代价)。**结论:未见 gap 尺度的 ~10× 退化主要是训练分布问题,
不是架构问题**——这也回头解释了机制线:两次探针比的是"谁在 OOD 下
摔得体面",而可行动的答案是"别 OOD"。注意:试点混合臂含 0.4,属
"训练尺度内插";**真外推(混合外更大 gap,如 1.6)仍未测**。
前沿调研(搜索配额尽,走 arXiv API,2026-08-29):①**TIDES**
(arXiv:2605.09742)诊断与我们的两次负结果精确互证——"输入依赖放在
步长上会失去物理意义"(=ad 探针的 MLP-λ 形式)、"Δ 保持物理但固定率
则无逐 token 表达"(=dt 探针),其方案是把输入依赖移到对角状态矩阵
(λ=exp(−Δt·rate(x)),Δt 物理保留在指数上)并建 Fading Flash OOD-Δ
外推基准;②**CSE**(arXiv:2608.17293)主张评测本身被采样分布偏置
(重要性加权修正);③领域风向对"纯架构故事"趋于怀疑(统计基线打平
深度模型 arXiv:2602.19531 等)。
**下一步定档(按证据)**:(A)配方线升级为预注册正式实验——新增
`benchmarks/synth_mixture_recipe.py`(双臂自含对照+真外推档 test dt=1.6
描述性记录),主张"混合训练修复训练尺度内偏移、分布内代价有界";
(B)TIDES 式第三形态(Δt 物理在指数上的可学习率)是唯一剩余架构假设,
但其开题需显式推翻"不再立探针"的归档决定并引用 TIDES 依据,留待
(A)落地与真外推档结果后再议——若混合训练在 1.6 档仍崩,TIDES 式
才有立项空间。

**§2.10 附记三(2026-08-30,下一步分工定档)**:优先级清单+架构改动规格
+测试工程师执行手册已成文 **docs/EDGE_NEXT_STEPS.md**(P0=mt_lnn_rate
TIDES 式探针[待架构同学落款开题决定]/P1=transformer 连续时间位置编码/
P2=配方实验收尾[在跑]/P3=训练稳定性消融/P4=真实域配方落地);配方实验
中期信号:混合训练修复 E2 且**部分**改善真外推 E3(0.93→0.38,残余 ~10×
缺口)——该残余即 P0 的立项依据。
