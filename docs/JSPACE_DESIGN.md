# J-Space:M1 的全局工作区升级设计
# J-Space: Upgrading M1's Global Workspace Toward Functional Access-Consciousness

> v1.0 · 2026-08-01 · 设计文档(实现分阶段,每阶段独立 ablation)
> 关联:`docs/ROADMAP_M2.md` §3 生物模块改造表 · `mt_lnn/gwtb.py`

## 0. 诚实框架(先说清楚我们声明什么、不声明什么)

本项目 2026-07 已把意识/Φ/麻醉主张从论文降级(证据站不住)。J-Space **不改变这个纪律**:

- ✅ 我们构建并测量:**功能性全局工作区**——竞争、点火、驻留、广播、可报告性,
  全部是可 ablation、可量化的机制(对应 GWT 理论里的 *access consciousness* 功能面)
- ❌ 我们不声明:模型"有意识"、有现象体验(qualia)。任何对外材料一律用
  "GWT-inspired workspace mechanisms",不用 "conscious"
- 判据始终是:**加了这个机制,哪个任务的哪个指标动了多少(≥5 seeds)**

## 1. 现状:GWTB 已有什么(mt_lnn/gwtb.py)

| 机制 | 状态 |
|---|---|
| 压缩瓶颈 832→104(d_gw)| ✅ `GWTBLayer` |
| 点火带宽门控(可调多少通道 fire)| ✅ `_apply_bandwidth_gate` + `set_bandwidth_bias_offset` |
| 竞争性竞标(world model/记忆/top-down 外部 bids)| ✅ `CompetitiveGWTBLayer._compete` |
| 因果工作区自注意力 | ✅ `_run_workspace_pipeline` |
| 广播回全网(gated residual)| ✅ 近恒等启动 |

**缺的是让它从"压缩层"变成"工作台"的三个机制**(下面就是 J-Space)。

## 2. J-Space 三机制(按实现顺序)

### J1 · 工作区驻留(Reverberation)——"思考发生在舞台上"

现状:工作区注意力每次 forward 只过一遍——信息上台就下台,没有"驻留回响"。
GWT 的核心主张恰恰是:内容在工作区**反复回响直到点火**,才成为全局可用的。

- **实现**:`workspace_iterations: int = 1`——工作区自注意力在 104 维内循环 N 次
  (第 k 轮输入 = 第 k-1 轮输出,权重绑定)。d_gw=104,成本约为主干迭代的 1/64,
  是全模型**最便宜的思考深度**。这正是 P0 探索报告的"备选 B"。
- **与 P0 的关系**:P0 证明"只循环液体核心≠思考、组合计算在注意力里"。
  工作区循环 = 在注意力上加深度,且只在瓶颈处加——理论上该有效,待实验裁决。
- **零回归**:默认 1 = 现路径;循环无新参数(可选零门融合)。
- **验收**:深度敏感任务(单环指针追踪)上 workspace_iterations ∈ {1,2,4} 的
  fixed sweep;若曲线上升,"便宜思考"成立。

### J2 · 工作区持存状态(Sustained Content)——"当下意识内容"

现状:工作区内容随 token 流走,没有一个跨 chunk 持存的"当前在想什么"。

- **实现**:`jspace_state: (B, m, d_gw)` 持存槽(m=4~8 个内容槽),类似
  slot-attention:每步工作区赢家**写入/覆盖**槽位(带衰减),槽位内容作为
  下一步竞争的默认竞标者之一。塞进现有 `ModelCacheStruct` 随流携带——
  这让 O(1) 状态里第一次有了**结构化的"注意焦点"**,而不只是液体状态。
- **与产品的关系**:`recurrent_only()` 已支持丢 KV 只留状态;jspace_state
  只加 m×104 个浮点(~2KB),不破坏 O(1) 卖点。
- **验收**:跨窗口记忆基准(现有 0.56 vs 0.00 那条)上,带槽位 vs 不带的对照;
  以及"任务切换后旧目标是否还影响行为"的持存探针。

### J3 · 可报告性(Reportability)——"说得出自己在想什么"

现状:工作区内容是内部激活,外部不可读。GWT 里可报告性是 access 的定义性特征。

- **实现**:一个冻结主干、只训探针的 **report head**:`d_gw → vocab` 的轻量解码器,
  训练它从工作区状态重建"当前最相关的输入片段/当前子目标"。这既是可解释性
  工具(测量工作区实际编码了什么),也是自监督信号(重建不出来 = 工作区没装到东西)。
- **附带产出**:每步可以 log "工作区快照"——demo 里可视化"模型现在在注意什么",
  是官网最直观的差异化展示。
- **验收**:report head 的重建准确率 vs 从随机层探针的基线;快照与任务相关性的人工抽检。

### J4(后置)· Surprise 门控写入——工作区 × 记忆的闭环

ROADMAP §3 已有:预测编码误差大 = surprise = 该写入 episodic memory 的时刻。
J-Space 把它接完整:**只有经过工作区点火的内容才有资格被写入长期记忆**
(海马编码依赖注意/意识状态,这是神经科学里有据的)。依赖 J2 + PC 辅助 loss,排最后。

## 3. 实施顺序与算力

| 阶段 | 工作量 | 算力 | 依赖 |
|---|---|---|---|
| J1 驻留 | 小(gwtb.py 一个循环 + 测试)| 本地 GPU 探针实验 | 无 |
| J2 持存槽 | 中(cache 结构 + slot 写入)| 本地 | J1 |
| J3 report head | 小(独立探针,不动主干)| 本地/CPU | J2 |
| J4 surprise 写入 | 中 | 本地 | J2 + PC loss(ROADMAP 模块)|

全部遵循仓库纪律:零回归默认值、每机制独立 ablation ≥5 seeds、负结果照记入
`ROADMAP_M2.md` §4.5 / `ABLATIONS.md`。

## 4. English Summary

J-Space is the refit of M1's existing GWT bottleneck (GWTB) from a compression
layer into a functional workspace with four measurable mechanisms: (J1)
**reverberation** — weight-tied iteration of the 104-d workspace attention, the
cheapest possible "thinking depth" (P0 showed composition lives in attention;
this adds attention depth only at the bottleneck); (J2) **sustained content
slots** carried in the O(1) state (~2 KB) — a structured "current focus" across
chunks; (J3) **reportability** — a frozen-backbone probe decoding what the
workspace actually holds, doubling as an interpretability tool and demo asset;
(J4) **surprise-gated episodic writes** admitted only for workspace-ignited
content. We claim functional access-consciousness mechanisms with ablation
evidence — never phenomenal consciousness. Every mechanism ships with a
zero-regression default and a ≥5-seed ablation gate.
