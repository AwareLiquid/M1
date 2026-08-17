# AwareLiquid Architecture (v2.0 — MT-LNN Next)

**Last updated:** 2026-06-06
**Status:** Active development — v2.0 module expansion in progress
**Companion docs:** [PRD.md](PRD.md) · [AWARENESS_NETWORK_PRD.md](docs/specs/AWARENESS_NETWORK_PRD.md) · [AWARELIQUID_SYSTEM_MVP.md](docs/specs/AWARELIQUID_SYSTEM_MVP.md)

---

## 1. Design Thesis

巨头在云端拼参数广度。我们在**推理体验的三个维度**做差异化:

| 维度 | Gemini 3.1 | AwareLiquid |
|---|---|---|
| 思考状态 | 每次 query 后蒸发 | 跨 session 持续演化 (capsule) |
| 计算分配 | 一刀切 thinking budget | 按语义熵 + Φ 信号动态路由 |
| 推理可信度 | 黑盒摘要 | 可回放、可归因的 Φ-trace |

知识广度不打，**用云端 API 当事实硬盘**，核心推理与状态留在端侧。

v2.0 在此基础上增加四个正交模块，每个都有独立开关，默认关闭，零回归风险。

---

## 2. 完整模块地图 (v2.0)

```
mt_lnn/                                       STATUS        TEST FILE
├── config.py              MTLNNConfig         ✅ 已实现
├── embedding.py           RoPE + TokenEmbed   ✅ 已实现
├── mt_attention.py        GQA 极性注意力       ✅ 已实现
├── mt_lnn_layer.py        13原丝 LTC + 侧向耦合 ✅ 已实现      test_model.py
│   ├── (内含) κ-gate      动态τ尺度门           ✅ 已实现
│   └── (内含) _hebb_signal Hebbian 信号采集     ✅ Phase D
├── rhythm.py              LAVI节律门控          ✅ 2026-06-06  test_rhythm.py
│   ├── LAVIEstimator      per-protofilament 节律检测  (13测试)
│   └── GlobalRhythmController 跨层节律校正
├── gwtb.py                GWT 压缩→SA→广播     ✅ 已实现
│   └── CompetitiveGWTBLayer 多源竞争广播       ✅ Phase A     test_gwt_competition.py
├── global_coherence.py    sparse top-k Orch-OR  ✅ 已实现
├── causality.py           因果一致性检测         ✅ Phase B     test_causality.py
│   └── principal_subspace()  暴露合法因果子空间基 (供 steerer 复用, 不重复 SVD)
├── causal_steering.py      因果激活转向 (STARS)    ✅ 已实现      test_causal_steering.py
│   └── CausalActivationSteerer  检测到断裂时把隐态正交投影回合法子空间 (0参数)
├── causal_decoding.py      L3 转向接入真实解码     ✅ 已实现      test_causal_decoding.py
│   └── CausalDecodeSteerer  generate() 的 step_callback: 在线纠正循环态 cache (0参数)
├── world_model.py         预测隐态头             ✅ Phase C     test_world_model.py
├── imagination.py         L4 潜空间多步想象 rollout ✅ 已实现    test_imagination.py
│   └── LatentImagination   把世界模型单步映射自回归滚 k 步成"想象轨迹" (0参数, 不耦合 backbone)
├── spatial_ops.py         可组合几何算子          ✅ 已实现      test_spatial_ops.py
│   └── 距离/方向/包含/邻近图/连通分量/可达性  纯函数, 0参数, 对解析真值可单测, 不耦合 backbone
├── physics_ops.py         可组合牛顿动力学算子    ✅ 已实现      test_physics_ops.py
│   └── 辛积分(半隐式 Euler + 2阶时间可逆速度 Verlet[integrate_verlet, 经 rollout(integrator="verlet") 选用, 长程 imagination rollout 能量漂移 O(dt²) 有界])/引力场/N体引力/碰撞冲量/盒壁反弹/守恒诊断/rollout  纯函数, 0参数, 复合即"脑内推演"
├── salience_events.py     全局工作空间点火事件    ✅ 已实现      test_salience_events.py
│   └── SalienceEventDetector  自适应基线+z分数+迟滞+不应期, 只读观察者, 0参数, 双速引擎触发接口
├── failsafe.py            断流盲推 + 输出断路器    ✅ 已实现      test_failsafe.py
│   └── BlindRolloutGuard 置信度门控盲推(借世界模型imagination盲滚) + CircuitBreaker 模型外硬断路器 + TopologyBreaker 表示"形状"触发器(锁定健康参考点云, 每拍测 SRTD H0条形码漂移+可选 Betti-0 分量数, 经与 CircuitBreaker 同款去抖 FSM 跳闸/复位; 复合 topology_ops, 0参数, 不耦合 model.py)
├── acoustic_ops.py        可组合声学/双耳听觉算子  ✅ 已实现      test_acoustic_ops.py
│   └── 传播延迟/球面扩散/ITD/ILD/多普勒/相位叠加干涉/方位反演定位+双耳场景, 纯函数 0参数, 复合即听觉空间推理
├── geometry_ops.py        可组合 Fisher-Rao 信息几何算子 ✅ 已实现  test_geometry_ops.py
│   └── 概率单纯形(PlaceCellCode 的 softmax 真正所在空间)上的 Bhattacharyya/Fisher-Rao 距离/测地线插值/exp-log 映射/平行移动/Fisher 度量/Karcher 重心, 纯函数 0参数, 不耦合 backbone; L2 当前用错误的欧氏度量读取 place codes, 这里给出正确的弯曲流形标尺(P0#1)
├── topology_ops.py        可组合拓扑数据分析(TDA)算子 ✅ 已实现   test_topology_ops.py
│   └── 状态/编码点云的最小生成树/H0 持续同调(条形码=排序后的 MST 边权, 精确)/Betti-0 + Betti-0 曲线(复用 spatial_ops 并查集连通分量)/总持续度/SRTD 对称相对拓扑散度(0参数拓扑漂移触发器), 纯函数 0参数, 不耦合 backbone; 数清吸引子/簇结构并捕捉拓扑突变(P0#2)
├── stdp_ops.py            可组合脉冲时序依赖可塑性(STDP)算子 ✅ 已实现  test_stdp_ops.py
│   └── 非对称指数 STDP 窗(stdp_kernel/window_integral)+全对全成对求和(pairwise_stdp, 定义式)+O(T) 在线资格迹更新(stdp_trace_update, 硬件实跑方式), 两者可证相等; 纯函数 0参数, 不耦合 backbone; 仅凭脉冲时序的局部、无反向传播学习, 与 plasticity.py 并行——损失级 Hebbian 管平滑连续时间 LTC 核, 事件驱动 STDP 管离散的 salience 点火/L2 place-code 事件流(plasticity.py 明言 STDP 不适用于连续核)(P0 学习)
├── attractor_ops.py       可组合吸引子/自稳定算子  ✅ 已实现       test_attractor_ops.py
│   └── 线性映射的不动点/谱半径/渐近收敛率/是否压缩(闭式)+任意步进映射的 relax 滚动+从观测轨迹读出沉降时间/经验收敛率/Lyapunov 能量下降+basin_radius(二分探针测吸引域半宽); analyze_linear_attractor 复合并交叉验证经验值==解析值; 纯函数 0参数, 不耦合 backbone; 量化沉降核收敛到何处、多快、吸引域能吸收多大扰动(P0 学习)
├── ingest_ops.py          传感器摄入/流对齐算子    ✅ 已实现      test_ingest_ops.py
│   └── 把抖动/带时间戳的非均匀采样重采样到固定dt栅格(线性/ZOH)+覆盖掩码标出长空洞→交盲推滑行, 纯函数 0参数, 输入侧前端
├── slow_layer.py          双速引擎的慢半边(点火时才唤醒) ✅ 已实现  test_slow_layer.py
│   └── SlowThreatAssessor  多步弹道前瞻(rollout+in_ball)→突破ETA/最近接近/CLEAR-WATCH-ENGAGE威胁等级, 0参数, 仅点火时付费
├── pipeline.py            双速哨兵编排(把各层串成一个环) ✅ 已实现  test_pipeline.py
│   └── DualSpeedSentry  感知(声学+空间)→预测(物理惊讶)→显著度点火→唤醒慢层多步评估→盲推续命→断路器限幅, 编排器 0新参数, 零核耦合
├── plasticity.py          Hebbian 正则           ✅ Phase D     test_plasticity.py
├── deliberation.py        熵三级路由 + causal hook ✅ 已实现    test_deliberation.py
├── thinking.py            自我思考 serve 路径     ✅ 已实现      test_thinking.py
│   ├── generate_with_thinking  路由驱动逐 token 生成 (模型无关)
│   ├── self_consistency_vote   SELF_CRITIQUE 重解码 (token 级自洽投票)
│   └── ThinkingTrace / render_*  内存思考轨迹 + 前端渲染
├── model.py               MTLNNModel (A-D集成)  ✅ 已实现      test_model.py
├── llama_adapter.py       HF 模型 MT 残差适配器  ✅ 已实现      test_llama_adapter.py
├── anesthesia.py          AVP 麻醉验证           ✅ 已实现
├── phi_hat.py             Φ̂ kNN 估算            ✅ 已实现
├── phi_iit.py             IIT 4.0 Φ (可选)      ✅ 已实现
├── memory.py              SQLite 持久 h_prev     ✅ 已实现      test_memory.py
├── capsule.py             信念状态 + 证据日志    ✅ 已实现      test_capsule_v2.py
├── streaming.py           state-only O(1) 推理   ✅ 已实现
├── recipes.py             Phase 5b 一键接入      ✅ 已实现
├── cloud_client.py        云端 Oracle 客户端     ✅ 已实现
├── observability.py       JSONL 指标写入         ✅ 已实现
├── reasoning_trace.py     推理时间线             ✅ 已实现
├── parallel_scan.py       pscan (Blelloch)       ✅ 已实现      test_parallel_scan.py
├── multimodal.py          视觉/任意模态前端       ✅ 已实现      test_multimodal.py
│   ├── VisionPatchEmbed   ViT patch → d_model token
│   ├── ModalityProjector  任意特征 → d_model token
│   └── CLIPModalityEncoder 冻结 CLIP → d_model token  test_multimodal_clip.py
├── sensory_frontend.py    原始流式传感器前端 (P1闭环) ✅ 已实现   test_sensory_frontend.py
│   ├── SensoryFrontend    抖动/丢帧原始流 → 固定dt栅格token (复合 ingest_ops.align_stream + ModalityProjector, 可训练 nn.Module, 不 import model.py)
│   └── SensoryEncoding    inputs_embeds + 覆盖/pad 掩码 (丢帧步标记, 交 BlindRolloutGuard 滑行)
├── spatial.py             空间计算前端 (栅格细胞)  ✅ 已实现      test_spatial.py
│   ├── GridCellEncoding   六边形/Fourier 位置码 (0参数, 固定; 输入侧)
│   ├── PlaceCellCode      位置细胞群体码 (高斯/DoG, 0参数; 监督靶/输出侧)
│   ├── SpatialCoordEncoder 连续坐标(+特征) → d_model token
│   ├── PointCloudEncoder  PointNet 式点云 → d_model token
│   └── VoxelPatchEmbed    Conv3d 体素patch → d_model token
├── spatial_reasoning.py   空间思考 (感知+审议)     ✅ 已实现      test_spatial_reasoning.py
│   ├── SpatialReasoner     SpatialCoordEncoder + backbone + 路由
│   └── SpatialThinkingResult  逐空间位置 ThinkingTrace + uncertain_positions()
├── spatial_memory.py      L2 空间序列记忆 (0参数)  ✅ 已实现      test_spatial_memory.py
│   └── SpatialMemory       位置索引联想记忆 (Hebbian 外积写 / 模式补全读, 复用 PlaceCellCode)
└── quantum_coupling.py    量子耦合 (可选)        ✅ 已实现
```

**模态前端契约 (multimodal.py / spatial.py / sensory_frontend.py)**: 所有前端统一产出 `(B, N, d_model)`
token,经 `fuse()` 与文本 token 拼接后由 `MTLNNModel.forward(inputs_embeds=...)`
进入 backbone。**这些模块从不 import model.py,与核心零耦合** —— 训练时与模型一同
放进 optimizer 即可。`spatial.py` 的 `GridCellEncoding` 是受内嗅皮层栅格细胞启发的
固定(0 参数)多尺度周期位置码,与 `grid-cell-emergence` 实验同源。

**实验日志 — grid-cell emergence (2026-06, 本地 CPU, 6k steps, n_place=512)**:
路径积分任务下对比 GRU 与 MT-LNN 是否自发涌现栅格细胞。关键发现:place-cell 调谐
决定一切——纯高斯靶 (`run_full`) 两个模型 grid_score_max 均为负 (GRU −0.038 /
MT-LNN −0.064),无栅格;改用 difference-of-Gaussians (Mexican-hat) 靶并加锐化温度
(`run_dog2`,`GC_PLACE_DOG=1 GC_DOG_TEMP=0.05`) 后 **GRU grid_score_max 升至 +0.284**
(接近栅格),MT-LNN 仍为 −0.072。复现了 Sorscher et al. 2019 的核心结论(DoG 是
触发条件);MT-LNN 在此极简设置 (6k step / CPU) 下未涌现栅格,留待更长训练验证。

**空间优化 (2026-06):把 DoG 触发条件提升为一等能力 (`PlaceCellCode`)**。此前
"place-cell 靶" 这一决定栅格涌现的关键杠杆只埋在一次性的 `grid_cell_emergence`
实验脚本里(未测、不可复用)。现把它抽成 `spatial.py` 的 `PlaceCellCode` 模块 ——
`(B,…,coord_dim) → (B,…,n_place)` 的 softmax 监督靶,`mode="gaussian"|"dog"`。它是
`GridCellEncoding`(输入侧:坐标→token)的**对偶**(输出侧:坐标→路径积分监督靶)。
关键校验内置:`dog_amp ∈ (0,1)` 强制为开区间 —— `amp≥1` 会把中心峰压平→均匀靶→loss
卡在 `ln N`(正是首次 DoG sweep 失效的 bug)。0 参数、不 import backbone(零耦合),
6 项单测固定契约(行和为 1、DoG≠高斯、非法参数报错、种子可复现)。默认仍用已验证的
高斯路径(见 `kaggle_kernels/grid_cell_emergence/`)。

**自我思考 serve 路径 (thinking.py)**: 把 `deliberation.py` 的*策略*（LOCAL /
SELF_CRITIQUE / CLOUD 三级路由）变成可在 demo 中逐 token 运行的*机制*。每步用
router 判定路由：低熵直接本地解码；不确定的 token 通过 token 级自洽投票
(`self_consistency_vote`) *重新斟酌*（这正是 `deliberation.py` 留下的
"Future: N-sample re-decode" 钩子）；需外部事实的 token 标记为 cloud。**仅 import
torch + deliberation,不碰 model.py** —— 适配 Qwen/Llama/MT-LNN `serve.pt`。在线
交互轨迹放内存 (`ThinkingTrace`),与 `reasoning_trace.py` 的离线 JSONL 持久化分工
明确。`app.py` 新增 "🧠 Self-Thinking" tab,可选导入失败则该 tab 自动隐藏（优雅降级）。

**空间思考 (spatial_reasoning.py)**: 把上面两块*收敛*成一个能力 —— 用
`SpatialCoordEncoder` 感知场景 (perception),再让 `deliberation.py` 的路由对
backbone *逐空间位置*的预测做审议 (deliberation)。`SpatialReasoner.reason()` 返回
一个 `ThinkingTrace`,其每一步对应一个空间 token:router 判 LOCAL（自信）/
SELF_CRITIQUE（重新斟酌）/ CLOUD（需外部事实），不确定位置触发自洽投票。
`uncertain_positions()` 即"模型在场景里何处停下来思考"的地图——这就是"空间思考"。
**只 import 公开 API**（`spatial` 编码器 / `multimodal.fuse` / `deliberation` 路由 /
`thinking` 轨迹）,通过 `forward(inputs_embeds=...) → out["logits"]` 契约触碰 backbone,
**无新增耦合**;它是 `nn.Module`,编码器可与模型联合训练,而 `reason()` 在 no_grad 下运行。
`SpatialReasoner` 新增**可选** `checker` / `steerer` 两个入参(默认 `None`,完全向后兼容):
传入后会把 backbone 逐位置的"信念轨迹"喂给 `CausalConsistencyChecker`,得到的一致性
分数作为 `consistency_signal` 交给 router —— 即使某位置 token 熵低看似自信,只要轨迹
发生突跳也会被标记去审议;`steerer` 则把该位置偏离合法因果子空间的程度作为诊断写入轨迹。
另新增**可选** `memory` 入参(L2 `SpatialMemory`,默认 `None`):`remember(coords)` 把感知到的
场景显式写入位置索引记忆,`reason()` 则额外报告每个空间位置的**记忆熟悉度**(当前感知与该位置
召回内容的 cosine,高值="这地方来过")并写进轨迹 note。写入是**显式调用方动作**——`reason()`
保持只读、绝不改动地图,故 reasoner 不会静默拥有长期记忆状态。0 参数、不 import model.py、完全向后兼容。

**因果激活转向 (causal_steering.py, STARS 启发)**: `CausalConsistencyChecker` 只*检测*
LNN 隐态轨迹的突跳,`CausalActivationSteerer` 则*纠正*它 —— 当检测到断裂(一致性分数
低于 `floor`)时,把漂移的隐态**正交投影**回"合法因果子空间"(近期轨迹真正所处的方向)。
几何上严格对应 STARS("Inference-time Stiefel Activation Steering", ICLR 2026):子空间
基 `Vk` 是正交行(Stiefel 流形上一点),转向即沿该子空间的正交投影,只移除落在合法子空间
*之外*的"非法"新颖分量,绝不引入任意新方向。**纯 Python、0 参数、不 import model.py**;
不重复造 SVD —— 通过 `CausalConsistencyChecker.principal_subspace()` 公共方法复用同一个
子空间,确保"检测器"与"执行器"对"何为一致"的定义永远一致。可选 `adaptive` 增益:断裂越深
拉回越强。全程 opt-in,不接线则行为完全不变。

**L3 闭环:转向真正接入解码 (causal_decoding.py)**: 上面的检测器/转向器此前只在
`SpatialReasoner` 里做**只读诊断**、或在 demo 的独立回路里跑 —— 纠正后的状态从未回灌进*正在
生成*的模型。`CausalDecodeSteerer` 闭合这一环:它是 `MTLNNModel.generate()` 的一个
`step_callback`,每步读出各层循环态 cache `cache.layers[i][1]` `(B,P,S,D)`,喂给**逐层独立**的
checker,检测到断裂就让 steerer 把它正交投影回合法子空间,并**写回 cache** —— 修正条件化*下一个*
token。执行器终于作用在它所监控的模型上。为此给 `generate()` 加了一个**通用** `step_callback(cache,
step)` 钩子(model.py 不 import 任何因果模块,保持解耦;钩子亦可用于日志/KV 驱逐等,非单用途死代码);
转向逻辑全在 `causal_decoding.py`,只 import `causality`+`causal_steering`,**绝不碰 model.py、0 参数**。
关键修复 `principal_subspace(exclude_last=True)`:`update(h)` 已把可疑状态 `h` 追加进历史,若构造合法
子空间时不剔除它,断裂方向会成为它*自己*子空间的主轴 → off-subspace 残差塌成 ~0 → 转向几乎不纠正
(实测 187× 差距,正是把皮毛变成真起作用的那一刀)。**安全性**:平滑(健康)轨迹永不触发门控,
纠正为 no-op、输出逐位不变 —— 这是门控式纠正,只在检测到断裂时出手;注入一次 off-subspace 扰动后,
转向把被污染的循环轨迹**实测拉回**基线(漂移下降)。token 级幻觉收益需在训练好的 `serve.pt` 上度量。
由 `tests/test_causal_decoding.py` 固定其契约(null 回调逐位一致、cache 突变流入下一步 logits、健康
run 零纠正且不变、注入断裂触发并降漂移、exclude_last 决定纠正幅度、层选择/reset、0 参数)。

**L3 解码闭环转向 demo (`examples/demo_causal_decoding.py`)**:这是 L3 的**生产路径验证** ——
区别于下面那个在模型*外*跑合成 2D 轨迹的 demo,本 demo 在真实 `MTLNNModel.generate()` 里、对*在线
循环态 cache* 动手。智能体解码时其"信念"就是 cache 跨 token 传递的 LNN 循环态;一次自信幻觉 = 该状态
的突跳。在解码中途往 cache 注入一次 off-subspace 扰动模拟这一跳,对比三条 rollout:基线(不注入)/
仅注入(突跳沿递归传播、belief 持续偏离基线)/ 注入+转向(`CausalDecodeSteerer` 作为 `step_callback`
检测断裂并把状态投影回合法子空间)。实测跨 5 seed:转向把注入后的循环轨迹漂移**降约 11–14%**;
且**安全性**——干净 run 永不触发、token 逐位不变。诚实边界:未训练模型 token 退化,故在*循环态轨迹*
层面度量;token 级幻觉收益待训练好的 `serve.pt` 上用同一套指标度量(换 checkpoint 即可)。仅用公开
API(`generate(step_callback=)`+`causal_decoding`),确定性、CPU 秒级、`--plot`。由
`tests/test_demo_causal_decoding.py` 10 项测试固定其行为契约。

**L3 因果空间转向 demo 原型 (`examples/demo_causal_spatial_steering.py`)**:把上述检测+
转向放进一个**递归轨迹回路**里跑通端到端(检测器/执行器本身只*诊断*,真正的纠正回灌属于
"拥有状态"的循环,故放在 demo 而非 `spatial_reasoning.py` 一次性路径里,保持零 model.py
耦合)。智能体在 2D 竞技场沿平滑路径移动,信念态落在一个 2 维合法流形(=合法因果子空间)上;
中途注入一次"瞬移"幻觉把信念态推出流形之外,在递归态里**持续传播**。对比两条 rollout:
**无转向** → 离流形漂移永久滞留 (off-manifold ≈ 4.2),检测器分数却自愈回升(说明"只检测
不够");**有转向** → 在瞬移那一步检测到断裂、正交投影掉非法分量并回灌,**off-manifold 塌回
≈ 0.07–0.22 并保持**。仅依赖 `causality`+`causal_steering` 公开 API,确定性可复现,CPU 秒级,
`--plot` 出对比图。由 `tests/test_demo_causal_spatial_steering.py` 10 项测试固定其行为契约。

**L2 空间序列记忆 (`mt_lnn/spatial_memory.py`)**:Bicanski & Burgess (2021)
"位置索引联想记忆"的计算核心 —— 智能体沿轨迹移动时把"此处所见"的内容 embedding 写进一块
关联记忆,之后给一个(哪怕近似/带噪的)位置就能**模式补全**召回内容。`SpatialMemory` 复用 L1
的 `PlaceCellCode` 把连续坐标转成软位置细胞 key:写入是 Hebbian 外积累加 `M += key⊗content`,
读出是线性联想读 `key@M`(按 key 质量归一)即吸引子式补全。**0 可训练参数**(纯 buffer+张量代数,
CPU 可跑),**不 import model.py**;读出形状 `(B,N,d_model)` 可直接接 `multimodal.fuse`,记忆变
成多模态里的一路 token。`decay<1` 支持遗忘(近期偏置)。持久化委托给 `memory.SessionMemory`
(本模块只 `state_blob`/`load_blob` 暴露 `[M, k_mass]`,不碰 SQLite,分类清晰)。由
`tests/test_spatial_memory.py` 11 项测试固定其契约(0 参数、写后即读、空间局部性、无串扰、
模式补全、遗忘、fuse 形状、SessionMemory 往返、共享 PlaceCellCode、确定性、reset)。

**L2 空间序列记忆 demo 原型 (`examples/demo_spatial_memory.py`)**:把上述写/读放进一个
端到端故事 —— 智能体沿椭圆路径在各地标处写入内容,事后用**带噪位置**(GPS 漂移)查询。结果:
on-landmark 召回近乎完美(cosine≈1.0),随查询噪声**优雅退化**(1.0→0.99→0.74→0.47,而非
崩溃),且**寻址果断**(正确内容比最像的错误内容高约 0.79 cosine,是检索而非模糊平均)。仅依赖
`spatial_memory` 公开 API,确定性可复现,CPU 秒级,`--plot` 出抗噪曲线。由
`tests/test_demo_spatial_memory.py` 12 项测试固定其行为契约。

**L4 潜空间多步想象 rollout (`mt_lnn/imagination.py`)**:世界模型 `PredictiveStateHead` 只学到
**单步**潜态转移;`LatentImagination` 把这个**已训练**的单步映射在归一化潜空间里自回归向前滚
`horizon` 步,得到一条"想象轨迹"——脑内"先在心里把场景推演几步再决策"的最小可信内核。每步附带
**置信度**(从 head 当前的 `1-last_pred_error` 起,随 horizon 与想象新异度衰减)与**新异度**
诊断,供调用方门控(如置信跌破阈值即放弃该计划)。**关键性质——复合**:在一个学过真实(旋转)动态
的 head 上,多步想象对**真实 k 步未来**的追踪显著优于"假设什么都不变"的静态基线(demo:horizon 5
时想象 cos≈0.80 vs 静态 -0.69),证明 rollout 是把学到的单步映射真正**复合**起来,而非包装。
**零新增参数**(纯推理期执行器,`n_parameters==0`)、**不耦合 backbone**(仅 import torch,绝不
import `model.py`,鸭子类型挂在 head 上),默认不实例化。`tests/test_imagination.py` 16 项测试固定
其契约。

**L4 想象 rollout demo (`examples/demo_imagination.py`)**:两段式——Part 1 把驱动挂到一个真实
`MTLNNModel`(world model 开启)的 `final_norm` 隐态上,证明零参数、零耦合的真模型接入面与置信度
衰减;Part 2 在旋转世界上训练 head,给出 `imagined_cos > static_cos` 的量化裁决。ASCII 输出、CPU
秒级、`--seed` 确定性。`tests/test_demo_imagination.py` 7 项测试固定其行为契约。

**可组合几何算子 (`mt_lnn/spatial_ops.py`)**:`spatial.py` 让 backbone **感知**几何(把坐标/点云/
体素变成 token),但感知只是**描述**场景,不能**计算**几何。这一层补上缺失的"对空间计算"——一组
**纯函数、零参数、合适处可微**的算子:距离矩阵 / 相对方向单位向量 / 平面方位角、盒/球包含判定、
半径图与 kNN 邻近图、连通分量(标签传播)、以及 **可达性**(在邻近图上做批量传递闭包 / BFS 跳数)。
要点是**复合**:`pairwise_distance → radius_graph(stride) → reachable_from(start)` 串起来就是一个
真正的空间推理查询("给定步长能否跨过这道缝到达目标?")——答案是从几何**算**出来的,不是记下来的
(正面回应"区分记忆 vs 真正推理")。纯函数、不是 `nn.Module`、仅 import torch、绝不 import
`model.py`,零 backbone 耦合。每个算子都对解析真值单测,`tests/test_spatial_ops.py` 19 项。

**可组合几何算子 demo (`examples/demo_spatial_ops.py`)**:智能体站在被一道缝隔开的踏脚石阵上,每步
最多跨 `--reach` 米——能否到达目标?用三个零参数算子复合求解并渲染成 ASCII 地图;再扫步长,显示步长
够大时对岸"解锁"。短步长(1.1)不可达、长步长(2.1)三跳到达、解锁阈值=2.0,均确定性。
`tests/test_demo_spatial_ops.py` 3 项测试固定其契约。

**可组合牛顿动力学算子 (`mt_lnn/physics_ops.py`)**:`spatial_ops.py` 让智能体**对几何计算**,但
场景是会演化的——"在脑海中模拟事件进展"需要一层**对物理动态计算**的能力:在力与接触下把
`(位置, 速度)` 状态向前推进。这一层补上:半隐式(辛)欧拉积分(能量稳定)、均匀引力场与软化 N 体引力
(精确守恒动量)、球-球碰撞检测 + 冲量响应(带恢复系数,`e=1` 守恒动能、`e<1` 耗散)、轴对齐盒壁反弹、
动能/动量守恒诊断,以及把它们复合成轨迹的 **rollout**(智能体内部的"接下来会怎样"推演)。要点同样是
**复合**:旗舰查询"弹跳的球能否越过墙?"是把 `uniform_gravity → integrate → reflect_in_box(地面)`
向前**滚**出来算的——抬高恢复系数,同一份代码报出不同结局(涌现、计算,而非查表)。诚实边界:这是
**确定性物理算子层(引擎)**,不是学出来的动力学模型;一阶辛积分能量稳定但不精确,测试只对它真正守恒
的解析不变量(恒定动量、恒加速下速度精确、弹性正碰速度互换、两体轨道半径近恒定)校验。纯函数、不是
`nn.Module`、仅 import torch、绝不 import `model.py`,零 backbone 耦合。`tests/test_physics_ops.py`
25 项,每个算子对解析真值单测。

**可组合物理算子 demo (`examples/demo_physics_ops.py`)**:一个球被抛出,前方立着一道墙,中间是一块完全
硬的地面——球会越过墙吗?用三个零参数算子复合滚出轨迹并渲染成 ASCII 侧视图;再扫地面恢复系数,看判决
翻转:有损弹跳(低 `e`)削弱第二段弧而撞墙、弹性弹跳(高 `e`)保住高度而越过。弹性地面越过(过墙高 0.64)、
恢复系数 0.4 撞墙(过墙高 0.00),均确定性。`tests/test_demo_physics_ops.py` 3 项测试固定其契约。

**全局工作空间点火事件 (`mt_lnn/salience_events.py`)**:快速液态核每拍都在跑,慢速/符号/云层不应该
轮询这个热环——它该只在**显著状态变更**时被唤醒。这一层就是那根触发线:一个零参数、不耦合 backbone
的**只读观察者**,把逐拍的显著度信号(最自然的是世界模型的预测误差 `last_pred_error`,即"惊讶")变成
离散的状态变更事件。脑机制是字面的,不是装饰:**预测编码**(以惊讶为显著度)、**适应/稳态**(EMA 基线只在
平静期更新,显著度按 z 分数相对"近期常态"度量——已适应的缓慢漂移不触发、突变触发)、**全局工作空间点火**
(z 越过 ignition 阈值才"赢得工作空间"发事件,事件携带显著度快照作广播载荷)、**Schmitt 迟滞**(较低的
release 阈值结束点火态,去抖)、**不应期**(事件后短暂静默,防事件风暴)。持续性 regime change 会被缓慢
重新基线化,使探测器 quiesce 并**重新武装**去捕捉下一次变化,而非永久 latch。是双速引擎现在缺的触发接口:
慢速层/云 LLM 据此订阅,而不污染 µs 级快环。`world_model_surprise(head)` 鸭子类型读 `last_pred_error`,
不 import `model.py`。`tests/test_salience_events.py` 16 项,对确定性合成流 + 真实预测头解析校验。

**点火事件 demo (`examples/demo_salience_events.py`)**:一条惊讶度流——长段平静、一段被适应的缓慢漂移、
然后一次 regime change 跳变、再 settle 到新常态。探测器渲染成 ASCII 时间线,标出点火(^)与 quiesce(v):
缓慢漂移不触发、tick 41 的真正突变触发(z=27.2)、settle 后 quiesce 并重新武装,均确定性。
`tests/test_demo_salience_events.py` 4 项测试固定其契约。

**断流盲推 + 输出断路器 (`mt_lnn/failsafe.py`)**:快速液态核落地野外时会遇到训练目标从未见过的两类故障,
这一层是对应的两个**零参数、不耦合 backbone、模型外**的安全反射。**`BlindRolloutGuard`(输入断流盲推)**:
当输入流卡顿(丢帧/丢包/feed 断流),与其给核喂一个陈旧/零帧悄悄污染其循环态,不如**借世界模型自己的
imagination 从最后一个好状态盲滚**——但只在 imagination 自身的**置信度**高于 floor 时续命,且盲推步数受预算
(默认 = `LatentImagination.horizon`)上限约束;一旦梦不再可信就**转 DARK**("我看不见了",`latent=None`)
而非永远幻觉下去。这是大脑在短暂遮挡期靠前向模型滑行、并知道何时放弃的字面对应。鸭子类型于
`LatentImagination`,`no_grad`。**`CircuitBreaker`(模型外输出断路器)**:作为不可绕过的最后一公里保证,
对每拍命令**无条件硬钳位**(NaN/Inf 洗刷 → 绝对边界 → 斜率限幅),并在原始命令**持续**越红线时带去抖地
**trip 到 fallback**(默认 hold-last-safe,或调用方提供的 PID 之类回退),恢复后再以**无扰切换**(斜率限幅
封住重连跳变)闭合——一个 Schmitt 式保护反射。这条安全保证无法活在学习图内部,必须是模型外的外挂闸。
两者皆 `n_parameters == 0`,不 import `model.py`。`tests/test_failsafe.py` 29 项,对解析/确定性流 + 真实
`LatentImagination`/`PredictiveStateHead` 桥接校验。

**断流盲推/断路器 demo (`examples/demo_failsafe.py`)**:两幕。幕一输入断流——3 拍 live、一段长 dropout、
再恢复:守卫先借真实世界模型 imagination 盲滚(置信度逐拍衰减),trust 跌破 floor / 超预算后转 DARK,feed 回来
再次盲滚,ASCII 渲染 feed/served/conf 三行。幕二输出越线——干净斜坡、NaN、越界猛冲、斜率尖峰、恢复:断路器
把每个发出值都钳进 [-1,1] 且有限,持续越线时 trip 到 setpoint 回退、恢复后无扰闭合,逐拍打印 raw/safe/trip/reason。
均确定性、纯 ASCII。`tests/test_demo_failsafe.py` 8 项测试固定其契约。

**可组合声学/双耳听觉算子 (`mt_lnn/acoustic_ops.py`)**:`spatial_ops` 算几何、`physics_ops` 推动力学,
听觉是 embodied agent 给空间定位的第三条路——大脑不存"声音在我左边",它从两耳收到信号的物理差异里**算**出
方向。这一层补上这个缺口:一组**可组合、零参数、解析**的算子,把声场几何变成大脑的听觉空间线索,并(旗舰)
把线索**反演**回朝向。脑机制是字面的:**飞行时间**(`distance/c`,一切的基底)、**球面扩散**(点源声压
按 `1/r` 衰减)、**ITD 双耳时间差**(低频主定位线索,medial superior olive 的 Jeffress 符合检测读出)、
**ILD 双耳声级差**(高频线索, dB)、**多普勒**(径向相对运动压缩/拉伸波前——"来还是去")、**波前叠加**(多
相干源以相量求和,路径差产生相长/相消干涉)、**定位反演**(`localize_azimuth` 把 ITD 反解成方位,即"算而非
记"那一步)。ITD→方位用远场平面波模型 `ITD=(head_width/c)sinθ`(远场精确、近头近似,逐函数标注)。纯函数、
0参数、不 import `model.py`;`(...,D)` 广播,整条源轨迹 `(T,D)` 一次流过。旗舰 `binaural_scene` 把各算子
在一条轨迹上复合,逐拍给出"它在哪(方位)+ 来还是去(多普勒)"。`tests/test_acoustic_ops.py` 36 项,对
闭式距离/相量/远场 ITD↔方位往返解析校验。

**声学算子 demo (`examples/demo_acoustic_ops.py`)**:两幕。幕一无人机掠过双耳头(左→右):仅凭 ITD 定位的
方位从左扫到右、过最近点时穿过正前方,多普勒音高在最近点穿过静止频率("它在哪+来还是去"),ASCII 渲染
方位箭头与音高升降。幕二两只相干扬声器+滑动麦克风:波前叠加产生交替的相长(响)/相消(静)干涉带。均确定性、
纯 ASCII。`tests/test_demo_acoustic_ops.py` 7 项测试固定其契约。

**可组合 Fisher-Rao 信息几何算子 (`mt_lnn/geometry_ops.py`)**:这一层修的是一个**真实存在**的度量不一致(P0#1)。
`PlaceCellCode.forward` 末尾是 `torch.softmax(...)`——每个 place code 都是一个**概率分布**,是单纯形 Δ^{n-1} 上的点,
而不是 Rⁿ 里的自由向量。可 L2 空间记忆 (`spatial_memory.py`) 却用纯欧氏代数 (`key @ M`) 读写这些 code:欧氏直线
**穿过**单纯形内部、用错误的标尺量距离。坐标无关的正确标尺是 **Fisher 信息度量**,在它之下单纯形是一块弯曲流形——
等距于半径 2 的球面正卦限(`φ(p)=√p` 把 Fisher 度量拉回 4 倍的圆度量)。于是一切都成了球面上的初等几何:
**Bhattacharyya 重叠** `BC(p,q)=Σ√(pᵢqᵢ)=⟨√p,√q⟩`、**Fisher-Rao 距离** `d=2·arccos(BC)∈[0,π]`、**测地线**=
对 √p、√q 做 slerp 再平方、**exp/log 映射**经球面切空间互逆、**平行移动**=大圆旋转(保 Fisher 范数与切性的等距)、
**Karcher/Fréchet 重心**经 log/exp 迭代。因 √p≥0 夹角落在 [0,π/2],距离落在 [0,π],仅在 p==q 处退化。纯函数、
**0 可训练参数**、只 import torch/dataclasses/math、不 import `model.py`;`(...,n)` 广播。旗舰 `fisher_rao_geodesic`
把子算子复合成 `Geodesic`(采样点/弧长/中点)。`tests/test_geometry_ops.py` 21 项对解析真值固定契约(顶点距离=π、
两点闭式、exp/log 互逆、平行移动等距、重心=中点、测地线≠欧氏弦),`tests/test_geometry_ops_properties.py` 14 项
用 Hypothesis 在整个单纯形上锁定几何**定律本身**(距离对称/零对角/非负/有界 π/三角不等式/共置换不变;BC∈[0,1];
测地线在单纯形上/命中端点/匀速/可逆/置换等变;exp-log 互逆且 log 为零和切向量;平行移动等距;重心置换等变)。
诚实边界:本层是**算子**,尚未改写 `spatial_memory.py` 的读出度量——它把度量不一致**量化、坐实**,让那次改写
有据可依而非皮毛。

**Fisher-Rao 几何 demo (`examples/demo_geometry_ops.py`)**:取两个真实形状的 place code(对不同 logits 取 softmax,
正是 `PlaceCellCode` 吐出的单纯形点),用两种方式从一个走到另一个——L2 当前假定的**欧氏弦** `(1-t)p+tq` vs 它本该用的
**Fisher-Rao 测地线**。demo 坐实两者**不一致**:欧氏中点不是几何"中点"(用正确标尺量它读数更长),而测地线是
**匀速**的、其中点恰是 **Karcher 重心**;ASCII 条形图并排画出两个分布,匀速律表逐 t 验证 `d(p,γ(t))=t·d(p,q)`。
均确定性、CPU、亚秒、纯 ASCII。`tests/test_demo_geometry_ops.py` 5 项测试固定其契约(测地线匀速且中点=半距、
欧氏弦确实偏离、Karcher 重心=测地线中点、报告打印 OK、steps 参数控制表长)。

**可组合拓扑数据分析算子 (`mt_lnn/topology_ops.py`)**:`geometry_ops` 量**距离与曲率**,这一层量**形状**——
一整群状态的连通不变量(P0#2)。一组 place code / L2 记忆键 / 隐藏态就是一个点云,"它有几个互相分离的簇(读作:
吸引子盆地 / 不同的位置野),它们分得有多开?"是个**拓扑**问题,答案是它的 **0 维持续同调**(随尺度生长的 Betti-0):
连通分量数随合并半径的变化曲线,以及每个分量并入他者之前的**持续度**(寿命)。本层精确且廉价地算出这一切,所依赖的
唯一事实是**单连接=最小生成树**:两个分量恰在尺度达到桥接它们的最短边时合并,而这些桥接边在整条 filtration 上正是
MST——故 H0 死亡尺度的多重集**就等于** MST 边权多重集,Betti-0(ε)=`N − #{MST 边权 ≤ ε}`(精确, 非近似)。算子:
`minimum_spanning_tree`(Prim 单连接骨架)、`persistent_homology_h0`(H0 条形码:全在 0 出生、死亡尺度=排序 MST 边权、
恰一个分量永生即无穷条)、`betti0`/`betti0_curve`(**复用** `spatial_ops` 的并查集式 `connected_components` 数半径图的
分量,且与 MST 视角在测试里交叉校验)、`total_persistence`(有限条寿命之和,一个可微的"聚簇程度"标量)、`srtd`(对称
相对拓扑散度:两点云 H0 条形码排序后逐项比较的 1-Wasserstein,对称、非负、仅在条形码重合时为零——拓扑突变触发器信号)。
纯函数、**0 可训练参数**、只 import torch/dataclasses/math + 兄弟算子 `spatial_ops`、不 import `model.py`。度量量(MST 边权、
总持续度、SRTD)对点位置可微;整数 Betti 数阈值化、分段常值(真·离散拓扑),逐项标注。诚实边界:仅 H0(连通分量)——
持续同调里廉价精确的那部分;高维同调(环 H1、空腔 H2)与完整多维 Representation-Topology-Divergence 故意不做,`srtd`
即 H0 排序条形码代理,如实声明。`tests/test_topology_ops.py` 16 项对解析真值固定契约(线段 MST、簇计数、条形码=排序
MST 边权、betti0_at 与并查集 Betti-0 一致、SRTD 闭式值/对称/差拓扑变大),`tests/test_topology_ops_properties.py`
11 项用 Hypothesis 在整个点云空间锁定拓扑**定律本身**(MST 恰 N-1 边且总权=总持续度;Betti-0 从 N 降到 1 且对尺度单调
非增、置换不变;条形码 N 条、有限死亡排序非负、betti0_at 匹配并查集;总持续度置换/平移不变且随云线性缩放;SRTD 对称/
非负/自零/双侧置换不变且等基数三角不等式)。

**TDA 算子 demo (`examples/demo_topology_ops.py`)**:一群隐藏态点云的拓扑健康检查。先建三个干净的簇(5 点 ×3),
展示 Betti-0 **算出**簇数=3、其条形码量出簇间间隙、总持续度量分离强度;再两面施压:(1)小抖动——拓扑不变, Betti-0 仍 3、
SRTD≈0;(2)簇坍缩——拓扑被破坏, Betti-0 跌、SRTD 飙升。SRTD 正是**拓扑失效保护**该盯的那种 0 参数触发器:活点云与
健康参照之间的大 SRTD 意味着"表征刚刚改变了形状"——模式坍缩 / 转向编辑失手 / 突发 regime 变化。均确定性、CPU、亚秒、
纯 ASCII。`tests/test_demo_topology_ops.py` 5 项测试固定其契约(Betti-0 读出 3 簇且坍缩破坏之、SRTD 把抖动与坍缩拉开
≥5×、Betti-0 曲线 15→1 且单调、报告打印 OK、间距越大总持续度越大)。

**可组合 STDP 可塑性算子 (`mt_lnn/stdp_ops.py`)**:这一层补的是一种**局部、无反向传播**的学习规则(P0 学习)。
`plasticity.py` 在损失级跑 Hebbian 巩固平滑连续时间 LTC 核,并**明确说明为何 STDP 不适用于核**:连续时间核
(`h_t = e^{-dt/tau} h_{t-1} + ...`)里没有离散的 pre/post 脉冲事件可计时,硬塞 STDP 等于人为离散化 LTC、毁掉它
的定义性质。但架构不止连续核:salience 层(`salience_events.py`)发出**真正离散、带时间戳的点火事件**,L2 place-code
写入也是事件式联想更新——那正是脉冲**时序**有定义、STDP 这把**局部无梯度**工具该上场的地方。故本模块作为**与
`plasticity.py` 并行**(而非替代)的算子提供该规则:损失级 Hebbian 管平滑核,事件驱动 STDP 管离散事件流。规则是
经典非对称指数窗 `W(dt)`:`dt>0`(pre 先于 post, 因果)**增强**, `dt<0`(反因果)**压抑**, `dt=0` 恰为 0(同时无因果序)。
提供并交叉验证两套等价计算:`pairwise_stdp` 是全对全成对 `W(t_post-t_pre)` 求和(定义式),`stdp_trace_update` 是
标准在线**资格迹**形式(衰减 pre/post 迹;post 脉冲读 pre 迹做 LTP, pre 脉冲读 post 迹做 LTD),`O(T)`、即 STDP 在
硬件上的实跑方式,可证恰等于全对全求和。纯函数 + 冻结 `STDPParams`,**0 可训练参数**,仅 import torch/dataclasses/math,
**从不 import model.py**;窗值对时序可微、迹更新对(实值)脉冲幅度可微,脉冲计数离散这点如实声明、不藏。
`tests/test_stdp_ops.py` 15 项对解析窗固定契约(taus 处窗值=A·e⁻¹、同时性=0、因果符号、平衡时反对称、窗积分=A·tau、
单对=窗值、迹==成对、delta_w=增强−压抑且双叶非负、纯因果只增强),`tests/test_stdp_ops_properties.py` 8 项基于
Hypothesis 的可塑性律(窗符号因果且被幅度界住、每叶内单调衰减、平衡反对称、任意多神经元迹==成对、纯因果train只增强、
对幅度线性、全局时移不变)。

**STDP 算子 demo (`examples/demo_stdp_ops.py`)**:用两条脉冲流驱动一个突触,展示权重**仅凭时序**移动——无损失、无梯度。
(1)因果流(pre 先于 post)**增强**且压抑叶恰为零;(2)角色互换流(post 先)**压抑**且增强叶恰为零;(3)单个同时脉冲对
(dt=0)毫无变化。并交叉验证两套计算:在线资格迹更新与全对全成对窗求和到机器精度一致(|gap|~1e-15)。均确定性、CPU、
亚秒、纯 ASCII(含 W(dt) 窗的 ASCII 侧影)。`tests/test_demo_stdp_ops.py` 5 项测试固定其契约(因果增强零压抑、反因果
压抑零增强且符号严格翻转、同时性零变化、迹==成对、报告打印 OK)。

**可组合吸引子/自稳定算子 (`mt_lnn/attractor_ops.py`)**:本架构的定义性对象是一个**连续时间**递归核——`ProtofilamentLTC`
以 `h_t = e^{-dt/tau} h_{t-1} + ...` 向状态依赖的平衡点压缩。整套"沉降到稳定表征"的叙事(记忆回放弛豫到存储模式、
想象 rollout 滑行到不动点、place code 锁定一个 basin)本质都是关于**吸引子**的论断:沉降到哪、多快、能吸收多大扰动
才不掉进另一个 basin——而既有各层都没有真正**度量**它。本模块补上这层缺失的动力系统仪表,两套互相交叉验证的视角:
**解析视角**(已知线性映射 `x→Ax+b`,全闭式精确)给不动点 `x*=(I-A)⁻¹b`、谱半径 `ρ(A)`、渐近收敛率 `-log ρ`
(压缩 ⟺ `ρ<1`);**经验视角**(任意观测沉降轨迹, 含非线性 LTC)直接从状态序列估计:`relax` 滚动任意步进映射、
`settling_time` 入容差带的首步、`convergence_rate` 对数距离斜率(线性映射下恰等于 `-log ρ`)、`lyapunov_descent`
逐步能量比 `V_{t+1}/V_t`(`V=‖x-x*‖²`,全 `<1` 即单调能量下降的离散 Lyapunov 稳定性证书)。如实声明:Lyapunov 用
**欧氏**能量,全 `<1` 是充分而非必要——**非正规**压缩(`ρ<1` 但 `AᵀA≠AAᵀ`)可短暂增长后再衰减,渐近保证仍是 `ρ`;
正规/对称映射下每步比被 `ρ²` 界住、单调下降。**鲁棒性**:`basin_radius` 沿某方向二分探测,找出仍能回到吸引子的最大
推动量(吸引域半宽);全局压缩下无界(返回探针上限),近处有不稳定边界时则还原之(如 `ẋ=-x+x³` 中 `0` 的吸引域为
`(-1,1)`,探针返回 `1`)。纯函数 + 冻结 `AttractorReport`,**0 可训练参数**,仅 import torch/dataclasses/math/typing,
**从不 import model.py**;交给 `relax`/`basin_radius` 的步进映射是任意 `x→x` 可调用对象(一个向量场),故本层保持
模型无关、日后可包住真核而不依赖之。`tests/test_attractor_ops.py` 17 项对解析真值固定契约(不动点求解+单位特征值拒绝、
谱半径=最慢模/旋转-缩放块、率=-log ρ、压缩阈值、relax 复现几何轨迹、经验率==解析率、沉降时间匹配几何估计、Lyapunov
认证单调沉降且发散映射能量每步增长、basin 还原非线性边界/全局压缩封顶/中心非吸引子归零/零方向拒绝、flagship 复合一致),
`tests/test_attractor_ops_properties.py` 8 项基于 Hypothesis 的动力系统律(不动点确为不动、谱半径=最大特征值模、
率=-log ρ 且压缩标志一致、任意起点压缩收敛到 x*、对称压缩 Lyapunov 能量不增、经验率==渐近率、全局压缩 basin 封顶、
谱在坐标置换相似下不变)。

**吸引子自稳定 demo (`examples/demo_attractor_ops.py`)**:在两个手算系统上给沉降装仪表——(1)稳定线性映射(`ρ<1`):
有不动点/谱半径/解析收敛率/沉降时间, 且从弛豫轨迹**测出**的率与闭式 `-log ρ` 吻合(log 尺度 ASCII 衰减阶梯);
(2)发散线性映射(`ρ>1`):无稳定吸引子, 率为负, Lyapunov 能量每步增长——诊断把它**标红**。再上一个非线性系统
`ẋ=-x+x³`(`0` 稳定、`±1` 不稳定):basin 探针从 `0` 向外推, 还原吸引域半宽=`1`(稳定性边缘)。均确定性、CPU、亚秒、
纯 ASCII。`tests/test_demo_attractor_ops.py` 6 项测试固定其契约(稳定映射率匹配闭式且能量下降、线性 basin 封顶、发散映射
被标红、非线性 basin 还原边界、报告打印 OK、ρ 越小沉降越快)。

**P1 工程优化①——高阶辛积分器 (`mt_lnn/physics_ops.py` 的 `integrate_verlet` + `rollout(integrator=…)`)**:
imagination/世界模型 rollout 要把状态在力下向前滑很多步。若积分器**漏能量**,长程 rollout 会慢慢
**凭空造出**它本没有的能量(行星螺旋飞出、单摆越摆越高),"接下来会怎样"的预测就偏离物理。修法不是
缩小步长(贵),而是换一个**辛**积分器,让能量误差对所有时间**有界**。这里在既有半隐式 Euler(1阶,
能量误差 `O(dt)` 且**累积**)之外,新增 2阶、时间可逆的**速度 Verlet**(kick-drift-kick;能量误差 `O(dt²)`
且**有界**, 无长期漂移),经 `rollout(integrator="verlet")` 一行切换。诚实边界:Verlet 把力当作**仅依赖位置**,
对 `gravity`/`accel_fn(pos)` 精确, 若力实际依赖速度则为近似(已注明);它假定力是位置函数(每子步冻结 `v`,
碰撞/壁反弹仍逐步重算以保正确)。`tests/test_physics_ops.py` 新增 7 项(kick-drift-kick 形式、恒加速下精确、
能量漂移随 `dt` 减半 Euler≈2× 而 Verlet≈4×[即 1阶 vs 2阶]、Verlet 能量守恒优于 Euler 10×、Verlet rollout 时间可逆、
拒绝未知 integrator)。

**P1 工程优化①demo (`examples/demo_symplectic_integrator.py`)**:让一颗星沿圆形开普勒轨道(`GM=1`,`r0=1`)飞约
32 圈, 对比两种积分器——半隐式 Euler 漏能量、轨道半径向外漂(半径漂移 ≈2.6e-2), 而速度 Verlet 稳稳守住轨道
(≈1.3e-3, 能量漂移小 ≈1600×);再固定时长、`dt` 减半, 用**对解析圆轨道的位置误差**给出无歧义的精度阶——Euler
误差降 ≈2×(1阶 `O(dt)`)、Verlet 降 ≈4×(2阶 `O(dt²)`)。两幅 ASCII 半径阶梯图直观对照"螺旋外飞"vs"稳定"。均
确定性、CPU、亚秒、纯 ASCII。`tests/test_demo_symplectic_integrator.py` 5 项测试固定其契约。

**P1 工程优化②——拓扑失效保护 (`mt_lnn/failsafe.py` 的 `TopologyBreaker`)**:健康的潜在状态群有**结构**——
彼此分离的若干模式/簇(比如 3 个概念)。两种失效会抹掉它:**模式坍缩**(所有状态融成一团, 模型不再区分事物)
与**机制突变**(簇数跳变)。损失曲线和激活范数此时可能看着正常, 变的是状态点云的**拓扑**。`TopologyBreaker`
正盯着它:锁定一个健康参考点云后, 每拍测活点云相对参考的 H0 条形码散度(SRTD), 并可选测其连通分量数
(Betti-0);经一台去抖有限状态机(与经典 `CircuitBreaker` 同款)在连续几拍坏后**跳闸**、在结构恢复后**复位**,
单个噪声拍绝不翻转它。诚实边界:它是纯**外部健康信号**, 0 可训练参数, **绝不**改写表示——跳闸后由调用方
决定怎么办(冻结、回滚一次 steering 编辑、退回安全策略);它复合 `topology_ops.srtd`/`betti0`, 零核耦合。
`tests/test_failsafe.py` 新增 8 项(良性抖动不跳闸、坍缩去抖后跳闸、清净跑后复位、Betti 检测分量数变化、首云
自锁参考、0 参数与可复位、配置校验[参数化]、确定性)。

**P1 工程优化②demo (`examples/demo_topology_failsafe.py`)**:给断路器喂一串点云——良性抖动(同结构、新噪声)
让 SRTD≈0、永不跳闸;模式坍缩(簇融向一团)使 Betti-0 由 3→1、SRTD 飙升, 在**第 2 个坏拍**(trip_after=2)经去抖
跳闸;随后恢复(结构归来)在**第 3 个清净拍**(reset_after=3)复位。逐拍日志列出 SRTD/Betti-0/是否跳闸/原因。
均确定性、CPU、亚秒、纯 ASCII。`tests/test_demo_topology_failsafe.py` 6 项测试固定其契约。

**P1 闭环① —— 原始流式传感器前端 (`mt_lnn/sensory_frontend.py` 的 `SensoryFrontend`)**:闭合感知回路缺的
那块**时间前端**。backbone 是模态无关的——任何 `(B, T, d_model)` 都能经 `MTLNNModel.forward(inputs_embeds=...)`
喂进去;`multimodal.py` 已把**特征块**投成 d_model token。但一条**闭合**的感知回路还需要时间侧的对齐:真实传感器
不在液态核的固定 `dt` 栅格上到达(抖动、丢帧、漂移)。`SensoryFrontend` 正是这块前端——它**零新增 backbone 耦合**地
复合两件已有零件:`ingest_ops.align_stream`(把抖动、带时间戳的原始流重采样到固定 `dt` 栅格,并标出哪些步是真采样
支撑、哪些深陷丢帧)与可训练的 `multimodal.ModalityProjector`(把对齐后的逐步特征向量投到 d_model token 空间)。
产出一个 `SensoryEncoding`:`(B, M, d_model)` 的 `inputs_embeds` 加上**两面呈现的覆盖掩码**——作为 `pad_mask`(让丢帧步
不污染注意力)与作为 `BlindRolloutGuard` 滑行所凭的**信任信号**。它是可训练的 `nn.Module`(与 multimodal 的编码器同款,
与模型一起进 optimizer),但**从不 import 或修改 model.py**;时间对齐是纯 0 参数 `ingest_ops`。两种输入态势:传 原始
`(values, timestamps)` 走对齐(全批共享一个时钟),或传已均匀的值(`timestamps=None`)跳过对齐直接投影;异构逐项时钟
用 `align_batch`。诚实边界:这是一条可用、有测试的时间注入路径——不是预训练的感知塔;线性重采样只对仿射信号精确,
过宽的空洞被**标记**而非臆造。`tests/test_sensory_frontend.py` 12 项对解析对齐+投影真值固定其契约。

**P1 闭环①demo (`examples/demo_sensory_frontend.py`)**:用一条 12 通道、时钟抖动且中途**爆发丢帧**的原始流驱动
`SensoryFrontend`,展示:(1) 流落到干净的 `dt` 栅格上;(2) 覆盖时间线 `#####....#####` 恰好标出丢帧步(`'.'` 是
`BlindRolloutGuard` 要滑行的);(3) 投出的 `(B, M, d_model=104)` token 经 `inputs_embeds` 驱动 backbone 出有限
logits——回路闭合,零 `model.py` 耦合。确定性、CPU、亚秒、纯 ASCII。`tests/test_demo_sensory_frontend.py` 6 项测试固定其契约。

**P1 闭环② —— 自顶向下调制 (`mt_lnn/model.py` 的 `top_down` 通路)**:大脑高层不只接收自下而上的感觉,还发出
**自顶向下反馈**,用当前目标/上下文**偏置**低层处理。这是本次唯一**经授权触碰 `model.py`** 的改动,边界刻意收窄到
**4 处、约 +20 行**:`MTLNNModel.forward` 新增 `top_down` 形参 → 透传给每个 `MTLNNBlock` → 在**注意力之后、液态核
之前**以一道**零初始化门控残差**折入:`x = x + tanh(gate)·proj(LayerNorm(top_down))`。整条特性由 `config.use_top_down`
门控且**默认 OFF**——默认模型不建任何参数、字节级不变(对所有既有 checkpoint/测试零回归)。两条诚实性质:**(1) 闭门即
严格恒等**——`gate` init 0 → `tanh(0)=0` → 残差恰为零 → 喂任何目标与不喂**逐位等价**(预训练权重在门学会打开前毫发无
伤);**(2) 开门即转向**——门一旦打开,两个不同目标会把**同一输入**推向可测不同的下一 token 分布。`proj` 取 small-but-
nonzero(std 0.02)使**门**在 init 处仍有活梯度(模型可学会开门);`LayerNorm` 是那道**初始值保险**——把任意量级的目标
向量界住,门打开后残差不会冲垮训练。可选 `config.top_down_to_gwtb` 还把目标作为一路**外部投标**接入顶层全局工作空间竞争
(复用世界模型 bid 通路;**预留的 GWT 对接入口**)——该路因模型全局 init 把竞争内部 bid 推离严格恒等,init 仅在与世界
模型 bid **同量级的 O(1e-4) softmax 归一化伪差**内恒等(非逐位,默认 OFF,诚实标注)。`tests/test_top_down_modulation.py`
11 项固定其契约(默认零参数、闭门逐位恒等、开门转向、门有活梯度、形状广播、KV-cache 一致、GWT 对接)。

**P1 闭环②demo (`examples/demo_top_down_modulation.py`)**:用一个固定 token 序列驱动开启 `use_top_down` 的小模型,
展示:**[1]** 闭门时 `max|logits(目标) − logits(无目标)| = 0`(严格 no-op,逐位等价);**[2]** 把每个 block 的门打开后,
两个不同目标使下一 token 分布发散(KL > 0、argmax 改变)——通路确实活、默认关、零回归。确定性、CPU、亚秒、纯 ASCII。
`tests/test_demo_top_down_modulation.py` 5 项测试固定其契约。

**传感器摄入/流对齐算子 (`mt_lnn/ingest_ops.py`)**:液态核以**固定步长**离散其连续动力学——`ProtofilamentLTC`
衰减为 `exp(-dt/tau)`,`dt = config.dt` 是编译期常数。这只在输入**真的**按均匀 `dt` 栅格到达时才成立;真实传感器
不会照办:到达间隔抖动、偶发丢帧、时钟漂移。把这种非均匀采样直接喂进固定 `dt` 递归会**悄悄**违反离散化(一个迟到
一倍的样本被当成准时的积分进去)。`streaming.py` 解决的是另一个问题(单 token 步间 O(1) 携带递归状态),它**假设**
调用方已按 `dt` 递交一帧;此前没有任何上游把**带时间戳**的不规则流转成那个均匀栅格。这一层正是那个缺失的前端:一组
**可组合、0 参数、解析**算子,把带时间戳的流重采样到核所要的均匀 `dt` 栅格(`resample_uniform`,线性对线性信号
**精确**/ZOH 采样保持),并——这是诚实的一半——用 `coverage_mask` 标出落在"长到无法插值"的丢帧空洞里的栅格步,
让下游 `failsafe.BlindRolloutGuard` 去**滑行**那段黑暗中段,而不是在空洞上编造平滑插值。旗舰 `align_stream` 把
重采样 + 覆盖判定复合成 `AlignedStream`(均匀值/其时刻/覆盖掩码/空洞步数),直接驱动固定 `dt` 核与其失效保护。
诚实边界:这是确定性**算子**对齐(无 Kalman 状态、无学习插值);跨长空洞它不臆造高阶拟合——而是标出空洞、交给预测器。
纯函数、**0 可训练参数**、不 import `model.py`。`tests/test_ingest_ops.py` 21 项对解析真值固定其契约(线性重采样
对斜坡**精确**、原均匀时刻即恒等、ZOH 采样保持、越界不外推只钳端点、空洞中点被标记/良采样边缘被信任、均匀时钟抖动为零、
端点处理、可微、确定性)。

**双速引擎的慢半边 (`mt_lnn/slow_layer.py`)**:双速系统若慢层从不真正运行,就只搭了一半。热环每拍只做**一步**
预测编码前推(便宜、常数时间、永远跑),这一步够**察觉**惊讶,但不够**推理**。这一层补上另一半:`SlowThreatAssessor`
**仅在点火时被唤醒**,用可组合牛顿算子(`physics_ops.rollout`)把目标按当前速度多步外推整整一个 horizon,逐步问
"它进保护区了吗"(`spatial_ops.in_ball`),据此算出**突破 ETA**(还有几步突破,0=此刻已在内)、**最近接近距离**
与离散**威胁等级**(CLEAR 不突破 / WATCH 会突破但不紧迫 / ENGAGE 即将或已突破)+ 处置姿态。这正是热环单步做不了
的审议,且遵守双速契约——只在真状态变化挣到它时才付这笔算力。纯算子复合、**0 可训练参数**、不 import `model.py`、
跨调用无状态、确定性。诚实边界:这是"若一切不变会怎样"的弹道前瞻(假定目标保持当前速度),是机动刚被发现那一刻
该问的问题,不是对抗性意图模型。`tests/test_slow_layer.py` 12 项固定其契约(迎面目标 ETA 精确、远离即 CLEAR、
切向掠过不突破、ETA 驱动 WATCH↔ENGAGE 升级、horizon 截断、0 参数、确定性)。

**双速哨兵编排 (`mt_lnn/pipeline.py`)**:前面每一层都是一个能力,但能力堆在一起不是系统——**环**才是系统。
这一层是产品本身:一个边缘周界哨兵,把所有层按各自**本职**串成一个闭环,只有这个环才让它们成为一个系统。
逐拍 **感知**(`acoustic_ops` 从声音的 ITD 反演无人机方位、多普勒听它来去)→ **推理**(`spatial_ops.in_ball`
每拍问"它进保护半径了吗")→ **预测**(`physics_ops` 一步弹道前推,与实测之差是训练无关的预测编码**惊讶**残差)
→ **触发 + 唤醒慢层**(`salience_events` 仅当惊讶**点火**——无人机机动时——才唤醒慢层,绝不轮询热环;而"唤醒"
是**真动作**:点火即调用 `slow_layer.SlowThreatAssessor` 做多步前瞻威胁评估,把 ETA/威胁等级随 `PerceptionEvent`
广播上去)→ **续命**(`BlindRolloutGuard` 在传感器断流时靠世界模型想象**盲推**续命,信心或预算耗尽则转**暗**
安全停)→ **安全执行**(`CircuitBreaker` 把转台瞄准命令钳在机械限位与转速率内,每拍无条件清 NaN/限幅/限速)。剧情:
无人机直线接近(哨兵跟踪但不惊动慢层)→ 急转规避(点火一次→唤醒慢层评估 ENGAGE)→ 传感器抖动两拍(盲推续命)→
突破保护区。编排器**零新增可训练参数**(`n_parameters==0`)、零 `model.py` 耦合;唯一带参的是 guard 盲推所依赖
的可选世界模型 head,且只用其可用性/信任预算门控(几何信任衰减,独立于是否训练),不靠其预测精度。诚实边界:远场
声学模型要求场景在头前方,近头方位饱和(钳位),执行器转速率限位正好挡住由此产生的尖峰。`tests/test_pipeline.py`
19 项固定层间组合契约(稳态静默/机动点火一次、点火真唤醒慢层并带回评估/未点火慢层零调用、断流盲推→恢复/长断流
转暗、逐拍区域判定、命令恒在限位+限速内),确定性、0 参数。

**双速哨兵 demo (`examples/demo_pipeline.py`)**:把整环跑成一个边缘周界哨兵故事——ASCII 俯视图(H 传感头、
O 区域中心、o 危险半径环、数字无人机轨迹、# 点火拍)+ 逐拍日志(方位/距离/多普勒/区域/瞄准/备注)+ 判决。
输出:稳态接近 14 拍唤醒零次,机动时**恰好一次**显著点火(z≈106,在机动而非接近时)→ 唤醒慢层评估出
**威胁 ENGAGE / 0 步突破 / 最近 0.9 m**(日志与摘要均显式打印),断流盲推 2 拍后恢复,5 拍在区域内,每条瞄准命令
恒在 ±90° 内且每拍 ≤20°。**断流不再硬编码**:原始馈送先经 `ingest_ops.align_stream` 对齐到固定 `dt` 栅格,丢帧
成为真实的**覆盖空洞**(19 个时间戳样本对齐到 21 步栅格、2 帧丢失→覆盖空洞→滑行),`sensor_ok` 由覆盖掩码**派生**
而非写死;干净时钟下覆盖步为恒等重采样,故所有既有行为契约不变。均确定性、纯 ASCII(Windows/GBK 安全)。
`tests/test_demo_pipeline.py` 11 项测试固定其行为契约(含慢层判决进入报告 + 摄入前端把丢帧识别为覆盖空洞)。

**自主认知智能体 demo (`examples/demo_cognitive_agent.py`)**:把整条类脑栈串成一个**自主智能体闭环**,在带障碍物的
2-D 场景里完成"感知→记忆→注意→想象+物理校验→故障盲推→安全输出"——ASCII 俯视图(S 起点、G 目标、# 障碍、. 提交路径)
+ 六阶段分项报告 + 判决。每一阶段都**真调用真实模块**(非 mock):**①感知** `GridCellEncoding`(L1)把每个物体坐标编成
36 维内嗅网格细胞群码,异物近正交(平均离对角余弦≈0.09,自余弦=1.0,空间作为度量空间);**②记忆** `SpatialMemory`(L2)
把(位置,内容)写进按位置索引的认知地图,锐化 place field(`sigma=0.04`<物体间距)解决中心簇障碍,模糊位置线索仍模式补全
召回(clean≈0.99/fuzzy≈0.99,0 可训练参数);**③注意** 把目标向量经 top_down 通路注入——闭门**严格 no-op**(隐状态逐位等价,
`closed_diff=0`),开门**转向**(`open_diff≈0.65`,更大门转向不减);**④想象** `LatentImagination`(L4)以目标调制后的隐态做
潜空间 rollout(0 参数、置信随地平线衰减),同时 `physics_ops.overlapping_pairs` + 有限差分速度/加速度对**三条候选路径**逐条
做碰撞与运动学校验(直线撞墙→绕行被提交);**⑤盲推** 中途切断感知,`BlindRolloutGuard` 靠内部世界模型滑行数拍,失信后转
**DARK**("失去感知,停止行动");**⑥安全** 提交路径的转向命令经 `CircuitBreaker` 硬钳到 ±45°/步包络(掺入 NaN+9.0 rad 尖刺验证钳位
真生效)。唯一动 model.py 的是 top_down,其余层全部 0 参数、零 model.py 耦合;均确定性、纯 ASCII、秒级。
`tests/test_demo_cognitive_agent.py` 10 项测试固定其端到端契约(六阶段逐项 + 整环判决 + 报告)。

**流式持续学习 demo (`examples/demo_streaming_continual.py`)**:把 P2 的"流式训练 + 回放 + 持续学习评测"落成一个可跑文件,
在**真实 MTLNNModel** 上证明回放对抗灾难性遗忘。同一任务序列(每个任务由前缀任务 token 标识、规则为该任务特定的循环移位
`x_{k+1}=(x_k+shift_t) mod V`——前缀使各任务**联合可学**,但顺序训练会遗忘)训练两遍:**朴素**顺序训练完美学会每个任务
(learning accuracy≈1.0)却**彻底遗忘**(最终准确率塌到≈1/任务数、forgetting measure≈1.0、BWT≈-1.0);interleave 一个有界
`mt_lnn.replay.ReservoirBuffer`(Vitter 蓄水池采样,O(C) 内存保持对全流的**均匀**样本)做排练后**几乎不遗忘**
(forgetting≈0、最终准确率≈1.0),且学新任务一样好、**新增 0 个模型参数**(回放是数据排练而非扩容)。两条 T×T 准确率矩阵
经 `mt_lnn.continual_eval` 的标准指标(平均/学习准确率、BWT、forgetting measure、forward transfer)打分。回放缓冲与评测均为
**0 参数、零 model.py 耦合**的纯算子;ASCII 准确率矩阵 + 判决。`tests/test_demo_streaming_continual.py` 7 项测试固定其契约
(朴素确遗忘 / 回放保留且仍在学 / 0 参数 / 矩阵方阵 / 生成器循环律 / 任务构造 / ASCII OK 报告;5 项训练型打 `slow` 标记)。

**算子代数数学白皮书 (`mt_lnn_operator_algebra_whitepaper.tex`)**:把整个算子层的数学**一次性、诚实地**收拢成一篇自包含的
LaTeX 技术论文(standard `article` + `amsmath/amsthm`,因本机无 LaTeX 编译器故交付 `.tex` 源,与仓库其它未编译论文一致)。
主旨:一套**零参数、与 backbone 解耦、由 property 测试钉住**的脑启发算子,环绕连续时间 LTC 核构成一个可复合的"代数"。
逐个给出 11 个算子(attractor/geometry/stdp/topology/physics/acoustic/spatial/ingest/replay/continual_eval/salience)与 LTC 核的
**精确数学定义**,并以定理式环境陈述每个算子的不变量。**核心诚实承诺**(§9):严格区分三档主张——少数**闭式恒等式**
(仿射不动点、MST=H0、辛 Euler 闭式、线性重采样对仿射信号精确、Vitter C/n)、多数**property 测试钉住的律**
(Hypothesis 在有界搜索空间内的经验固定,**非**形式化证明)、以及两处**统计/场景**主张(replay 蒙特卡洛均匀性、手算矩阵)。
§10 罗列诚实边界:eps 地板破坏精确尺度不变、一阶辛积分能量稳定非精确、H0-only 拓扑代理、远场声学近似、阈值不可微、
STDP/黎曼算子按设计**未**接入训练目标。并据实纠正代码 prose 的过度主张(`continual_eval` 的 "FM≥0" 被其自身测试证伪——
正向回迁下 FM 可为负)。

**类脑 Phase-1 三件套行为测试(补齐覆盖空洞)**:路线图 `docs/specs/BRAIN_INSPIRED_ROADMAP.md` 第一阶段三机制——多尺度**预测编码 loss**、
**O(1) 工作记忆衰减**、内源性**动态 κ 通道门控**——均**默认开启**(`use_predictive_coding=True` / `use_decay_wm=True` /
`dynamic_scale_gates=True`)却长期**无行为测试**(此前测试只把它们设 `False` 以静默)。现补三份测试钉住其真实契约:
`tests/test_predictive_coding_loss.py` 9 项[`W_pred` 形状(P,S-1,D,D)仅在启用且 S>1 时存在、训练态 `last_pred_error>0` 而 eval 态恒 0、
模型 `pred_loss` = 各块之和、**梯度抵达 `W_pred`(证明 loss 真接入)**、`predictive_loss_weight` 线性缩放贡献];
`tests/test_decay_working_memory.py` 6 项[`update_gate`/`decay_rate` 旗标接线、**层级与模型级 coherence cache 在单 token 流式下恒为
O(1)** 而 legacy 路径 O(T) 线性增长(招牌"无限轮长对话"首次被真测)、零输入下工作记忆几何遗忘、非零输入触发写入];
`tests/test_dynamic_scale_gating.py` 7 项[`kappa_gate` 按旗标存废、门控**真随输入变化**而静态路径恒为全 1、**梯度抵达 `kappa_gate`**、
稀疏核恰选出 top-k 个尺度且比率正确、稀疏前向相对稠密改变输出]。诚实说明:此前文档"tests for everything"对这三项**不成立**,
此补丁堵上空洞;并据实纠正——动态 κ 默认仅**重加权**不省 FLOPs(真正跳算需 `sparse_resonance_kernel=True`,默认关)。

**Test coverage**: 967 tests in `tests/` (含 `test_spatial.py` 17 项空间前端测试[含
`PlaceCellCode` 5 项]、`test_thinking.py` 10 项自我思考测试、`test_spatial_reasoning.py`
14 项空间思考测试[含 7 项 L2 记忆侧通道]、`test_causal_steering.py` 9 项因果转向测试、
`test_causal_decoding.py` 10 项 L3 解码闭环转向测试、`test_demo_causal_decoding.py`
10 项 L3 解码闭环 demo 测试、`test_demo_causal_spatial_steering.py`
10 项 L3 转向 demo 测试、`test_spatial_memory.py` 11 项 L2 空间序列记忆测试、
`test_demo_spatial_memory.py` 12 项 L2 记忆 demo 测试、`test_imagination.py`
16 项 L4 潜空间想象 rollout 测试、`test_demo_imagination.py` 7 项 L4 想象 demo 测试、
`test_spatial_ops.py` 19 项可组合几何算子测试、`test_demo_spatial_ops.py` 3 项几何算子 demo 测试、
`test_spatial_ops_properties.py` 14 项基于 Hypothesis 的几何不变量测试[距离对称/零对角/非负/三角不等式/
刚体运动(平移+旋转)不变、相对方向单位且反对称、bearing 与方向一致、半径图对称且按距离阈值且对半径单调、
kNN 图对称且度数≥k、可达性含种子且与跳距一致且对跳预算单调、完全图全可达、连通分量沿边恒定且匹配可达集]、
`test_physics_ops.py` 32 项可组合牛顿动力学算子测试[含 PhysicsRollout 无批次/批次摘要访问器回归 +
速度 Verlet 7 项: kick-drift-kick 形式/恒加速精确/能量漂移随 dt 减半 Euler≈2×而 Verlet≈4×/Verlet 能量守恒优于 Euler 10×/
Verlet rollout 时间可逆/拒绝未知 integrator]、
`test_physics_ops_properties.py` 12 项基于 Hypothesis 的物理不变量测试[symplectic Euler 精确闭式、
N 体引力守恒动量(牛顿第三定律)+平移不变、碰撞恒守动量、弹性恒守能量、非弹性不增能、墙反射保速]、
`test_demo_physics_ops.py` 3 项物理算子 demo 测试、
`test_demo_symplectic_integrator.py` 5 项辛积分器(Verlet vs Euler)demo 测试、
`test_salience_events.py` 16 项全局工作空间点火事件测试、`test_demo_salience_events.py` 4 项点火事件 demo 测试、
`test_salience_events_properties.py` 11 项基于 Hypothesis 的点火检测器不变量测试[确定性且 update 与 observe 一致、
warmup 窗口内不触发、点火/沉寂严格交替且首个必为点火(Schmitt 滞回)、相邻事件间隔≥refractory+1、
点火 z≥ignite_z 且沉寂 z≤release_z 且字段自洽、常量流永不触发、平稳基线上单脉冲恰触发一次点火、
z 分数(及整条事件流)在信号加性 DC 偏移下精确不变、reset 清空全部状态、world_model_surprise 鸭子类型桥接]、
`test_failsafe.py` 41 项断流盲推 + 输出断路器测试[含 TopologyBreaker 8 项: 良性抖动不跳闸/坍缩去抖后跳闸/清净跑后复位/
Betti 检测分量数变化/首云自锁参考/0 参数与可复位/配置校验/确定性]、`test_demo_failsafe.py` 8 项断流盲推/断路器 demo 测试、
`test_demo_topology_failsafe.py` 6 项拓扑失效保护 demo 测试、
`test_sensory_frontend.py` 12 项原始流式传感器前端测试[对齐前向 shape+pad_mask 别名/均匀态势跳过对齐全覆盖/
无批维输入压缩批维/线性重采样对仿射信号精确/宽丢帧被覆盖掩码标记/共享时钟覆盖跨批一致/embeds 经 inputs_embeds 驱动
backbone 出有限 logits/梯度只达投影器/配置校验/坏 rank 拒绝]、`test_demo_sensory_frontend.py` 6 项感官前端 demo 测试、
`test_top_down_modulation.py` 11 项自顶向下调制测试[默认关零参数且忽略 top_down/启用建每块适配器/闭门逐位恒等/开门改输出/
门有活梯度/开门后 proj+目标得梯度/形状广播与逐步皆可/KV-cache 与全前向一致/GWT 对接需竞争式 GWTB/对接 init 在 softmax
伪差内恒等/对接开门改变竞争]、`test_demo_top_down_modulation.py` 5 项自顶向下调制 demo 测试、
`test_acoustic_ops.py` 36 项可组合声学/双耳听觉算子测试、`test_demo_acoustic_ops.py` 7 项声学算子 demo 测试、
`test_acoustic_ops_properties.py` 15 项基于 Hypothesis 的声学不变量测试[传播延迟非负/对称/还原距离/与声速成反比、
球面扩散增益正且还原 ref_dist 且随距离单调下降、ITD 交换双耳反号且受 head_width 界约束且在垂直平分面上为零、
ILD 交换双耳反号且与 ITD 同号、Doppler 静止场恒等且接近升频远离降频、波前叠加对幅度线性且两同相重合源幅度翻倍且模满足三角不等式、
定位精确反演远场平面波模型且单调且在极点饱和、binaural_scene 逐场与各 cue 算子一致]、
`test_geometry_ops.py` 21 项可组合 Fisher-Rao 信息几何算子测试[顶点距离=π/两点闭式/exp-log 互逆/平行移动等距/
重心=测地线中点/测地线≠欧氏弦]、`test_demo_geometry_ops.py` 5 项 Fisher-Rao 几何 demo 测试、
`test_geometry_ops_properties.py` 14 项基于 Hypothesis 的信息几何不变量测试[距离对称/零对角/非负/有界 π/
三角不等式/共置换不变、BC∈[0,1]、测地线在单纯形上且命中端点且匀速且可逆且置换等变、exp-log 互逆且 log 为零和切向量、
平行移动保 Fisher 范数与切性(等距)、Karcher 重心=测地线中点且置换等变]、
`test_topology_ops.py` 16 项可组合 TDA 算子测试[线段 MST=最近邻链、簇计数、条形码=排序 MST 边权、betti0_at 与并查集
Betti-0 一致、总持续度可微、SRTD 闭式值/对称/差拓扑变大/可微]、`test_demo_topology_ops.py` 5 项 TDA 算子 demo 测试、
`test_topology_ops_properties.py` 11 项基于 Hypothesis 的拓扑不变量测试[MST 恰 N-1 边且总权=总持续度、Betti-0 从 N 降到 1
且对尺度单调非增且置换不变、条形码 N 条且有限死亡排序非负、betti0_at 匹配并查集、总持续度置换/平移不变且随云线性缩放、
SRTD 对称/非负/自零/双侧置换不变且等基数三角不等式]、
`test_stdp_ops.py` 15 项可组合 STDP 算子测试[taus 处窗值=A·e⁻¹、同时性=0、因果符号、平衡反对称、窗积分=A·tau、
单对=窗值、迹==成对(单/多神经元)、delta_w=增强−压抑且双叶非负、纯因果只增强反之只压抑、单同时脉冲零变化、参数校验]、
`test_demo_stdp_ops.py` 5 项 STDP 算子 demo 测试、`test_stdp_ops_properties.py` 8 项基于 Hypothesis 的可塑性律测试
[窗符号因果且被幅度界住、每叶内单调衰减、平衡反对称、任意多神经元迹==成对、delta_w 双叶非负、纯因果 train 只增强、
对幅度线性、全局时移不变]、
`test_attractor_ops.py` 17 项可组合吸引子/自稳定算子测试[不动点求解+单位特征值拒绝、谱半径=最慢模/旋转-缩放块、
率=-log ρ、压缩阈值、relax 复现几何轨迹、经验率==解析率、沉降时间匹配几何估计、Lyapunov 单调沉降+发散映射能量每步增长、
basin 还原非线性边界/全局压缩封顶/中心非吸引子归零/零方向拒绝、flagship 复合一致]、`test_demo_attractor_ops.py` 6 项
吸引子自稳定 demo 测试、`test_attractor_ops_properties.py` 8 项基于 Hypothesis 的动力系统律测试[不动点确为不动、
谱半径=最大特征值模、率=-log ρ 且压缩标志一致、任意起点压缩收敛到 x*、对称压缩 Lyapunov 能量不增、经验率==渐近率、
全局压缩 basin 封顶、谱在坐标置换相似下不变]、
`test_ingest_ops.py` 21 项传感器摄入/流对齐算子测试、
`test_ingest_ops_properties.py` 10 项基于 Hypothesis 的算子不变量测试[线性重采样在
**任意**仿射信号上精确、ZOH 只输出真实样本值、覆盖掩码对 max_gap 单调、抖动在均匀时钟上为零等]、
`test_slow_layer.py` 12 项双速引擎慢层多步威胁评估测试、
`test_pipeline.py` 19 项双速哨兵编排集成测试、`test_demo_pipeline.py` 11 项双速哨兵 demo 测试、
`test_long_context_memory.py` 4 项 O(1) 流式内存回归测试[在 T = 20× RoPE 窗口处钉死
state-only cache 字节恒定,并与 KV cache 的 O(T) 线性增长做对比]、
`test_demo_cognitive_agent.py` 10 项自主认知智能体闭环 demo 测试[网格细胞码分离/认知地图按位置召回/
闭门 no-op-开门转向/想象 0 参数+置信衰减/直线撞墙-绕行可行被提交/盲推滑行后转 DARK/命令恒在安全包络内+整环判决]、
`test_replay.py` 14 项经验回放缓冲测试[首次 add 推断字段 schema 且并行字段逐行对齐、填充阶段全保留后封顶、
**蓄水池均匀性不变量**(n≫C 后每条流元素留存频率≈C/n,首尾无偏)、按种子确定、采样有放回可超容量/无放回封顶且对齐、
add_batch==逐条 add、错误显式、类均衡蓄水池按标签路由+预算均分+采样跨类铺开]、
`test_continual_eval.py` 9 项灾难性遗忘指标测试[完美保留→BWT=0/FM=0、完全遗忘→精确量化、正向后向迁移、
FM 用历史最好非刚学、forward transfer 高于基线、一次性汇总 bundle、单任务边界为零、形状校验]、
`test_demo_streaming_continual.py` 7 项流式持续学习 demo 测试[朴素确遗忘/回放保留且仍在学/0 参数/矩阵方阵/
生成器循环律/任务构造/ASCII OK 报告])。

其中 12 项「真训练循环 / 下载 CLIP 权重」的重量级测试打了 `@pytest.mark.slow` 标记
(`test_real_clip_vision_tower_smoke`、`test_world_model_long_run_surprise_bounded_no_collapse`、
`test_overfit_single_batch`、`test_v2_mechanism_effectiveness.py` 中 4 项多步训练测试,以及
`test_demo_streaming_continual.py` 中 5 项真训练持续学习测试)。
全套 `python -m pytest tests/` ≈ 8 分钟(其中单是 CLIP 权重下载就占 ~258s);快速冒烟路径
`python -m pytest tests/ -m "not slow"` 跑 955 项 ≈ 99s(5× 加速),markers 仅启用筛选、不改变默认全跑。

---

## 3. 层级架构 (five-layer view)

```mermaid
flowchart TB
    User([User Input]) --> L4

    subgraph L4["Layer 4 · Meta-Learning Plane (离线)"]
        ML["Capsule Clustering → Personal Prior<br/>memory.py · meta_learning.py"]
    end

    subgraph L3["Layer 3 · Verifiable Trace Plane"]
        PHI["Φ-IIT Monitor  phi_iit.py"]
        UI["Reasoning Timeline  trace_timeline.html"]
        PHI --> UI
    end

    subgraph L2["Layer 2 · Deliberation Router"]
        E1{"Token Entropy"}
        E2{"Semantic Entropy (N-sample)"}
        E3{"Fact Gap?"}
        E4{"Causal Consistency? ← Phase B"}
        E1 -->|low| OUT1[Local direct]
        E1 -->|mid| E2
        E2 -->|converge| OUT2[Local deep think]
        E2 -->|diverge| E3
        E4 -->|break| OUT2
        E3 -->|yes| CLOUD[Cloud Oracle]
        E3 -->|no| OUT3[Self-revise]
    end

    subgraph L1["Layer 1 · Stateful Reasoning Plane"]
        CAP["Capsule v2: belief_state + open_q + evidence_log"]
        STREAM["streaming_inference  O(1) state-only"]
        CAP <--> STREAM
    end

    subgraph L0["Layer 0 · Neural Backbone"]
        MT["MTLNNLayer: 13-filament LTC + κ-gate + LAVI rhythm"]
        CGWTB["CompetitiveGWTBLayer ← Phase A"]
        WM["PredictiveStateHead ← Phase C"]
        HEBB["HebbianRegularizer (train only) ← Phase D"]
        MT --> CGWTB --> WM
    end

    L4 -.feeds prior.-> L1
    User --> L1
    L1 --> L2
    L2 --> L3
    L0 -.serves.-> L1
    L0 -.serves.-> L2
    CLOUD -.quiet inject.-> CAP
    L3 --> Output([Answer + Φ-Trace])
```

---

## 4. Layer 0 — Neural Backbone 组件详解

### 4.1 已完成组件

#### 微管 LTC 层 (`mt_lnn_layer.py`)
- 13 protofilaments × 5 τ时间尺度，完全向量化并行
- LateralCoupling: 静态W_lat + 近邻耦合 + RMC内容感知
- MAPGate: per-protofilament 稳定性门控
- GTP水解周期性重置: 长上下文不丢失侧向混合

#### κ-gate 动态计算跳过 (`mt_lnn_layer.py:VectorizedMultiScaleResonance`)
- 基于**当前输入内容**的 τ 尺度权重
- 回答: "现在在处理什么" → 选择激活哪些时间尺度
- 稀疏模式 (sparse_resonance_kernel=True): top-k 尺度计算跳过

#### LAVI 节律门控 (`rhythm.py`) — 2026-06-06 完成
- 基于 **h_prev vs 输入相似度** 的节律检测
- 回答: "状态有多稳定" → 持续/瞬态模式切换
- 高 LAVI → 慢 τ 主导 (上下文维持)
- 低 LAVI → 快 τ 主导 (快速适应)
- GlobalRhythmController: 跨层节律聚合 + GWTB 前残差校正

```
κ-gate (内容信号) ⊕ LAVI (历史信号) = 完整稳定性-灵活性权衡
```

#### GWTB — 全局工作区瓶颈 (`gwtb.py`)
- 压缩 d_model→d_gw → 工作区SA → 广播回 d_model
- broadcast_gate 初始化为 0.01: 从 identity 出发渐进学习
- 实现 Baars/Dehaene GWT 的容量约束 (bottleneck = 意识瓶颈)

#### GlobalCoherenceLayer (`global_coherence.py`)
- Sparse top-k 因果注意力 (保留最高10%分数)
- Orch-OR 坍缩门: 基于注意力能量的二值化
- 可选衰减工作记忆 (use_decay_wm=True): O(1) 空间复杂度

---

### 4.2 Phase A — CompetitiveGWTBLayer 🔨 进行中

**生物原型**: GWT 的核心主张是"意识内容"通过**竞争**决定——多个专用处理模块(视觉、记忆、语言、推理)同时向全局工作区"投标"，只有赢家的表示被全局广播。

**现状缺口**: 当前 `GWTBLayer` 将单条 `x` 流直接压缩，没有多源竞争。

**实现方案**:

```
CompetitiveGWTBLayer
  ├── 继承 GWTBLayer (退化路径，module_bids=None 时完全兼容)
  ├── 新增: ScoreHead — 对每个 bid 打分 (B,T,d_model) → (B,T,1)
  ├── 竞争: softmax(scores) 加权融合 bids → z_combined
  └── 广播: 走原有 compress→SA→broadcast 流程

module_bids 来源 (通过 MTLNNModel.forward 可选注入):
  - lnn_out       : 微管 LTC 隐态 (已有)
  - attn_out      : 多头注意力输出 (已有)
  - coherence_out : 全局相干信号 (已有)
  未来可扩展: world_model_pred / causal_signal
```

**Config 开关**: `use_competitive_gwtb: bool = False`
**触碰现有测试**: 零（默认 False）
**新增测试**: `tests/test_gwt_competition.py` (8+ 测试)

---

### 4.3 Phase B — CausalConsistencyChecker ✅ 已实现 (v2.1)

**生物原型**: 前额叶皮层对推理链进行持续的"预测误差"监测——当当前输出违背已建立的因果结构时，产生错误信号并触发重新评估。

**现状缺口**: `deliberation.py` 的路由只基于熵值，不检测生成内容是否与先前状态逻辑一致。

**注意**: 原方案建议接 Prolog/CaRing 推理引擎。**不采纳**。原因:
- Prolog 查询延迟 10-100ms，违反 I2 不变量 (<50ms/token)
- 触发频率 <1%，ROI 接近零
- 正确方案: 用已有的 h_prev 轨迹本身作为因果一致性信号

**实现方案**:

```python
# mt_lnn/causality.py
class CausalConsistencyChecker:
    """
    检测隐态轨迹中的因果断裂。
    
    原理: 正常因果链的 h_t 是 h_{t-1} 的平滑演化。
    断裂信号: 连续 window 步内 cosine_sim(h_t, h_{t-1}) < threshold
    
    接口: checker.update(h_prev) → consistency_score ∈ [0,1]
    集成: deliberation.py RouterDecision 增加 causal_consistency 字段
          consistency_score < 0.3 → 强制 SELF_CRITIQUE，无论熵值
    """
```

**deliberation.py 改动**: `RouteDecision` 新增可选字段，`decide()` 新增可选参数。完全向后兼容。

**v2.1 实现要点 — 两种 method**:

`CausalConsistencyChecker(window, ema_alpha, threshold, method="cosine"|"subspace", energy_keep=0.9)`。

- `method="cosine"` (默认，向后兼容): `unit_cosine_similarity(h_t, mean(window))` ∈ [0,1]。
- `method="subspace"` (各向异性鲁棒): 对 LLM 隐态而言，余弦相似度因 anisotropy（所有隐态共享一个主导方向）而饱和接近 1，对真实的因果断裂"失明"。子空间残差法先**用窗口均值中心化**（移除共享方向），对中心化窗口做 SVD，保留捕获 `energy_keep` 方差的前 k 个右奇异向量，把中心化的当前向量投影到该子空间，**子空间外的残差能量 = 新颖度**，`consistency = 1 - novelty`。实测：话题切换时 cosine delta ≈ 0.000（盲），subspace ≈ 0.414（min 0.172 越过 0.35 触发地板）。
- `effective_rank` 属性: `(Σλ)² / Σλ²`（participation ratio），表征当前表示维度，写入诊断 `causal_effective_rank`。
- `from_config(config, **overrides)` classmethod: 读取 `causal_check_*` 字段构造。
- 共享工具: cosine→[0,1] 仿射映射统一为 `utils.unit_cosine_similarity()`。

---

### 4.4 Phase C — PredictiveStateHead ✅ 已实现 (v2.1)

**生物原型**: 预测编码理论 (Friston Free Energy Principle) — 大脑持续预测下一时刻的感知状态，以预测误差驱动学习。现有 `VectorizedMultiScaleResonance.use_predictive_coding` 只在**τ尺度之间**预测（慢τ预测快τ），缺少 token 级别的前向预测。

**实现方案**:

```python
# mt_lnn/world_model.py
class PredictiveStateHead(nn.Module):
    """
    给定 h_t，预测 h_{t+1} 的嵌入向量。
    
    损失: MSE(W_pred · h_t, embed(x_{t+1}).detach())
    权重: config.world_model_loss_weight = 0.01 (极小，不影响LM loss)
    
    推理时附加功能:
    - 预测误差 pred_error_t 作为 LAVI 的补充输入
      (模型自己对下一步的预测是否准确 → 更精准的节律感知)
    - pred_error 写入 last_pred_error buffer (监控用)
    """
```

**与 rhythm.py 的联动**: PredictiveStateHead 的误差信号可以增强 LAVIEstimator — 当预测误差突增，也标志着瞬态模式应该启动。

**v2.1 实现要点 — BYOL/V-JEPA 自预测，防表示坍缩**:

朴素方案 `predict(h_t) → h_{t+1}.detach()` 让模型"预测自己"，会导致**表示坍缩**（所有输入映射到同一潜方向，pairwise|cos|→1，surprise 信号失去意义）。v2.1 改为 BYOL (arxiv 2006.07733) / V-JEPA (arxiv 2404.08471) 风格的非对称自蒸馏：

- `online_proj: Linear(d_model, proj_dim, bias=False)` + `predictor: [Linear→GELU→Linear]`（在线分支，可训练）。
- `target_proj: Linear(d_model, proj_dim, bias=False)`，`requires_grad=False`，由 online_proj 的 **EMA** 跟踪（`ema_decay=0.99`，stop-grad），并在 `warmup_steps` 内使用更温和的 decay（≤0.9）以避免早期目标过僵。
- 损失在**归一化潜向量**上：`residual = normalize(online) - normalize(target); loss = residual²·sum(-1).mean()`。
- **surprise 信号归一化到 [0,1]**: `last_pred_error = ((1 - cos) · 0.5).clamp(0,1)`，供 LAVI 稳定耦合（`wm_correction = tanh(pred_error_scale)·pred_error`）；同时保留原始 MSE 量级 `last_pred_error_raw` 供调试。
- 初始化顺序修复: 先初始化 online_proj (std=0.02)，再 copy 到 target_proj；predictor 用 small-init（非零）。

**科学发现 (见 docs/reviews/V2_REVIEW.md §8)**: 实测发现 `use_ema_target=False` 分支（stop-grad + predictor + 无偏归一化投影）即 **SimSiam** (Chen & He 2021)，**本身就不坍缩**（3 seeds pairwise|cos|≈0.33）。即 EMA 对本架构的防坍缩**非必需**；EMA 在此规模下收敛速度≈SimSiam（非传言的 +25%）。`use_ema_target` 因此作为消融开关保留，默认 True。

---

### 4.5 Phase D — HebbianRegularizer ✅ 已实现 (v2.1)

**生物原型**: Hebb 法则 — "neurons that fire together, wire together"。激活模式相关的突触应当被巩固，减少灾难性遗忘。

**关于 STDP 的说明**: 原方案建议在 `mt_ltc_cell.py forward()` 里加 STDP。**不采纳**，原因:
- ProtofilamentLTC 的状态更新是连续时间衰减: `h_t = decay·h_{t-1} + (1-decay)·A_t`
- STDP 要求"突触前/后脉冲时序"，这在连续时间 LTC 里没有对应物
- 强行离散化会破坏 LTC 的核心动力学特性

**正确实现: Hebbian 损失项**（不改 forward，只改训练损失）:

```python
# mt_lnn/plasticity.py
class HebbianRegularizer(nn.Module):
    """
    Hebbian 正则项 (仅训练时启用):
    L_hebb = -α × mean(h_t ⊙ A_t)   # h=状态, A=输入投影, ⊙=逐元素乘
    
    当 h_t 与 A_t 同向时 (neurons fire together)，该项为负，
    加入总损失后鼓励这种共激活模式的权重被强化。
    
    LAVI 联动: α = base_lr × sigmoid(lavi_mean)
      - 持续模式 (高LAVI) → α 高 → 更强的记忆巩固
      - 瞬态模式 (低LAVI) → α 低 → 不在切换时过度巩固
    
    train.py 接入: total_loss += model.get_hebbian_loss()
    Config: use_hebbian=False (默认)，hebbian_lr=1e-4
    """
```

---

## 5. 数据流 (v2.0 增强版，单次 query)

```
User query
   ↓
[L1] prefill_state_only → 加载 capsule.belief_state (h_prev)
   ↓
[L0] per-token forward:
   MTLNNLayer (LTC + κ-gate + LAVI rhythm)
      ↓
   CompetitiveGWTBLayer (Phase A):
     bids = [lnn_out, attn_out, coherence_out]
     winner = softmax_competition(bids)
     broadcast(winner) → x
      ↓
   PredictiveStateHead (Phase C):
     pred_error = MSE(predict(h_t), h_{t+1})
     → LAVI 信号补充
   ↓
[L2] 每步 token → DeliberationRouter:
   ├─ CausalConsistencyChecker (Phase B):
   │    consistency = trajectory_smoothness(h_prev_window)
   │    if consistency < 0.3 → SELF_CRITIQUE
   ├─ 熵三级路由 (已有)
   └─ cloud inject (已有)
   ↓
[L3] 全程记录 (Φ, E, LAVI, causal_score, route_decision)
   ↓
[L1] capsule.save() ← belief + open_q + evidence
   ↓
Output: answer + reasoning timeline
```

---

## 6. 不变量 (Invariants)

| # | 不变量 | 守护机制 |
|---|---|---|
| I1 | Capsule ≤ 5KB | `capsule.py` 序列化大小断言 |
| I2 | 单 token 端侧延迟 < 50ms | L2 低熵路径不走 cloud；因果检测 <1ms (纯向量运算) |
| I3 | Φ 在生成全程可计算 | L3 与 L1 同步采样，不阻塞主 loop |
| I4 | 离线运行可用 | cloud router 失败必须 graceful degrade 到 self-critique |
| I5 | Evidence 可审计 | 每次 cloud inject 必须写 evidence_log |
| I6 | 新模块默认关闭 | 所有 Phase A-D 开关默认 False；零回归风险 |
| I7 | forward() 签名不变 | mt_lnn_layer.py forward / parallel_scan 不改签名 |

---

## 7. 耦合风险矩阵

```
                    config  mt_lnn_layer  gwtb    deliberation  model  train.py
─────────────────────────────────────────────────────────────────────────────────
Phase A GWT竞争      ✓(+2)              ✓扩展             ✓(+1)
Phase B 因果检测      ✓(+2)                      ✓(+1可选)  ✓(+1)
Phase C 预测头        ✓(+2)                                 ✓(+1)
Phase D Hebbian      ✓(+2)   ✓(读buffer)                         ✓(+1)
模态前端 spatial      —        —             —        —          —(仅用 inputs_embeds 入口)  —
─────────────────────────────────────────────────────────────────────────────────
风险等级:             低      极低          低      极低        低     低
```

所有改动都通过**可选参数 + 默认 False**实现，不破坏任何现有测试路径。

**`spatial.py` / `multimodal.py` 耦合 = 零**:这两个模块不触碰上表任何一列。它们
只依赖 backbone 早已稳定的 `forward(inputs_embeds=...)` 入口契约(产出 `(B,N,d_model)`
→ `fuse()` → 入模型),既不改 `config.py` 也不改 `model.py`,因此对训练路径无任何
风险。新增能力走"前端 + 融合"而非"改核心",是本仓库扩张模态能力的标准模式。

---

## 8. 取舍记录 (v2.0 新增)

### 为什么不接 Prolog/CaRing 推理引擎?
Prolog 查询延迟 10-100ms。在 token 生成路径里，这违反 I2 不变量。且触发频率 <1%，维护一个独立推理引擎的 ROI 接近零。正确方案是用 h_prev 轨迹本身作为因果信号（Phase B）。

### 为什么不在 forward() 里加 STDP?
ProtofilamentLTC 是连续时间 ODE，没有离散脉冲事件。STDP 的数学前提（突触前/后脉冲时序）在这里不存在。强行加入意味着把连续动力学离散化——破坏 LTC 的核心特性。正确方案是 Hebbian 损失项（Phase D）。

### 为什么不重新组织目录结构 (core/gwt/causality/)?
现有 40+ 文件按功能组织在 `mt_lnn/` 下，有完整的 import 路径。重组只是移文件，没有架构价值，还会破坏现有所有 import。

### 为什么不做内在情绪/价值系统?
没有可验证的神经科学对应实现，没有可量化的工程验收标准。3年以上的事情不在当前 roadmap。

---

## 9. 实施时间线

| Phase | 模块 | 关键文件 | 工作量 |
|---|---|---|---|
| ✅ 已完成 | LAVI节律门控 | `rhythm.py` | 完成 |
| ✅ A | CompetitiveGWTBLayer | `gwtb.py` (扩展) | 完成 |
| ✅ B | CausalConsistencyChecker (cosine + subspace) | `causality.py` + `deliberation.py` | 完成 (v2.1) |
| ✅ B+ | CausalActivationSteerer (STARS 启发, 子空间正交投影) | `causal_steering.py` + `spatial_reasoning.py` (可选接线) | 完成 |
| ✅ L3 | 转向接入真实解码 (generate step_callback 在线纠正 cache) | `causal_decoding.py` + `model.py` (通用钩子) | 完成 |
| ✅ L3 原型 | 因果空间转向 demo (递归回路: 检测断裂→正交投影回灌→轨迹恢复) | `examples/demo_causal_spatial_steering.py` + `test_demo_causal_spatial_steering.py` | 完成 (路演原型) |
| ✅ L2 | 空间序列记忆 (位置索引联想记忆, Hebbian 写 / 模式补全读, 复用 PlaceCellCode, 0 参数) | `mt_lnn/spatial_memory.py` + `test_spatial_memory.py` | 完成 |
| ✅ L2 原型 | 空间序列记忆 demo (写轨迹→带噪位置召回, 优雅模式补全) | `examples/demo_spatial_memory.py` + `test_demo_spatial_memory.py` | 完成 (路演原型) |
| ✅ C | PredictiveStateHead (BYOL/V-JEPA EMA) | `world_model.py` + `model.py` | 完成 (v2.1) |
| ✅ L4 | 潜空间多步想象 rollout (单步映射复合成"想象轨迹" + 置信衰减, 0参数, 不耦合 backbone) | `imagination.py` + `examples/demo_imagination.py` + `test_imagination.py` | 完成 |
| ✅ L4 | 可组合几何算子 (距离/方向/包含/邻近图/连通分量/可达性, 纯函数 0参数, 复合即推理) | `spatial_ops.py` + `examples/demo_spatial_ops.py` + `test_spatial_ops.py` | 完成 |
| ✅ L4 | 可组合牛顿动力学算子 (辛积分[半隐式 Euler + 2阶时间可逆速度 Verlet, rollout 可选, 长程能量漂移 O(dt²) 有界]/引力/碰撞冲量/盒壁反弹/守恒诊断/rollout, 纯函数 0参数, 复合即脑内物理推演) | `physics_ops.py` + `examples/demo_physics_ops.py` + `examples/demo_symplectic_integrator.py` + `test_physics_ops.py` | 完成 |
| ✅ L4 | 全局工作空间点火事件 (自适应基线+z分数+迟滞+不应期, 只读观察者 0参数, 双速引擎触发接口) | `salience_events.py` + `examples/demo_salience_events.py` + `test_salience_events.py` | 完成 |
| ✅ L4 | 可组合声学/双耳听觉算子 (传播延迟/球面扩散/ITD/ILD/多普勒/相位叠加干涉/方位反演定位, 纯函数 0参数, 复合即听觉空间推理) | `acoustic_ops.py` + `examples/demo_acoustic_ops.py` + `test_acoustic_ops.py` | 完成 |
| ✅ L4 | 可组合 Fisher-Rao 信息几何算子 (单纯形上 Bhattacharyya/距离/测地线/exp-log/平行移动/Fisher 度量/Karcher 重心, 纯函数 0参数, place codes 的正确弯曲流形度量, P0#1) | `geometry_ops.py` + `examples/demo_geometry_ops.py` + `test_geometry_ops.py` | 完成 |
| ✅ L4 | 可组合拓扑数据分析(TDA)算子 (MST/H0 持续同调条形码/Betti-0+曲线/总持续度/SRTD 拓扑漂移触发器, 复用并查集, 纯函数 0参数, 数清吸引子簇结构+拓扑失效保护信号, P0#2) | `topology_ops.py` + `examples/demo_topology_ops.py` + `test_topology_ops.py` | 完成 |
| ✅ L4 | 可组合 STDP 可塑性算子 (非对称指数 STDP 窗 + 全对全成对求和 + O(T) 在线资格迹更新[可证相等], 纯函数 0参数, 局部无反向传播的脉冲时序学习, 与 plasticity.py 并行——损失级 Hebbian 管连续 LTC 核, 事件驱动 STDP 管离散 salience 点火/L2 写入事件流, P0 学习) | `stdp_ops.py` + `examples/demo_stdp_ops.py` + `test_stdp_ops.py` | 完成 |
| ✅ L4 | 可组合吸引子/自稳定算子 (线性映射不动点/谱半径/渐近率/压缩判定[闭式] + relax 滚动 + 经验沉降时间/收敛率/Lyapunov 能量下降 + basin_radius 吸引域半宽二分探针, 纯函数 0参数, 量化沉降核收敛到何处/多快/能吸收多大扰动, P0 学习) | `attractor_ops.py` + `examples/demo_attractor_ops.py` + `test_attractor_ops.py` | 完成 |
| ✅ 落地 | 传感器摄入/流对齐算子 (把抖动/带时间戳的非均匀采样重采样到固定dt栅格[线性/ZOH]+覆盖掩码标长空洞→交盲推滑行, 纯算子 0参数, 输入侧前端, 闭合固定dt离散化与真实传感时钟的缝) | `ingest_ops.py` + `test_ingest_ops.py` (+`demo_pipeline` 摄入前端) | 完成 |
| ✅ 闭环 | 原始流式传感器前端 (P1 闭环①: 把抖动/丢帧的原始流变成 backbone-ready token——复合 ingest_ops.align_stream[对齐固定dt栅格+标丢帧步] 与可训练 ModalityProjector[投 d_model], 产 SensoryEncoding[inputs_embeds + 覆盖/pad 掩码, 即 BlindRolloutGuard 滑行的信任信号]; 可训练 nn.Module 但从不 import model.py, 经 inputs_embeds 注入) | `sensory_frontend.py` + `examples/demo_sensory_frontend.py` + `test_sensory_frontend.py` | 完成 |
| ✅ 闭环 | 自顶向下调制 (P1 闭环②, 唯一经授权动 model.py: forward 新增 top_down 形参→透传每个 block→注意力后/液态核前折入零初始化门控残差 x+=tanh(gate)·proj(LayerNorm(td)); config.use_top_down 门控默认 OFF→默认模型字节级不变; 闭门严格恒等[gate=0→逐位等价], 开门转向[不同目标推同一输入到不同下一token分布], 门有活梯度可学开门, LayerNorm 为初始值保险; 可选 top_down_to_gwtb 把目标作外部bid接入全局工作空间竞争[预留GWT对接入口, init 在 O(1e-4) softmax 伪差内恒等]) | `mt_lnn/model.py`(top_down 通路) + `mt_lnn/config.py` + `examples/demo_top_down_modulation.py` + `test_top_down_modulation.py` | 完成 |
| ✅ 落地 | 断流盲推 + 输出断路器 + 拓扑失效保护 (置信度门控盲推[借 imagination 盲滚, 失信转 DARK] + 模型外硬钳位/去抖 trip/无扰切换 + TopologyBreaker[盯表示形状: SRTD H0 漂移+可选 Betti-0, 同款去抖 FSM 跳闸/复位, 复合 topology_ops], 0参数, 不耦合 backbone) | `failsafe.py` + `examples/demo_failsafe.py` + `examples/demo_topology_failsafe.py` + `test_failsafe.py` | 完成 |
| ✅ 落地 | 双速引擎慢半边 (点火时才唤醒的多步弹道前瞻威胁评估: rollout+in_ball→突破ETA/最近接近/CLEAR-WATCH-ENGAGE等级+处置姿态, 纯算子 0参数, 仅点火付费) | `slow_layer.py` + `test_slow_layer.py` | 完成 |
| ✅ 落地 | 双速哨兵编排 (感知[声学+空间]→预测[物理惊讶]→显著度点火真唤醒慢层多步评估→盲推续命→断路器限幅, 把各层串成一个商用闭环, 编排器 0新参数, 零 model.py 耦合) | `pipeline.py` + `examples/demo_pipeline.py` + `test_pipeline.py` | 完成 |
| ✅ 闭环 | 自主认知智能体 (2-D 障碍场景里把整条类脑栈串成自主闭环: 感知[L1 网格细胞群码]→记忆[L2 SpatialMemory 认知地图]→注意[top_down 目标注入, 闭门 no-op/开门转向]→想象[L4 潜空间 rollout 0参数]+物理校验[physics_ops 碰撞/运动学逐路径筛选, 直线撞墙→绕行被提交]→盲推[BlindRolloutGuard 滑行后转 DARK]→安全[CircuitBreaker 硬钳包络]; 唯 top_down 动 model.py, 余层 0参数零耦合) | `examples/demo_cognitive_agent.py` + `test_demo_cognitive_agent.py` (复合 spatial/spatial_memory/imagination/physics_ops/failsafe) | 完成 |
| ✅ P2 | 流式持续学习 (真实 MTLNNModel 上的回放 vs 灾难性遗忘: 同序列训两遍, 朴素顺序训练彻底遗忘[最终准确率≈1/任务数, forgetting≈1.0], interleave 有界 ReservoirBuffer[Vitter 蓄水池, 对全流均匀样本]排练后几乎不遗忘[≈1.0]且新增 0 参数; 两条 T×T 准确率矩阵经 continual_eval 标准指标打分) | `examples/demo_streaming_continual.py` + `test_demo_streaming_continual.py` (复合 `replay.py` + `continual_eval.py`, 二者 0 参数零 model.py 耦合) | 完成 |
| ✅ D | HebbianRegularizer | `plasticity.py` + `train.py` | 完成 |
| ✅ 观测 | v2 模块 JSONL 指标 | `observability.py` (`v2_module_metrics`/`record_v2_metrics`) | 完成 (v2.1) |
| 🔲 E | 完整 125M 预训练 + A-D 验证 | 全栈 | 进行中 |

---

## 10. v2.0/v2.1 参数调优参考

所有开关默认 **False/关闭**，零回归。下表为已验证的安全默认与调优方向。

| Config 字段 | 默认 | 作用 | 调优提示 |
|---|---|---|---|
| `use_competitive_gwtb` | False | Phase A 多源竞争广播 | 开启后监控 `gwtb_competition_entropy`：→0 表示路由坍缩（某一源垄断），→log(K) 表示均匀；坍缩时降低 `gwtb_broadcast_init` |
| `gwtb_broadcast_init` | 0.01 | 门控残差初值 | 过大会早期主导主干，保持 ≤0.05 |
| `use_world_model` | False | Phase C 预测头 | 训练损失加 `world_model_loss_weight × L_wm` |
| `world_model_loss_weight` | 0.01 | 预测损失权重 | 极小以保证 LM loss 主导；>0.05 可能干扰语言建模 |
| `world_model_proj_ratio` | 0.5 | 潜投影维 / d_model | 更小→更强瓶颈、更省算力 |
| `world_model_ema_decay` | 0.99 | target_proj EMA 动量 | 大模型可调到 0.996–0.999；监控 `world_model_pred_error` 不应长期贴 0（坍缩）或贴 1（不学习） |
| `world_model_use_ema_target` | True | False=SimSiam 消融 | 实测两者均不坍缩；保留作对照 |
| `world_model_warmup_steps` | 1000 | EMA 温和期 | 与总步数同量级缩放 |
| `world_model_grad_clip` | 1.0 (train.py) | 预测头专属梯度裁剪 | 先于全局裁剪，防早期 surprise 爆梯度 |
| `use_hebbian` | False | Phase D 巩固正则 | 训练损失加 Hebbian 项；监控 `hebbian_signal_mean` |
| `hebbian_lr` | 1e-4 | 共激活权重 | 过大→过度巩固、灾难性偏置 |
| `hebbian_lavi_gate` | True | α 由 LAVI 门控 | 关闭则 α 恒定 |
| `causal_check_method` | "cosine" | Phase B 一致性度量 | LLM 隐态各向异性强时用 "subspace" |
| `causal_check_window` | 5 | 历史窗口 | subspace 法需 ≥2 才生效 |
| `causal_check_threshold` | 0.3 | 自我批判触发地板 | consistency < threshold → 强制 SELF_CRITIQUE |
| `CausalActivationSteerer(strength)` | 1.0 | 移除非法残差的比例 | 1.0=全投影到合法子空间;过小则纠正不足,留过多漂移 |
| `CausalActivationSteerer(floor)` | 0.3 | 转向触发地板 | 应与 `causal_check_threshold` 对齐,检测/路由/转向共用一个阈值 |
| `CausalActivationSteerer(adaptive)` | True | 增益随断裂深度缩放 | 浅跳轻推、深断用满 `strength`;False=触发即固定 strength |
| `CausalActivationSteerer(energy_keep)` | None | 合法子空间占窗口方差比例 | None=沿用 checker 的 `energy_keep`(检测器与执行器锁步);调高→子空间更大、更宽容 |
| `global_rhythm` | False | 跨层节律聚合 | 监控 `global_rhythm_scale` |

**观测建议**: 预训练中每 100 步调用 `record_v2_metrics(writer, model, step, checker)`，所有标量写入 JSONL（`world_model_pred_error`、`gwtb_competition_entropy` 等均归一化到 [0,1] 或有界），便于离线绘制坍缩/路由健康曲线。
