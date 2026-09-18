---
id: ADJ-013
date: 2026-09-19
refs: [docs/LIQUID_ATOM.md, mt_lnn/mt_lnn_layer.py, docs/LATENT_RECURSION.md, kb/hypotheses/H001-stack-depth-monotonic.md, notes/searches/topics/frontier-lnn-fit.md, notes/overnight/backlog.md]
---

# ADJ-013 — 液态原子立宪：命名、接口冻结与演进纪律

**背景**：2026-09-19 用户命题链（外部反馈"LNN 用 Transformer 协议迭代有问题"
→ 原子工程化思想调研 → 只读审计）的结论：代码层原子早已是一等公民
（`VectorizedMultiScaleResonance`，默认 off 逐位不变 + 契约单测的演进纪律
已在 signed/selective/chunkwise/ladder 四连迭代中执行），缺的是户口层——
原子的一行方程只活在 docstring 里，接口从未宣示，主从关系含糊
（`MTLNNBlock` 自称 "transformer-style block"，即原子装在 Transformer
骨架里而非反之）。用户拍板"参考这个开始迭代"→ B-26 转正，本 ADJ 落
三件套之①（宪章），②由 docs/LIQUID_ATOM.md 承载，③保持绑定。

**决策**：

1. **原子命名**：对外叙事名定为 **τ-resonance atom**（液态时间常数共振
   原子），一行方程与"τ 为可学习参数"的宪法级分界（vs Mamba 线性 Δ 族）
   写死于 docs/LIQUID_ATOM.md §一。实现名
   `VectorizedMultiScaleResonance` 不改（bit-exact 资产与 git 谱系不动）。
   生物隐喻（微管/原纤维/MAP/GTP）降级为参数语义层，不再充当架构叙事
   主语——审计发现 80+ 文件的隐喻噪声淹没了原子本身。

2. **接口冻结**：`(B,T,d_model) → (B,T,d_model)` + 流式状态
   `(B,P,S,D)` + `ladder_iter` kwargs，即 MTLNNLayer.forward 现行签名
   （mt_lnn_layer.py:866）；深度旋钮 `set_core_iterations` /
   `set_stack_iterations` 属接口一部分。**冻结含义**：此后任何原子演进
   必须保持该契约（新参数只准加 kwargs 默认位、默认 off 逐位不变），
   等价于 residual stream 对 token mixer 社区的意义——全族可互相换件。

3. **方向声明（主从关系写死）**：原子定义不引用 attention；attention 是
   block 级可选平级件（`has_attn`/`attention_layers` 抽稀已存在）。
   "Transformer 的东西放进原子里研究"的合法槽位 = 原子内部件
   （LateralCoupling 的 RMC 式 content attention、rhythm/hebbian 挂饰）
   与 H001 core-vs-stack 对照——但这类件不入原子定义，只入族谱附表。

4. **演进纪律升格**：docs/LIQUID_ATOM.md §三 φ_τ 阶梯表（档 0-5）为原子
   迭代的唯一路线图；任何新档开跑前须绑预注册判据与敏感带域；
   rung-3（liquid_step_ladder）出数绑 H006 复活条件（B-2″ BAND_HIT），
   rung-4/5 不预注册。判据不过 = 该档如实归档，不开第二条腿。

**拒绝的替代方案**：

- 重写 block 为纯液态骨架——破坏全部 bit-exact 回归资产与在跑实验的
  可比性，收益只是叙事纯度；
- 追昇腾达芬奇/RISC-V/寒武纪等异构后端——数学上同型可达
  （对角递推+einsum），但工具链成本远超机队收益，机队口径见 ADJ-012
  （GPU/TPU/A100 三条 lane 已够）；
- 现在就写专用 CUDA/XLA 核（B-26③）——MambaOut 教训：原子域内优势
  未证先修路 = 给可能输掉的架构修路；③保持触发绑定；
- 新建一套"液态工程化思想"——审计+检索双重确认：社区 LNN 工程路线
  （动力学库+求解器）已哑火（本轮搜索全族只剩营销号噪声），创新点
  是两轴合并而非第三发明。

**承诺**：① docs/LIQUID_ATOM.md 随每次原子变体合并更新族谱行（合并窗口
检查项）；② B-26 backlog 状态改 ACTIVE 并勘误其"无人认领格"表述
（resonance mixing 格代码已在 iter/latent-recursion 实现，属"已存在、
待立户"，非"新提出"）；③ 命名清洗（对外 τ-resonance atom）只用于新
工件与文档，不回改历史数据文件。
