# 不规则采样证据线 · 下一步优先级清单与执行分工（2026-08-30）

**现状一句话**：机制线（衰减接线）已四层证据关闭；训练配方线（Δt 混合
训练）正式实验在跑（`synth_mixture_recipe`，预注册 `6bbacd8`）；中期数据
显示混合训练修复训练尺度内偏移（mt_lnn E2 0.44→0.04）并**部分**改善真外推
（E3 0.93→0.38，但对比分布内 0.03 仍有 ~10× 残余缺口）——这个残余缺口
就是架构线的唯一立项依据。

**分工**：架构同学＝只改架构代码（本文 §2 规格）；测试工程师＝拉代码跑
测试（本文 §3 手册）；维护者＝预注册/生产者脚本/归档。

---

## 1. 优先级清单

| 优先级 | 事项 | 类型 | 负责人 | 状态 |
|---|---|---|---|---|
| **P0** | **mt_lnn_rate 探针**（TIDES 式 λ=exp(−Δt·rate(x))，规格见 §2.1） | 架构 | 架构同学实现 + 测试工程师跑 | **待架构同学签发开题决定**（见 §2.1 决定记录） |
| **P1** | Transformer 连续时间位置编码（规格见 §2.2，~20 行） | 架构 | 架构同学 | 待开工，无前置 |
| P2 | 配方实验收尾：读 RECIPE VERDICT、独立复核、归档 | 测试+维护 | 维护者 | **在跑**（PID 见 HANDOFF） |
| P3 | 训练稳定性消融（15/42 配置 seed 方差失控，std/mean>0.8） | 测试 | 测试工程师（脚本由维护者备，§3.2） | 待脚本 |
| P4 | 配方落地真实域：air/battery driver 加 Δt 混合采样 | 数据管线 | 架构同学+测试工程师 | 待 P2 判定 |

**不做清单**（证据已闭环，不得重开）：Δt 特征接线（三域 1/3 判负）、
固定结构衰减（dt 探针双负）、MLP-λ 混合衰减（ad 探针双负）、
衰减去向/锚点（因果消融 t=+0.27 否决）。

---

## 2. 架构改动规格（给架构同学）

### 2.1 P0：mt_lnn_rate —— TIDES 式"物理 Δt × 可学习率"探针

**为什么是它（三条证据缺一不可）**：
1. **残余缺口真实存在**：混合训练修复了"训练尺度内"偏移（E2），但真外推
   E3（dt=1.6，4× 于训练最大尺度）仍有 ~10× 残余——配方只能内插；
2. **前沿互证**：TIDES（arXiv:2605.09742）诊断"输入依赖放在步长上会失去
   物理意义"＝我们 ad 探针失败的形态；"Δ 物理但率固定则无逐 token 表达"
   ＝我们 dt 探针失败的形态；其方案＝**率可学习、Δt 物理保留在指数上**，
   正是我们没测过的第三形态；
3. **单变量归因链干净**：与 LiquidDTRegressor 逐项一致，只改 λ 公式。

**⚠️ 开题前置——决定记录（需架构同学落款）**：2026-08-29 预注册写死
"衰减方向永久归档，不得再立第三探针"。开立本探针必须显式推翻该决定并
引用上述依据，决定记录写入 HANDOFF（两次负结果作为"特征式/MLP 式输入
依赖"的发现保留，不因本决定失效）。**未落款前不得写任何探针代码。**

**精确公式**（`benchmarks/battery_soh_edge.py` 新增 `LiquidRateRegressor`）：

    λ_{p,s,t} = exp( −Δt_t · softplus( rate_{p,s}(x_t) ) )
    rate_{p,s}: 两层 MLP，输入 = 逐 proto 摘要(seq)（P 维，同 ad 的
    lam_proj 路线），hidden 16，输出 P·S；**不喂 Δt**（Δt 只作物理乘子，
    这是与 ad 探针的本质区别——ad 是 exp(−softplus(MLP([x;dt])))，dt 当
    特征；本探针 dt 在指数上物理缩放）

**结构约束**（保持单变量归因）：
- 其余一切（P=13, S=5, blend softmax, mix MLP, norm+head, pscan, dt 从
  末通道读, clamp(min=0)）与 LiquidDTRegressor **逐项一致**；
- **初始化等价 dt 探针**：rate 输出层 bias 初始化为 softplus⁻¹(1/τ_ladder)
  （使初始 λ≈exp(−Δt/τ_ladder)，即起点≈mt_lnn_dt，训练再按内容分化率）；
- λ∈(0,1] 由构造保证（rate≥0）；rate→0 ⇒ λ→1（跨任意间隔完全保持——
  这是两种旧形态都表达不了的"拒衰减"自由度，也是外推缺口的候选解）；
- 参数量落在 0.5×~1.5× gru（74,569 @d=78,3feat）带内，超了减 hidden。

**单测清单**（`tests/test_liquid_rate.py`，对照 test_liquid_dt.py 九项）：
初始化≈dt 探针（同 seed 同参数下 bank 输出 allclose）/ rate→0 强制时
输出对 Δt 不敏感（完全保持）/ rate 收到非零梯度 / λ∈(0,1] / pscan 顺序
一致性 / 参数带 / 工厂可 patch inp/head / CPU 快速 fwd·bwd /
rate MLP 不含 Δt 输入（防回归成 ad 形态）。

**预注册规则文本**（维护者写入 `synth_ct_control.py` docstring + JSON，
**先于任何裁决运行提交**；架构同学只管实现，不得自行改判定）：
- R2''（偏移档 extrap 0.05→0.4）：mt_lnn_rate 对 gru_d AND lstm AND gru
  AND mt_lnn_dt 全部 |t|≥2；
- R1''（分布内）：≥2/3 个 cv=1 档对 gru_d AND lstm AND gru 全部 |t|≥2；
- R3''（真外推档 dt=1.6，本探针的立项目标）：对 gru_d AND lstm AND gru
  全部 |t|≥2（生产者需加 OOD 评测档，维护者备）；
- 映射：R3''+任一 → "率式自适应衰减"成立，进三域全套；仅 R3'' → 外推
  鲁棒单一卖点；皆负 → 输入依赖三形态全部证伪，衰减问题按"已穷尽"结案。

### 2.2 P1：Transformer 连续时间位置编码

**依据**：transformer 全档垫底（密集档 0.0314-0.0567，参数最多 152k 却差
最好 RNN 2-3×），根因是均匀网格位置表 + Δt 只走特征通道；前沿对应物
WrapFlow（2607.28035）"gap-aware tokens"、EvtGraph（2608.04368）。

**规格**（`TransformerRegressor`，~20 行）：
- 删 `self.pos` 均匀网格表，改为连续时间编码：`tau_t = clamp(dt,0).cumsum`
  （dt 从末通道读）→ `log1p(tau)` 的 32 频率 sin/cos 特征 → 一个可学习
  Linear 投影到 d_model（约 5K 参数，仍在带内）；
- 单测：只改 Δt 输出必变 / 参数带 / 可 patch / fwd·bwd 快。

---

## 3. 测试工程师执行手册（仿 MODERN_TRUNK §9 格式）

**前置环境**：本仓 venv（`.venv/bin/python`，torch≥2.0 numpy≥2.0），
**纯 CPU**（MPS 双进程会崩，已有两次事故）；检出分支
`iter/irregular-streaming-edge`；勿动 `mt_lnn/session_state.py` 等并行
会话 WIP 文件。

### 3.1 在跑任务确认（配方实验，P2）
进程若仍在（`pgrep -f synth_mixture_recipe`），等它跑完（总时长 ~1h），
然后 `grep "RECIPE VERDICT" benchmarks/results/synth_mixture_recipe.log`。
**只读不动**：判定与归档由维护者执行，测试工程师无需操作。

### 3.2 训练稳定性消融（P3，脚本由维护者备好后本节生效）
1. `python -m pytest tests/ -q -k "liquid or gru_d"` → 全绿才继续；
2. `python benchmarks/train_stability_ablate.py --out .../train_stability_ablate.json`
   （对照 3 个最差配置：mt_lnn_dt@dt0.05cv1、mt_lnn@dt0.15cv1、
   gru@dt0.15cv0；消融臂：恒定 lr（基线）/ cosine / 24ep+早停 / EMA）；
3. 预期产物：JSON（每配置×臂×5 seeds 的 RMSE±std）。
**判读**：哪臂把 std/mean 压到 <0.4 且 mean 不升 → 建议设为默认训练环。
**红线**：不得改模型代码；数字不进 RESULTS.md。

### 3.3 架构探针裁决（P0 实现 + 预注册 commit 之后才执行）
前置检查（缺一停止并报告）：`git log --oneline` 能看到探针实现 commit、
预注册 commit、以及 HANDOFF 中的开题决定记录；`pytest tests/test_liquid_rate.py
tests/test_liquid_dt.py -q` 全绿。

1. 冒烟（分钟级，数字无意义只验管道）：
   `python benchmarks/synth_ct_control.py --archs mt_lnn_rate,lstm --dt-means 0.15 --dt-cvs 1.0 --seeds 0 --epochs 2 --train-windows 100 --test-windows 40 --dev cpu --out /tmp/rate_smoke.json`
   → RMSE 有限、无 UNSTABLE；
2. 正式（~2.5h，显式输出路径防覆盖）：
   `nohup python -u benchmarks/synth_ct_control.py --out benchmarks/results/synth_ct_control_rate.json > benchmarks/results/synth_ct_control_rate.log 2>&1 &`
3. 判读：`grep "MECHANISM VERDICT" ...rate.log`，R2''/R1''/R3'' 与映射
   见生产者 docstring；把 JSON+log 交回维护者归档。

**红线（违反即返工）**：长训练一律 `--dev cpu` + 显式 `--out`；出结果后
不得移动阈值；数字不写 RESULTS/README（归档由维护者做）；不在运行机改
代码——报错原样报告（含栈+已产 JSON）。

---

## 4. 回填
P2 判定后：维护者按 RESULTS（配方主张按 R1+R2 结果入 Proven/Null）+
BENCHMARKS 新 §9 + HANDOFF §2.10 附记三回填；P0 判定后同路径 + 决定记录
闭环。本文档随每项状态变化更新 §1 表格。
