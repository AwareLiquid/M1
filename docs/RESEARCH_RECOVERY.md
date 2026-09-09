# 历史成果保留与训练准入

最终组件分类与研发决定见 [COMPONENT_AUDIT.md](COMPONENT_AUDIT.md)。
Physics组件另线；modern-trunk是质量工程，不是液态创新的替代路线。

核对日期：2026-09-09。Owner：Everest。
基准版本：everest-an/M1 `c3f3008`。这是成果索引与后续准入要求，不是新的实验结果。
结论冲突时以原始工件、后续裁决及 RESULTS.md 为准，不恢复已撤回宣传。

## 仓库边界

- M1：语言建模质量、ARR 质量/内存折中、持久联想记忆的稳定工程与验证。
- M2：新机制研究，包括融合式预测编码核的迁移实验；不得把小型时序结果写成语言模型能力。
- 原始代码和数据保留在 M1；跨仓使用记录来源提交，不移动或删除唯一副本。
- AwareLiquid/M1 是公开镜像。当前镜像 b792c1b 尚缺开发主线 ARR 生产者更新，不能只搬结果表。

## 保留清单

| 资产 | 来源/证据 | 已成立的范围 | 下一步，不重复旧实验 |
|---|---|---|---|
| Modern trunk | c283f8c；docs/MODERN_TRUNK.md；benchmarks/results/modern_trunk_screen_*20000*.json | all_on 三种子 PPL 73.97，但约195M，对照约144M | 同参数确认；不是直接宣布战胜 Transformer |
| ARR 混合回填 | 3e2aac4；benchmarks/arr_ratio_out/；benchmarks/results/rebuilt/ | 1:4 六种子配对优于纯 ARR；存在双峰，1:2 有发散 | 核对保存权重及学习率实际轨迹，再做单变量稳定性实验 |
| 持久快权重记忆 | RESULTS.md；benchmarks/results/parametric_memory_summary.json | 离散联想绑定、跨窗及恢复；不同任务的召回数不能混用 | 补真实会话、冲突更新、容量/干扰与现代记忆对照 |
| selective exp | 2459caf；benchmarks/results/parity_exp_param.json | 小型 parity 外推均值 exp 0.953125、tanh 0.0234375 | 保留任务专用配方；文本收益未成立，不全局启用 |
| 融合式 PC 核与生成式重放 | eb6493c；experiments/liquid_pc/model.py；report_nchain_continual.json | 五任务五种子，PC 最差任务误差0.846、GRU 1.383；双方仍明显退化 | M2 候选，先强对照与真实序列，不直接并进主干 |
| 真实 recurrence/scan | 39d2f10；tests/test_parallel_scan.py | 真实时间递归与缓存一致性；不能退回广播近似冒充递归 | 继续保留正确性护栏；chunkwise 按设备实测后决定 |

## 不能一并恢复的路线

- LoRA 归因错误的 adapter PPL 宣称；欠训练的31%优势；单seed指针满分。
- 电池不规则采样普遍优势、意识/Φ指标、全部生物模块默认开启。
- snap/STE 修复无限外推：239aeaa 已记录失败；长度课程仅是待验证方向。
- “多循环=多思考”：现有裁决没有支持；新试验必须有不同的机制假设。
- EWC、胶质门和回放全部叠加：旧实验不支持无条件叠加；通用回放必须是强对照。

## 启动训练前的准入清单

1. 保存 commit、工作区差异、完整展开模型配置、tokenizer/数据版本和校验和。
2. 固定总token、有效batch、优化器步数与microstep口径；记录实际学习率轨迹。
3. 检查已有结果和checkpoint：结果JSON不是权重，只有mixer权重也不是完整续训状态。
4. 完整续训须包含模型、优化器、scaler/scheduler（如有）、进度及数据/RNG状态，并验证连续/恢复一致性。
5. 预先固定对照、主指标、统计方法和预算。筛选结果不能取代收敛结果；双峰报告逐seed和成功率。
6. 记忆费用分列模型权重、循环状态、KV、外部存储、峰值运行内存；不能拿模块O(1)代表全模型O(1)。

## 入口漂移：已确认，尚未修改代码

- `MTLNNConfig` 三件套默认关闭；普通 `train.py` 不是 modern-trunk 确认实验入口。
- `train.py` 的 `use_predictive_coding=not args.no_predictive_coding` 会覆盖 config 的关闭默认。
- 开启 `selective_decay` 不会自动选择 exp；还要明确 `selective_decay_mode`、tau及任务配方。
- 不全局改默认，不冒充旧checkpoint可用。后续实现命名训练档时需测试：默认展开值、显式覆盖、配置落盘、恢复一致性。

## 执行次序

1. 工程：命名训练档及配置落盘、公开镜像依赖完整同步。
2. M1：modern-trunk 同参数确认；持久记忆真实会话闭环。
3. M2：融合式PC核候选复核；因果正确的新循环机制实验。
4. 小规模收益复现后才扩大模型；目前不启动2B，不重扫已完成矩阵。

本次仅固化索引和准入条件，未恢复权重、改训练默认或启动GPU任务。
