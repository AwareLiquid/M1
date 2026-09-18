# M1 夜间循环批次1 · 结案结论（2026-09-09 ~ 09-11）

> 覆盖：首航/白班/夜班/二航/三航 + 9/10 A100 执行 + 9/11 判读。
> 全部判定对照预注册标准，证据 JSON 均已入库（路径见各条目）。

## 判定账本（5 方向结案，池内 2 DONE / 2 DEAD 新增）

| 方向 | 判定 | 关键数字 | 证据 |
|---|---|---|---|
| B-1 selective_decay 预算放大 | **DEAD** | A100 32K 3-seed：387.10±3.49 vs 386.15±4.10，2/3 胜但 **p=0.5** | backlog 登记文本（A100 JSON 待白班补落） |
| B-6 PMB 参照重跑 | **DONE** | none 0 / rag 难格 **0.839** / oracle 超窗塌陷 | `benchmarks/results/pmb/`（9 JSON） |
| B-7 Falcon 更新规则 | **DEAD** | 难格 3-seed **[0,0,0]**，均值 0.000 ≤ 0.739 | `benchmarks/results/pmb_falcon/`（6 JSON） |
| B-8 t4/t5 任务族 | **DONE** | JOIN/拒答双失败模式实证 | `benchmarks/results/pmb_t45/`（18 JSON） |
| B-9 编码器敏感度 | **DONE** | t1 难格 e5 **1.0**；t4 难格 e5 **0.5-0.75** vs hash 0.0；t5 幻答 e5=hash=1.0 | `benchmarks/results/pmb_e5/`（3 JSON） |

## 三条可引用结论

1. **预算放大不拯救 selective_decay 文本增益**（B-1，A100 3-seed，p=0.5）——
   该方向按预注册 DEAD，selective_decay 的文本价值退回"小预算探针现象"。
2. **更新规则强度 ≠ 持久化语义**（B-7）——Falcon-1 在序列内 LM 任务上有竞争力
   （arXiv:2608.27763），但跨会话持久化基准上难格 recall 为 0。
3. **检索式记忆的两个结构性失败模式**（B-8+B-9 联合）：
   - **JOIN 依赖编码器**：hash 词袋 0.0 → e5 语义 0.5-0.75（跨会话拼接靠
     语义匹配，词袋锚点失效）；
   - **拒答编码器不变**：幻答率 100% 与编码器无关——evidence 模式对未知查询
     结构性无法拒答，**基准需要显式拒答协议**（t5b 方向）。

## 对两主轴的影响

- **主轴 (a) 跨会话 fast-weight 记忆**：缺口确认开放且收窄中（B-7 排除更新规则
  路径；Falcon/TTT 系 3 篇/14 个月）；PMB 已为 fastweight 备好假设检验位置
  （t4 JOIN + t5 拒答，等 GPU adapter）。
- **主轴 (b) Zoology 基准**：能力矩阵 4/5（IE/KU/JOIN/遗忘曲线）+ t5b 拒答协议
  为最后补齐点；PMB 参照一手数据齐备。

## 遗留与风险

1. 全部证据链在 PR **#58**（overnight/loop → main）待复核合并；
2. GPU 通道阻塞不变（#53 + Kaggle Secret）——B-2/B-3/B-4 判负实验挂起；
3. B-1 selective 侧 A100 结果 JSON 待白班补落（backlog 已标注）；
4. 教训自曝：多会话共写分支需开场 pull 工作分支自身（B-1 冗余重跑乌龙，
   L 工件候选）；L008（本机禁 >1h 任务）已登记。

## 池子终态

**PREREGISTERED 2（B-2/B-4，均挂 GPU）/ CANDIDATE 2（B-3、t5b 新浮现）/
DONE 3（B-6/B-8/B-9）/ DEAD 2（B-1/B-7）**
