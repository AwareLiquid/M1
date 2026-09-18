---
id: ADJ-012
date: 2026-09-18
refs: [docs/COMPUTE_TIERS.md, kb/lessons/L009-same-run-cross-time-comparison.md, kb/lessons/L012-single-seed-depth-gain-grokking-artifact.md, notes/searches/topics/compute-fleet.md]
---

# ADJ-012 — 机队定形：三级探真漏斗 + TPU lane 激活 + 跨级复用四规则

**决策**：

1. **机队组织定形为三级探真漏斗**（更新 COMPUTE_TIERS §一 语境）：
   T-本机=先验门（回答"值不值一格配额"，≤1h，L008 纪律）→
   T-Kaggle=配额计价重批（TPU∥T4 双 lane，回答"值不值配额周"）→
   T-A100=金钱计价判决（回答"值不值真钱"，逐次用户拍板）。
   术语定死三轴正交：**并行=配置之间、接力=单配置内超会话长、漏斗=层级之间**。
   本机按**单机可变**计容量（M2 Max 或 M1 Pro 手边一台，双 Mac 不是两条
   lane）；任何机器均可作控制面客户端，状态不落本机（真相源=git+远端原子 JSON）。

2. **TPU lane 激活**，撤销 §九 "20h 维持闲置" 结论：其重启条件②
   （"出现天然 XLA 友好工作负载=大 batch 静态 shape 训练"）已由 B-2″ 固定
   深度配置满足，且移植成本已付清——`train_xla` 镜像逐行等价 bundle
   `train_model`（CPU 回放 loss 逐位证明），编译税实测 ~70s/配置，单核
   0.25 s/step（medium d4，实测）。路由细则见 COMPUTE_TIERS §十。

3. **跨级复用四规则 R1-R4**（防"训了又训"，源：ASHA/Hyperband 调度语义 +
   种子纪律文献，调研记录见 notes/searches inbox 09-18）：
   - R1 权重不跨级：checkpoint 死在本级；"渐进训练"作为 treatment 时例外，
     须预注册。依据：低保真运行的价值在筛配置不在省几步；本项目实证 =
     L012（+0.054 伪影）。
   - R2 结果跨级复用：(config, seed, steps) 一周一格，逐级 resume-skip
     （既有契约，COMPUTE_TIERS §六）。
   - R3 探真产出=下一级的配置清单+标定数，不是模型。
   - R4 判决跑 seed 与探真 seed 不同源；探真数据只选配置、不进判读。
   一句话判据：复用能"少跑几个配置"→复用；只想"同一配置少训几步"→不复用。

**理由**：用户 09-18 提出机队协作问题，社区调研（SkyPilot/ASHA/种子纪律，
链接见 refs 的 topic 卡）与仓库既有件（COMPUTE_TIERS 自建路由器、resume-skip、
L008/L010/L011）对账后确认：骨架早已就位，缺的是**层级语义与复用规则的显式化**。
查重（本 ADJ 立项前）发现原拟新建 notes/decisions + notes/postmortems 属重复
建设——kb/ 三件套（H/ADJ/L）即仓库自有的 ADR/复盘体系，遂全部落既有编号系。

**约定**：
- backlog 每轮预注册新增两栏：T-本机晋级线、T-Kaggle 晋级线（预算帽+判据三级写全）；
- 判决类 run 的 seed 登记须过 R4 检查（与当期探真 seed 不同源）；
- TPU lane 用法与容量核算以 COMPUTE_TIERS §十 为准，本文件不改写 §九 原文。

**替代方案（被否）**：
- 新建 notes/decisions、notes/postmortems 目录——与 kb/decisions(ADJ###)、
  kb/lessons(L###) 重复，09-18 查重整合（本条即"先查再造"纪律的执行）；
- 引入 SkyPilot/dstack——§四 已裁定自建轻路由器，维持不变；
- 全量跨级热启动（探真权重直接续到正式跑）——R1 否决，L012 为直接反面教材。
