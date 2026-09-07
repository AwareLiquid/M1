# Streaming Edge Landing — 不规则采样流式边缘时序线

一页落地定位：这条产品线回答"**传感流不按固定网格来**时，端侧模型该怎么办"。
对应证据线：`benchmarks/battery_irregular_sampling.py`（NASA 电池，含 GRU-D
对照）、`benchmarks/airquality_irregular.py`（UCI 北京空气质量，自然缺失）、
`benchmarks/synth_ct_control.py`（Van der Pol 合成控制组）、
`benchmarks/streaming_edge_profile.py`（CPU/ONNX 部署画像）。判定由
`benchmarks/analyze_irregular_line.py` 从 JSON 复算，规则预先注册、不可事后
移动。数字以 [RESULTS.md](../RESULTS.md) 与 `benchmarks/results/*.json` 为准。

## 2026-08-29 判定之后的诚实定位（先说结论）

预注册总判定是**负**：三域里只有空气质量域成立（1/3）。battery 域被
GRU-D 吸收（\|t\|=1.75<2，且 gru_d 退化 +12.8pp 好于 mt_lnn +20.8pp）；
合成控制组（唯一可严格归因连续时间偏置的域）在高不规则档对 gru_d/lstm/gru
**无一显著**——这是对机制叙事的真实反证。因此本线**不再主张**"架构级
不规则采样优势"，只主张下面 capability card 里逐条列出的、有 JSON 背书的
具体事实。落地叙事相应收缩为："在空气质量这类多通道、自然缺失的传感预测
任务上，液态核心是实测最优且唯一打败 transformer 的架构，同时可 ONNX/
CPU/int8 部署、变 Δt 是运行时输入、流式状态恒定"。

## 场景

| 场景 | 不规则性来源 | 为什么现有方案不顺手 |
|---|---|---|
| **BMS / 储能** | 控制器事件唤醒、负载高时丢采样、占空比变化 | MCU 预算装不下注意力 KV cache；重采样糊掉事件 |
| **预测性维护**（工业传感） | 振动/温度/电流多源不同步、掉线重连后补传 | 定长 patching 假设规则网格，gap 只能填 imputation |
| **可穿戴 / 移动健康** | 按需测量、蓝牙断连、省电间歇采样 | 云端 TSFM 无法常驻，本地模型又对 gap 敏感 |

共同结构：**流是无界的、采样是不规则的、算力是受限的**。三个条件同时
成立才轮到这条线说话；注意 battery 域的实测已经表明，单凭"不规则"并不
自动偏爱液态核心——优势是否出现是任务相关的（见上，air 有、battery/
synth 无）。

## 与三类现有方案的机制对比

| 方案 | 处理不规则采样的机制 | 端侧代价 |
|---|---|---|
| **TSFM（时序基础模型）patching** | 把序列切成固定 patch——**假设规则网格**；gap 用 imputation/占位 patch 顶 | 模型大（数十 M+），patch 内时间信息被平均掉 |
| **GRU-D（Che et al. 2018）** | 可训练衰减 + mask：缺失越久，隐藏态越向 running mean 收缩 | 需要 mask 通道与衰减机制；实测它在 battery/synth 域与液态核心打得有来有回，在 air 域被显著超过 |
| **Neural CDE / ODE-RNN** | 真正的连续时间动力学，插值路径 + ODE 求解器 | ODE 求解器迭代次数数据相关、延迟不可控，慢于端侧预算 |
| **本架构（液态核心）** | 数学上原生（衰减=exp(−dt/τ)），但出厂布线里 dt 是 config 固定标量（`mt_lnn/mt_lnn_layer.py:55`）。**裁决实验已做完（2026-08-29）：把 Δt 真正接进衰减（`mt_lnn_dt` 探针，R1/R2 预注册）后，分布内 0/3 档胜出、分布偏移档输给 GRU-D——机制故事按预注册规则 CLOSED。** 本线不再主张任何连续时间机制优势 | O(1) 流式状态 + CPU 推理 26 µs/步（见画像表）；int8 在此规模不缩小（如实报告） |

诚实边界：**O(1) 流式状态不是液态独有**——LSTM/GRU 同样恒定且更小
（624–1,248 B vs 3,120 B）；本线只主张"任务实测优势 + 常数内存 + 参数
效率"的组合。与 LM/长上下文相关的任何暗示都不在本线主张内。

## Honest capability card

| claim | status | evidence |
|---|---|---|
| 空气质量多步预测（自然缺失+60% 丢采样）显著优于 gru_d/lstm/gru 与 transformer | ✔ **本域成立** | PM2.5 t=+6.43/+3.93/+7.25，TEMP t=+5.40/+2.54/+3.83（n=5，held-out Dingling） |
| 不规则采样鲁棒性是架构级规律 | ✘ **已判负**（预注册 1/3） | battery 被 GRU-D 吸收；synth 高不规则档全不显著 |
| Δt 接进衰减后机制复活（mt_lnn_dt 探针） | ✘ **CLOSED**（R1/R2 双判负） | 分布内 0/3 档；偏移档输 GRU-D（gru_d 0.2485 vs 探针 0.3005）；详见 RESULTS Null 区 |
| battery（BMS）域保持对离散 RNN 的优势 | ✘ 已撤回 | d=78 重跑：gru_d/lstm 退化均好于 mt_lnn；\|t\| 全 <2（除 lstm 2.06 边缘） |
| ONNX 导出 + 变 Δt 运行时输入 + ORT 数值一致 | ✔ | parity 5.96e-08 / 变 Δt 1.19e-07（门 ≤1e-5） |
| CPU 单核步延迟 ~26 µs、状态恒定 3,120 B | ✔（非独有） | streaming_edge_profile.json；lstm/gru 更小更快 |
| int8 缩小液态图体积 | ✘ 此规模不缩小（1,616 vs 1,480 KiB） | 动态量化 Q/DQ 节点开销 > 权重节省；lstm 缩 3.5×、gru 不变 |
| 规则采样下精度优势 | ✘ 不主张（battery 四架构打平为历史结论） | 且 synth 规则稠密档 mt_lnn 3× 优势是玩具信号，不当卖点 |

## 部署画像（d=78 / 2 层 / T=128 / Δt 运行时输入；空闲机器实测）

| arch | params | int8 KiB | 单核 µs/步 | 全核 µs/步 | 流式状态 B |
|---|---|---|---|---|---|
| mt_lnn | 55,161 | 1,616 | 26.0 | 31.2 | 3,120 |
| lstm | 99,217 | 113 | 21.7 | 23.8 | 1,248 |
| gru | 74,569 | 298 | 20.7 | 21.7 | 624 |
| mt_lnn via ORT | — | — | 64.0 | 62.7 | — |

门：ONNX parity **5.96e-08**（≤1e-5 PASS）；变 Δt（规则/突发/对数正态
抖动，同 checkpoint）**1.19e-07** PASS。每步延迟 = 一次定长前向 ÷ T
（导出图为固定窗口，占空比控制器按此批处理）。

## 怎么用

```bash
# 三域 sweep（判定规则预注册在各自 docstring）
python benchmarks/battery_irregular_sampling.py \
    --archs mt_lnn,lstm,gru,gru_d,transformer --drops 0.0,0.3,0.6,0.8
python benchmarks/airquality_irregular.py --drops 0.0,0.3,0.6
python benchmarks/synth_ct_control.py
# 判定复算（从 JSON, 不重跑训练）
python benchmarks/analyze_irregular_line.py
# 导出 + 画像（空闲 CPU）
python benchmarks/streaming_edge_profile.py
```
