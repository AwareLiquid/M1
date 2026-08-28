# Streaming Edge Landing — 不规则采样流式边缘时序线

一页落地定位：这条产品线回答"**传感流不按固定网格来**时，端侧模型该怎么办"。
对应证据线：`benchmarks/battery_irregular_sampling.py`（NASA 电池，含 GRU-D
对照）、`benchmarks/airquality_irregular.py`（UCI 北京空气质量，自然缺失）、
`benchmarks/synth_ct_control.py`（Van der Pol 合成控制组）、
`benchmarks/streaming_edge_profile.py`（CPU/ONNX 部署画像）。
数字以 [RESULTS.md](../RESULTS.md) 与 `benchmarks/results/*.json` 为准，
本文不引入任何未入 RESULTS.md 的数字。

## 场景

| 场景 | 不规则性来源 | 为什么现有方案不顺手 |
|---|---|---|
| **BMS / 储能** | 控制器事件唤醒、负载高时丢采样、占空比变化 | MCU 预算装不下注意力 KV cache；重采样糊掉事件 |
| **预测性维护**（工业传感） | 振动/温度/电流多源不同步、掉线重连后补传 | 定长 patching 假设规则网格，gap 只能填 imputation |
| **可穿戴 / 移动健康** | 按需测量、蓝牙断连、省电间歇采样 | 云端 TSFM 无法常驻，本地模型又对 gap 敏感 |

共同结构：**流是无界的、采样是不规则的、算力是受限的**。三个条件同时成立时，
才轮到这条线的差异化说话；只成立其中两条（如规则采样 + 端侧），普通 RNN 量化后
已经够用。

## 与三类现有方案的机制对比

| 方案 | 处理不规则采样的机制 | 端侧代价 |
|---|---|---|
| **TSFM（时序基础模型）patching** | 把序列切成固定 patch——**假设规则网格**；gap 用 imputation/占位 patch 顶 | 模型大（数十 M+），patch 内时间信息被平均掉 |
| **GRU-D（Che et al. 2018）** | 可训练衰减 + mask：缺失越久，隐藏态越向 running mean 收缩 | 需要 mask 通道与衰减状态机制；仍是离散步进，对 Δt 的积分是近似的 |
| **Neural CDE / ODE-RNN** | 真正的连续时间动力学，插值路径 + ODE 求解器 | ODE 求解器迭代次数数据相关、延迟不可控，慢于端侧预算 |
| **本架构（液态核心）** | Δt 是**原生运行时输入**：时间常数 τ 直接对 elapsed time 积分，无 mask/衰减附加机制 | O(1) 流式状态 + int8 CPU 推理（见画像表） |

诚实边界：**O(1) 流式状态不是液态独有**——LSTM/GRU 同样恒定且更小；本线
主张的是"**不规则采样鲁棒性 + 常数内存 + 参数效率**"的**组合**，且该鲁棒性
是对离散 RNN 唯一有统计显著的架构级优势（见 RESULTS.md 对应行）。与
LM/长上下文无关的任何暗示都不在本线主张内。

## Honest capability card

| claim | status | evidence |
|---|---|---|
| 不规则采样下显著优于 LSTM/GRU（含 Δt 输入的公平对照） | ✔/✘ 见 RESULTS.md（GRU-D 对照后） | battery 10 seeds Welch t；air/synth ≥5 seeds |
| 对 GRU-D（canonical 不规则基线）仍保持优势 | 见 RESULTS.md（正负都如实记录） | 预注册判定：\|t\|<2 且退化差<10pp 即判负 |
| ONNX 导出 + 变 Δt 运行时输入 + ORT 数值一致 | ✔ | streaming_edge_profile：parity ≤1e-5 门（实测见 JSON） |
| int8 CPU 部署 | 见画像表（int8 体积/漂移实测） | 量化对小图未必缩小——如实报告 |
| 常数内存流式推理 | ✔ 但**非独有**（LSTM/GRU 更小） | streaming state bytes 表 |
| 规则采样下精度优势 | ✘ 不主张 | battery 10 seeds 四架构统计打平 |

## 部署画像（数字由 streaming_edge_profile.json 生成，此处同步）

<!-- PROFILE_TABLE -->

## 怎么用

```bash
# 训练对照（battery 协议模板）
python benchmarks/battery_irregular_sampling.py \
    --archs mt_lnn,lstm,gru,gru_d,transformer --drops 0.0,0.3,0.6,0.8
# 导出 + 画像
python benchmarks/streaming_edge_profile.py
```
