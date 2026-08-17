# 🧠 MT-LNN 架构可视化文档

> **⚠️ CORRECTION (2026-07-05) — supersedes the adapter results below.**
> The Phase 5/5b adapter numbers quoted in this document (−28.5 %/−27.7 %/−34.4 % PPL
> at "0.196 %/0.139 %/0.117 % trainable") are **retracted**: those runs predate the
> re-arm fix (`8d9d741`) — PEFT had silently frozen the MT adapters at random init,
> so **only LoRA trained**, and the quoted "trainable" counts are exactly the
> LoRA-only parameter counts. A controlled 6-config attribution confirms plain LoRA
> reproduces those PPL gains; the MT adapter adds ≈nothing on in-window perplexity.
> The architecture's real, reproducible differentiator is **cross-window recall
> through streaming state** (fast-weight memory: 0.62 accuracy where attention/LoRA
> are 0 by construction), delivered by the 7.5× smaller v2s adapter now serving.
> Authoritative results and protocols: **BENCHMARKS.md** (attribution, cross-window
> recall, out-of-window LM nulls, ARR distillation).


> **Microtubule-Inspired Liquid Neural Network Architecture**  
> 从生物微管到AI架构的完整映射

---

## 📊 架构总览图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MT-LNN Complete Architecture                      │
└─────────────────────────────────────────────────────────────────────┘

        Input Token Ids (B, T)
                 │
                 ▼
    ╔════════════════════════════╗
    ║  Token Embedding           ║  d_model = 832 = 13 × 64
    ║  + Positional Encoding     ║
    ╚════════════════════════════╝
                 │
                 ▼
    ┌────────────────────────────┐
    │   MTLNNBlock × 12 layers   │◄─── 核心重复单元
    └────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
  ┌─────────┐         ┌──────────┐
  │ Attn    │         │ MT-DL    │
  │ (MT增强)│         │ (液态)    │
  └─────────┘         └──────────┘
      │                     │
      └──────────┬──────────┘
                 ▼
    ╔════════════════════════════╗
    ║  Global Workspace (GWTB)   ║  压缩 → 广播
    ║  d_gw = d_model // 8       ║
    ╚════════════════════════════╝
                 │
                 ▼
    ╔════════════════════════════╗
    ║  Global Coherence Layer    ║  稀疏注意力 + O(1) WM
    ║  (Orch-OR Collapse)        ║
    ╚════════════════════════════╝
                 │
                 ▼
    ╔════════════════════════════╗
    ║  LM Head (Weight-Tied)     ║
    ╚════════════════════════════╝
                 │
                 ▼
        Logits (B, T, vocab_size)
```

---

## 🔬 层级1：Microtubule Dynamic Layer (MT-DL) 详解

### 生物学映射

```
生物微管 (Biological Microtubule)
├─ 13根原纤维 (Protofilaments)          → n_protofilaments = 13
├─ α/β-tubulin二聚体                    → Polarity Attention
├─ GTP水解动力学                        → GTP Gating (gamma_init)
├─ MAP蛋白调控                          → MAP Gate (MLP)
└─ 横向B晶格键                          → Lateral Coupling
```

### 核心公式

对于每个 protofilament `p` 和时间尺度 `s`：

```python
# 1. 液态时间常数更新 (Closed-form LTC)
τ_{p,s} = softplus(log_tau[p, s]) + tau_min
decay_{p,s} = exp(-dt / τ_{p,s})

# 2. 激活投影 (Per-scale input projection)
A_{t,p,s} = σ(W_in[p,s] @ x_{t,p} + b_in[p,s])

# 3. Recurrent状态更新 (Parallel Scan)
h^{p,s}_{t} = decay_{p,s} * h^{p,s}_{t-1} + (1 - decay_{p,s}) * A_{t,p,s}

# 4. 动态门控混合 (Kappa Gate)
κ_{t,p,s} = sigmoid(W_kappa @ x_{t,p})
w_{p,s} = softmax(blend_weights[p]) * κ_{t,p,s}

# 5. 跨尺度融合
h_{t,p} = Σ_s w_{p,s} * h^{p,s}_{t}
```

### 架构细节

```
┌─────────────────────────────────────────────────────┐
│           VectorizedMultiScaleResonance             │
│  (13 Protofilaments × 5 Time Scales)               │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Proto 0      Proto 1    ...    Proto 12         │
│   ┌─────┐      ┌─────┐          ┌─────┐           │
│   │ τ₁  │      │ τ₁  │          │ τ₁  │  ← Fast   │
│   │ τ₂  │      │ τ₂  │          │ τ₂  │           │
│   │ τ₃  │      │ τ₃  │          │ τ₃  │  ← Medium │
│   │ τ₄  │      │ τ₄  │          │ τ₄  │           │
│   │ τ₅  │      │ τ₅  │          │ τ₅  │  ← Slow   │
│   └─────┘      └─────┘          └─────┘           │
│      ↓            ↓                ↓               │
│   [Kappa Gate - Dynamic Channel Selection]        │
│      ↓            ↓                ↓               │
│   [Blend Weights - Scale Mixing]                  │
│      ↓            ↓                ↓               │
│   ┌───────────────────────────────────┐           │
│   │  Lateral Coupling (RMC-style SA) │           │
│   │  模拟B晶格横向作用力               │           │
│   └───────────────────────────────────┘           │
│                    ↓                               │
│   ┌───────────────────────────────────┐           │
│   │  MAP Gate (Per-proto MLP)        │           │
│   │  模拟MAP蛋白稳定性调控             │           │
│   └───────────────────────────────────┘           │
│                    ↓                               │
│               Output (B,T,P,D)                    │
└─────────────────────────────────────────────────────┘
```

### 优化亮点

| 特性 | 实现 | 收益 |
|------|------|------|
| **全向量化** | 无Python循环，einsum实现 | GPU利用率 >90% |
| **Parallel Scan** | O(N) recurrence | 替代O(N²) 循环 |
| **动态计算跳过** | Kappa gate + threshold | 推理加速 20-40% |
| **稀疏共振核** | Top-K scale selection | 训练加速 30% |
| **预测编码** | 快通道预测慢通道 | PPL -5% |

---

## 🌐 层级2：Global Workspace Theory Bottleneck (GWTB)

模拟Baars的全局工作空间理论：

```
Input (B, T, d_model=832)
        ↓
  ┌─────────────┐
  │  Compress   │  832 → 104 (8×压缩)
  └─────────────┘
        ↓
  ┌─────────────┐
  │ Workspace SA│  4-head self-attention
  │ (Bottleneck)│  信息竞争 & 融合
  └─────────────┘
        ↓
  ┌─────────────┐
  │  Broadcast  │  104 → 832
  │  + γ gating │  γ初始=0.01，可学习
  └─────────────┘
        ↓
    Residual Add
```

**作用**：强制模型将分散信息流压缩到"意识瓶颈"，模拟人脑的注意力聚焦机制。

---

## 🧬 层级3：Global Coherence Layer (O(1) Working Memory)

### 传统 vs MT-LNN

```
┌──────────────────┬──────────────────┬────────────────┐
│                  │  Transformer     │  MT-LNN        │
├──────────────────┼──────────────────┼────────────────┤
│ KV Cache大小     │  O(N)            │  O(1) 固定     │
│ 长文本内存       │  线性增长         │  指数衰减保留   │
│ 注意力密度       │  100% (O(N²))    │  10% (稀疏)    │
│ Orch-OR collapse │  ❌              │  ✅            │
└──────────────────┴──────────────────┴────────────────┘
```

### 实现细节

```python
# 1. 稀疏注意力 (Top-k保留)
scores = Q @ K^T / sqrt(d_k)
mask = top_k_mask(scores, k=int(0.1 * seq_len))
attention = softmax(scores.masked_fill(~mask, -inf))

# 2. 指数衰减工作记忆
decay_rate = sigmoid(learnable_decay)  # 初始≈0.99
K_decay = decay_rate^t * K
V_decay = decay_rate^t * V

# 3. Collapse Gate (Orch-OR启发)
collapse = sigmoid(W_collapse @ x)
output = collapse * attention_out + (1-collapse) * x
```

---

## 🎯 层级4：Deliberation Router (运行时)

```
                    Token生成
                        ↓
                  ┌──────────┐
                  │  熵计算   │
                  └──────────┘
                        ↓
            ┌───────────┴───────────┐
            │                       │
        E < θ₁                  E ≥ θ₁
            ↓                       ↓
     ┌──────────┐          ┌──────────────┐
     │ 本地直出  │          │ 语义熵检查    │
     │ (LOCAL)  │          │ (N-sample)   │
     └──────────┘          └──────────────┘
                                  ↓
                        ┌─────────┴─────────┐
                    收敛                 发散
                        ↓                   ↓
                 ┌──────────┐      ┌──────────────┐
                 │自我修正   │      │ 事实缺口检测  │
                 │(REVISE)  │      │ (检索相关度)  │
                 └──────────┘      └──────────────┘
                                          ↓
                                    缺口存在
                                          ↓
                                  ┌──────────────┐
                                  │ Cloud Oracle │
                                  │ + Quiet Inject│
                                  └──────────────┘
```

---

## 🔬 麻醉验证协议 (Anesthesia Validation Protocol)

```
正常状态 (κ=0)          部分麻醉 (κ=0.5)        完全麻醉 (κ=1.0)
     ↓                       ↓                       ↓
┌──────────┐           ┌──────────┐           ┌──────────┐
│  MT-DL   │           │  MT-DL   │           │  MT-DL   │
│  Output  │──×(1-0)─→ │  Output  │──×(1-0.5)→│  Output  │──×(1-1.0)→
└──────────┘           └──────────┘           └──────────┘
     ↓                       ↓                       ↓
 h_prev×1               h_prev×0.5              h_prev×0
     ↓                       ↓                       ↓
┌──────────┐           ┌──────────┐           ┌──────────┐
│ GWTB广播 │           │ GWTB广播  │           │ GWTB广播  │
│  ×1      │           │  ×0.5     │           │  ×0       │
└──────────┘           └──────────┘           └──────────┘
     ↓                       ↓                       ↓
  连贯输出               降质输出                高熵噪声
  Φ̂ = high             Φ̂ = medium             Φ̂ = low
```

**实验结果**：
```python
Transformer: Δ Φ̂ = 0.000  # 无响应 ❌
LNN:         Δ Φ̂ = 0.000  # 无响应 ❌
MT-LNN:      Δ Φ̂ = +7.578 # 显著响应 ✅
```

---

## 📈 性能对比可视化

### Selective Copy任务 (200K参数级别)

```
精确匹配率 (Seq-Exact Accuracy)
1.0 ┤                                         ╭─ MT-LNN
    │                                      ╭──╯
0.8 ┤                                   ╭──╯
    │                                ╭──╯
0.6 ┤                             ╭──╯
    │                          ╭──╯
0.4 ┤                       ╭──╯
    │                    ╭──╯
0.2 ┤─ Transformer ────╯
    │─ LNN ─────────────
0.0 ┤
    └────┬────┬────┬────┬────┬────┬────┬────►
        37   53   69   85  101  133  165  229  序列长度T
        
MT-LNN优势随T增长：12× (T=37) → 34× (T=101)
```

### 1.1B规模：TinyLlama + MT Adapter

```
困惑度 (PPL) - WikiText-2验证集

12 ┤
   │  ● Base: 9.161
10 ┤
   │
 8 ┤
   │                  ● +Adapter: 6.553
 6 ┤                     (-28.5%)
   │
 4 ┤
   │
 2 ┤
   └───┴───┴───┴───┴───┴───┴───►
       Base    Adapter   参数开销

Adapter参数量：2.3M (0.196% of base)
推理速度：862 tok/s (vs 959, -10%)
```

---

## 🎨 现有可视化工具

### 1. **静态架构图** (已生成)

```bash
# 运行以下脚本生成图表
python scripts/plots/plot_architecture.py     # 完整架构
python scripts/plots/plot_microtubules.py     # 生物学映射
python scripts/plots/plot_awareness_network.py # AwareLiquid系统
```

**输出文件**：
- `fig_architecture.{pdf,svg,png}` - 完整架构图
- `fig_microtubules.{pdf,png}` - 微管映射图
- `fig_awareness_network.{pdf,png}` - Edge-Cloud架构

### 2. **交互式Web UI** (`ui.html`)

实时可视化：
- ✅ 实时熵监控 (Entropy Bar)
- ✅ Capsule状态显示 (4.1KB O(1))
- ✅ 路由决策时间线 (Layer 3 Trace)
- ✅ Cloud调用追踪

```bash
# 启动演示服务器
python -m http.server 8000
# 访问 http://localhost:8000/ui.html
```

### 3. **ReasoningTrace Timeline** (`trace_timeline.html`)

推理过程回放：
- Token级别的熵变化
- 路由决策轨迹 (LOCAL/CLOUD/REVISE)
- Φ值采样点
- Cloud inject事件

---

## 🚀 推荐3D可视化组件

### 开源方案

#### **1. Three.js + React Three Fiber** ⭐⭐⭐⭐⭐

最适合展示MT-LNN的3D结构：

```javascript
// 示例：渲染13根protofilaments
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Cylinder } from '@react-three/drei'

function Protofilament({ position, timeScales }) {
  return (
    <group position={position}>
      {timeScales.map((tau, i) => (
        <Cylinder 
          key={i}
          args={[0.1, 0.1, tau]} 
          position={[0, i*0.5, 0]}
          material={{ color: heatmap(tau) }}
        />
      ))}
    </group>
  )
}

function MTLNNVisualization() {
  return (
    <Canvas>
      <OrbitControls />
      {Array(13).fill().map((_, i) => {
        const angle = (i / 13) * Math.PI * 2
        return <Protofilament 
          key={i}
          position={[Math.cos(angle)*2, 0, Math.sin(angle)*2]}
          timeScales={[0.01, 0.1, 1, 5, 10]}
        />
      })}
    </Canvas>
  )
}
```

**优点**：
- 硬件加速 (WebGL)
- 丰富的交互控件
- 可导出为视频/GIF

**资源**：
- 官网：https://threejs.org
- R3F：https://docs.pmnd.rs/react-three-fiber

---

#### **2. Plotly 3D Scatter/Surface** ⭐⭐⭐⭐

Python原生，适合数据可视化：

```python
import plotly.graph_objects as go
import numpy as np

# 可视化13×5的protofilament-scale矩阵
fig = go.Figure(data=[go.Surface(
    z=resonance_matrix,  # (13, 5) 激活强度
    x=np.arange(13),     # Protofilaments
    y=np.arange(5),      # Time scales
    colorscale='Viridis'
)])

fig.update_layout(
    title='MT-LNN Resonance Heatmap',
    scene=dict(
        xaxis_title='Protofilament',
        yaxis_title='Time Scale',
        zaxis_title='Activation'
    )
)
fig.show()
```

**优点**：
- 与PyTorch无缝集成
- 导出HTML可交互
- 适合论文附件

---

#### **3. Manim (数学动画引擎)** ⭐⭐⭐⭐⭐

制作讲解视频的终极工具：

```python
from manim import *

class MTLNNAnimation(Scene):
    def construct(self):
        # 创建13根圆柱体表示protofilaments
        protos = VGroup(*[
            Cylinder(radius=0.1, height=3)
            .shift(RIGHT * i * 0.3)
            for i in range(13)
        ])
        
        # 动画：GTP水解过程
        self.play(Create(protos))
        self.play(
            protos.animate.set_color_by_gradient(BLUE, RED),
            run_time=2
        )
        
        # 横向耦合动画
        lateral_lines = VGroup(*[
            Line(protos[i].get_top(), protos[i+1].get_top())
            for i in range(12)
        ])
        self.play(Create(lateral_lines), run_time=1)
```

**优点**：
- 3Blue1Brown同款工具
- 导出高质量视频
- 适合技术讲解

**资源**：https://www.manim.community

---

#### **4. Netron (神经网络可视化)** ⭐⭐⭐

直接可视化ONNX/PyTorch模型：

```bash
# 导出MT-LNN为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mt_lnn.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}}
)

# 在Netron中打开
netron mt_lnn.onnx
```

**优点**：
- 自动生成交互式图
- 显示参数量和形状
- 浏览器直接访问

**资源**：https://netron.app

---

#### **5. TensorBoard Graph + Projector** ⭐⭐⭐

嵌入空间3D投影：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/mt_lnn')

# 记录模型结构
writer.add_graph(model, input_ids)

# 3D嵌入可视化 (h_prev状态)
writer.add_embedding(
    h_prev.reshape(-1, d_proto),
    metadata=proto_labels,
    tag='protofilament_states'
)
```

**访问**：
```bash
tensorboard --logdir=runs
```

---

## 💡 推荐可视化方案

根据您的需求：

### **方案A：论文/演示用** (静态高质量)
1. ✅ 已有的matplotlib脚本 (PDF矢量图)
2. ➕ Manim制作动画视频
3. ➕ Plotly生成交互式HTML

### **方案B：开发/调试用** (实时监控)
1. ✅ 已有的ui.html (实时熵/路由)
2. ➕ TensorBoard (训练监控)
3. ➕ Netron (模型结构)

### **方案C：科普/讲解用** (直观易懂)
1. ➕ Three.js制作交互式网页
2. ➕ Manim制作讲解视频
3. ✅ 已有的architecture图

---

## 🎬 快速启动

### 生成所有静态图表

```bash
# 一键生成所有可视化
cd scripts/plots
python plot_architecture.py
python plot_microtubules.py
python plot_awareness_network.py
python plot_experiments.py

# 输出在当前目录：
ls fig_*.{pdf,png,svg}
```

### 启动交互式UI

```bash
# 方法1：简单HTTP服务器
python -m http.server 8000
# 访问 http://localhost:8000/ui.html

# 方法2：完整演示服务
python app.py  # Gradio界面
```

### 导出3D模型

```python
# 保存为ONNX供Netron查看
python -c "
import torch
from mt_lnn.model import MTLNN
model = MTLNN.from_pretrained('checkpoints/mt_lnn.pt')
dummy = torch.randint(0, 200, (1, 10))
torch.onnx.export(model, dummy, 'mt_lnn.onnx')
"
```

---

## 📚 参考资源

- 📄 **架构文档**：`ARCHITECTURE.md`
- 📄 **技术规范**：`SPEC.md`
- 📊 **Benchmark结果**：`BENCHMARKS.md`
- 🎨 **可视化脚本**：`scripts/plots/`
- 🌐 **交互式UI**：`ui.html`, `trace_timeline.html`

---

生成时间：2026-05-29  
版本：v2.0
