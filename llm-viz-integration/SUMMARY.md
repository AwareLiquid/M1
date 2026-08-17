# 🎨 MT-LNN + llm-viz 集成总结

## ✅ 已完成的工作

### 1. **创建的文件**
```
llm-viz-integration/
├── README.md                      # 完整集成指南
├── export_mt_lnn_for_viz.py      # 权重导出脚本
└── mt_lnn_weights.json           # 导出的权重文件（运行后生成）
```

### 2. **集成方案**

#### 📦 方案A：使用llm-viz可视化（推荐）

**特点**：
- ✅ 3D交互式WebGL渲染
- ✅ 实时拖拽旋转
- ✅ 逐层动画播放
- ✅ Token流向追踪

**快速开始**：
```bash
# 1. 导出MT-LNN权重
cd e:/M1
python llm-viz-integration/export_mt_lnn_for_viz.py

# 2. 克隆llm-viz
cd ..
git clone https://github.com/bbycroft/llm-viz.git
cd llm-viz

# 3. 安装依赖
npm install

# 4. 复制权重文件
cp ../M1/llm-viz-integration/mt_lnn_weights.json public/

# 5. 启动开发服务器
npm run dev

# 6. 访问 http://localhost:3002/llm
```

---

#### 📊 方案B：使用已有的matplotlib可视化（即用）

**特点**：
- ✅ 已生成PDF/PNG静态图
- ✅ 适合论文发表
- ✅ 零依赖（已完成）

**已生成的图表**：
```bash
ls fig_*.{pdf,png}

fig_architecture.{pdf,svg,png}       # 完整架构图
fig_awareness_network.{pdf,png}      # Edge-Cloud架构
fig_microtubules.{pdf,png}           # 生物学映射
```

**查看**：
```bash
# Windows
start fig_architecture.png

# 或直接在项目根目录打开图片
```

---

## 🎯 推荐使用流程

### 对于演示/分享：使用llm-viz

1. **安装Node.js** (如果没有)
   ```bash
   # 下载：https://nodejs.org/
   node --version  # 确认 >= v18.0.0
   ```

2. **克隆并运行llm-viz**
   ```bash
   git clone https://github.com/bbycroft/llm-viz.git
   cd llm-viz
   npm install
   npm run dev
   ```

3. **导出MT-LNN数据**
   ```bash
   cd ../M1
   python llm-viz-integration/export_mt_lnn_for_viz.py
   ```

4. **查看效果**
   - 访问 http://localhost:3002/llm
   - 点击播放按钮查看推理过程
   - 使用鼠标拖拽旋转3D视图

---

### 对于论文/文档：使用静态图

**直接使用已生成的图表**：

1. **架构总览图**：`fig_architecture.png`
   - 显示完整的MT-LNN流程
   - 13个protofilaments
   - 5个时间尺度
   - GWTB瓶颈

2. **生物学映射**：`fig_microtubules.png`
   - 从人脑 → 神经元 → 微管 → 液态神经网络
   - 清晰的映射关系

3. **系统架构**：`fig_awareness_network.png`
   - Edge设备 + Cloud Oracle
   - Capsule状态管理
   - 路由决策机制

**在LaTeX中使用**：
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{fig_architecture.pdf}
    \caption{MT-LNN Architecture Overview}
    \label{fig:mt-lnn-arch}
\end{figure}
```

---

## 🛠️ 如果llm-viz安装失败

### 备选方案1：在线查看llm-viz Demo

访问原作者的在线版本：
- https://bbycroft.net/llm

虽然展示的是GPT-2，但可以看到**相同的3D交互效果**，了解：
- 如何旋转视角
- 如何逐层播放
- 如何追踪token流向

---

### 备选方案2：使用Netron查看模型结构

**最简单的方式**：

```bash
# 1. 安装netron
pip install netron

# 2. 导出ONNX
python -c "
import torch
from mt_lnn.model import MTLNN, MTLNNConfig
config = MTLNNConfig(vocab_size=200, n_layers=2, d_model=832, n_protofilaments=13)
model = MTLNN(config)
dummy = torch.randint(0, 200, (1, 10))
torch.onnx.export(model, dummy, 'mt_lnn.onnx')
"

# 3. 在浏览器中查看
netron mt_lnn.onnx
```

**效果**：
- 自动打开浏览器
- 显示完整的计算图
- 点击任意节点查看参数
- 支持搜索和导出

---

### 备选方案3：使用TensorBoard

**如果训练时已集成TensorBoard**：

```bash
tensorboard --logdir=runs
# 访问 http://localhost:6006
```

查看：
- 模型结构图（GRAPHS标签）
- 训练曲线（SCALARS标签）
- 嵌入空间（PROJECTOR标签）

---

## 📚 文档汇总

### 创建的集成文档

1. **`llm-viz-integration/README.md`**
   - 完整的集成指南
   - TypeScript代码示例
   - 渲染器实现
   - 调试技巧

2. **`MT_LNN_ARCHITECTURE_VISUAL.md`**
   - 详细的架构图（ASCII art）
   - 公式推导
   - 性能对比
   - 优化亮点

3. **`VISUALIZATION_TOOLS.md`**
   - 10+个开源工具推荐
   - 每个工具的使用场景
   - GitHub链接和Star数
   - 快速开始命令

---

## 🎨 可视化效果对比

### llm-viz (3D交互式)
```
优点：
✅ 最直观，3D拖拽旋转
✅ 动画演示推理过程
✅ 适合演讲和Demo
✅ Token级别追踪

缺点：
❌ 需要Node.js环境
❌ 需要修改TypeScript代码
❌ 首次设置较复杂
```

### Netron (模型结构)
```
优点：
✅ 零代码，开箱即用
✅ 一键导出ONNX即可
✅ 适合快速查看
✅ 显示参数量和形状

缺点：
❌ 静态，无动画
❌ 不显示数值
❌ 主要看结构，不看数据流
```

### Matplotlib静态图 (已生成)
```
优点：
✅ 已经生成，直接使用
✅ 矢量图，适合论文
✅ 可自定义样式
✅ 导出PDF/SVG

缺点：
❌ 静态，无交互
❌ 2D，不是3D
❌ 需要重新运行脚本修改
```

---

## 💡 我的推荐

### 场景1：快速了解架构
**使用**：`fig_architecture.png`（已生成）
- 打开就能看
- 信息完整
- 无需安装

### 场景2：技术分享/演讲
**使用**：llm-viz（需配置）
- 最酷炫
- 互动性强
- 观众印象深刻

### 场景3：论文发表
**使用**：已生成的PDF图
- 符合期刊要求
- 矢量图高清
- 专业正式

### 场景4：调试开发
**使用**：Netron + TensorBoard
- Netron看结构
- TensorBoard看训练
- 快速定位问题

---

## 🚀 下一步行动

### 立即可做（0分钟）
```bash
# 查看已生成的图表
ls fig_*.png
start fig_architecture.png  # Windows
open fig_architecture.png   # Mac
```

### 5分钟内
```bash
# 导出ONNX用Netron查看
pip install netron
python -c "..." # 上面的导出命令
netron mt_lnn.onnx
```

### 30分钟内（如果想要3D效果）
```bash
# 设置llm-viz
git clone https://github.com/bbycroft/llm-viz.git
cd llm-viz
npm install
npm run dev
```

---

## ❓ 遇到问题？

### Q: llm-viz npm install失败？
**A**: 确认Node.js版本 >= 18
```bash
node --version
# 如果低于18，从 https://nodejs.org/ 下载最新版
```

### Q: 权重导出脚本报错？
**A**: 检查PYTHONPATH
```bash
cd e:/M1
PYTHONPATH=. python llm-viz-integration/export_mt_lnn_for_viz.py
```

### Q: 想要更简单的方案？
**A**: 直接用已生成的图片
```bash
# 所有图都在项目根目录
ls fig_*.{png,pdf}
```

---

## 📊 文件清单

```
e:/M1/
├── fig_architecture.{pdf,png,svg}        ✅ 已生成
├── fig_microtubules.{pdf,png}            ✅ 已生成
├── fig_awareness_network.{pdf,png}       ✅ 已生成
├── MT_LNN_ARCHITECTURE_VISUAL.md         ✅ 详细架构文档
├── VISUALIZATION_TOOLS.md                ✅ 工具推荐
└── llm-viz-integration/
    ├── README.md                         ✅ 集成指南
    ├── export_mt_lnn_for_viz.py         ✅ 导出脚本
    └── SUMMARY.md                        ✅ 本文件
```

---

生成时间：2026-05-29  
版本：v1.0

**建议**：先用已有的PNG图快速了解，有需要再配置llm-viz 🎯
