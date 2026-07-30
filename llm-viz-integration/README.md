# MT-LNN + LLM-Viz 集成指南

> 使用 [llm-viz](https://github.com/bbycroft/llm-viz) 3D可视化MT-LNN架构

---

## 🎯 集成方案

llm-viz是一个**3D交互式GPT可视化工具**，我们将适配它来展示MT-LNN的独特架构：
- 13根Protofilaments的微管结构
- 5个时间尺度的液态神经元
- 横向耦合和GTP门控
- 全局工作空间瓶颈

---

## 🚀 快速开始

### Step 1: 克隆并设置llm-viz

```bash
# 克隆仓库
git clone https://github.com/bbycroft/llm-viz.git
cd llm-viz

# 安装依赖（需要Node.js 18+）
npm install
# 或
yarn install

# 启动开发服务器
npm run dev
# 访问 http://localhost:3002/llm
```

---

### Step 2: 导出MT-LNN权重到llm-viz格式

创建导出脚本 `export_mt_lnn_for_viz.py`：

```python
"""
将MT-LNN模型导出为llm-viz兼容的格式
"""
import torch
import numpy as np
import json
from mt_lnn.model import MTLNN, MTLNNConfig

def export_mt_lnn_weights(model, output_path="mt_lnn_weights.json"):
    """
    导出MT-LNN权重为JSON格式，兼容llm-viz的数据加载器
    """
    weights = {}
    
    # 1. Embedding层
    weights['embedding'] = {
        'token': model.embedding.token_embed.weight.detach().cpu().numpy().tolist(),
        'pos': model.embedding.pos_embed.weight.detach().cpu().numpy().tolist() if hasattr(model.embedding, 'pos_embed') else None
    }
    
    # 2. MTLNNBlock层
    weights['blocks'] = []
    for i, block in enumerate(model.blocks):
        block_weights = {
            'layer_idx': i,
            
            # Attention部分
            'attention': {
                'q_proj': block.attn.q_proj.weight.detach().cpu().numpy().tolist(),
                'k_proj': block.attn.k_proj.weight.detach().cpu().numpy().tolist(),
                'v_proj': block.attn.v_proj.weight.detach().cpu().numpy().tolist(),
                'o_proj': block.attn.o_proj.weight.detach().cpu().numpy().tolist(),
            },
            
            # MT-DL (Microtubule Dynamic Layer)
            'mt_dl': {
                'n_protofilaments': 13,
                'n_time_scales': 5,
                
                # 多尺度共振
                'resonance': {
                    'W_in': block.mt_dl.resonance.W_in.detach().cpu().numpy().tolist(),  # (13,5,D,D)
                    'b_in': block.mt_dl.resonance.b_in.detach().cpu().numpy().tolist(),  # (13,5,D)
                    'log_tau': block.mt_dl.resonance.log_tau.detach().cpu().numpy().tolist(),  # (13,5)
                    'blend_weights': block.mt_dl.resonance.blend_weights.detach().cpu().numpy().tolist(),  # (13,5)
                },
                
                # 横向耦合
                'lateral_coupling': {
                    'type': 'RMC-style',
                    'weights': block.mt_dl.lateral.weights.detach().cpu().numpy().tolist() if hasattr(block.mt_dl, 'lateral') else None
                },
                
                # MAP门控
                'map_gate': {
                    'mlp_weights': [
                        block.mt_dl.map_gate[i].weight.detach().cpu().numpy().tolist() 
                        for i in range(len(block.mt_dl.map_gate))
                    ] if hasattr(block.mt_dl, 'map_gate') else None
                }
            },
            
            # Layer Norm
            'ln1': {
                'weight': block.ln1.weight.detach().cpu().numpy().tolist(),
                'bias': block.ln1.bias.detach().cpu().numpy().tolist() if block.ln1.bias is not None else None
            },
            'ln2': {
                'weight': block.ln2.weight.detach().cpu().numpy().tolist(),
                'bias': block.ln2.bias.detach().cpu().numpy().tolist() if block.ln2.bias is not None else None
            }
        }
        weights['blocks'].append(block_weights)
    
    # 3. GWTB (Global Workspace Theory Bottleneck)
    if hasattr(model, 'gwtb'):
        weights['gwtb'] = {
            'compression_ratio': 8,
            'compress': model.gwtb.compress.weight.detach().cpu().numpy().tolist(),
            'broadcast': model.gwtb.broadcast.weight.detach().cpu().numpy().tolist(),
            'broadcast_gate': model.gwtb.broadcast_gate.detach().cpu().item()
        }
    
    # 4. Global Coherence Layer
    if hasattr(model, 'coherence'):
        weights['coherence'] = {
            'sparsity': 0.1,
            'collapse_gate': model.coherence.collapse_gate.weight.detach().cpu().numpy().tolist() if hasattr(model.coherence, 'collapse_gate') else None
        }
    
    # 5. LM Head
    weights['lm_head'] = {
        'weight': model.lm_head.weight.detach().cpu().numpy().tolist(),
        'vocab_size': model.config.vocab_size
    }
    
    # 6. 配置元数据
    weights['config'] = {
        'vocab_size': model.config.vocab_size,
        'd_model': model.config.d_model,
        'n_layers': model.config.n_layers,
        'n_heads': model.config.n_heads,
        'n_protofilaments': model.config.n_protofilaments,
        'n_time_scales': model.config.n_time_scales,
        'max_seq_len': model.config.max_seq_len
    }
    
    # 保存为JSON
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"✅ Exported MT-LNN weights to {output_path}")
    print(f"   Vocab size: {model.config.vocab_size}")
    print(f"   Layers: {model.config.n_layers}")
    print(f"   D_model: {model.config.d_model}")
    print(f"   Protofilaments: {model.config.n_protofilaments}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # 加载训练好的MT-LNN模型
    config = MTLNNConfig(
        vocab_size=200,
        n_layers=2,  # 简化用于可视化
        d_model=832,
        n_protofilaments=13,
        n_time_scales=5
    )
    
    model = MTLNN(config)
    
    # 如果有预训练权重，加载它
    # checkpoint = torch.load('checkpoints/mt_lnn.pt')
    # model.load_state_dict(checkpoint['model'])
    
    model.eval()
    
    # 导出
    export_mt_lnn_weights(model, "llm-viz/public/mt_lnn_weights.json")
```

**运行导出**：
```bash
python export_mt_lnn_for_viz.py
```

---

### Step 3: 创建MT-LNN可视化配置

在 `llm-viz/src/llm/` 下创建 `MtLnnModel.ts`：

```typescript
/**
 * MT-LNN模型配置和布局
 * 扩展自GptModel.ts，添加微管和液态神经网络特性
 */

import { Vec3 } from '@/src/utils/vector';

export interface MtLnnConfig {
    vocabSize: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    n_protofilaments: number;  // 新增：13根原纤维
    n_time_scales: number;      // 新增：5个时间尺度
    max_seq_len: number;
}

export interface ProtofilamentLayout {
    index: number;
    position: Vec3;  // 圆形排列
    timeScales: TimeScaleNode[];
}

export interface TimeScaleNode {
    scale: number;
    tau: number;
    activation: number;
}

export class MtLnnModelLayout {
    config: MtLnnConfig;
    
    // 布局参数
    protoRadius: number = 5.0;  // 圆形排列半径
    scaleHeight: number = 0.5;  // 时间尺度垂直间距
    layerSpacing: number = 8.0; // 层间距
    
    constructor(config: MtLnnConfig) {
        this.config = config;
    }
    
    /**
     * 计算13根protofilaments的3D位置（圆形排列）
     */
    getProtofilamentPositions(layerIdx: number): ProtofilamentLayout[] {
        const positions: ProtofilamentLayout[] = [];
        const n = this.config.n_protofilaments;
        
        for (let i = 0; i < n; i++) {
            const angle = (i / n) * Math.PI * 2;
            const x = this.protoRadius * Math.cos(angle);
            const z = this.protoRadius * Math.sin(angle);
            const y = layerIdx * this.layerSpacing;
            
            // 每个proto有5个时间尺度
            const timeScales: TimeScaleNode[] = [];
            for (let s = 0; s < this.config.n_time_scales; s++) {
                timeScales.push({
                    scale: s,
                    tau: 0.01 * Math.pow(10, s), // 几何分布
                    activation: 0.0  // 运行时更新
                });
            }
            
            positions.push({
                index: i,
                position: { x, y, z },
                timeScales
            });
        }
        
        return positions;
    }
    
    /**
     * 横向耦合连接（B晶格键）
     */
    getLateralCouplingEdges(layerIdx: number): [Vec3, Vec3][] {
        const edges: [Vec3, Vec3][] = [];
        const protos = this.getProtofilamentPositions(layerIdx);
        
        for (let i = 0; i < protos.length; i++) {
            const j = (i + 1) % protos.length;  // 环形连接
            edges.push([protos[i].position, protos[j].position]);
        }
        
        return edges;
    }
    
    /**
     * GWTB瓶颈位置
     */
    getGWTBPosition(layerIdx: number): Vec3 {
        return {
            x: 0,
            y: layerIdx * this.layerSpacing + 4,
            z: 0
        };
    }
}

/**
 * 运行时状态追踪
 */
export interface MtLnnRunState {
    currentLayer: number;
    protoActivations: number[][];  // [proto][scale]
    scaleGates: number[][];         // Kappa门控值
    gtpLevel: number;               // GTP水解状态
    coherenceLevel: number;         // 全局一致性
}
```

---

### Step 4: 修改llm-viz以支持MT-LNN

在 `llm-viz/src/app/llm/page.tsx` 中添加MT-LNN模式：

```typescript
// 在文件顶部添加导入
import { MtLnnModelLayout } from '@/src/llm/MtLnnModel';

// 在组件中添加模型选择
export default function LlmPage() {
    const [modelType, setModelType] = useState<'gpt' | 'mt-lnn'>('gpt');
    
    // ... 现有代码
    
    return (
        <div>
            {/* 模型选择器 */}
            <div className="model-selector">
                <button onClick={() => setModelType('gpt')}>
                    GPT-2 Standard
                </button>
                <button onClick={() => setModelType('mt-lnn')}>
                    MT-LNN (Microtubule + Liquid)
                </button>
            </div>
            
            {/* 根据选择渲染不同模型 */}
            {modelType === 'mt-lnn' ? (
                <MtLnnVisualization />
            ) : (
                <GptVisualization />
            )}
        </div>
    );
}
```

---

### Step 5: 创建MT-LNN特定的渲染器

创建 `llm-viz/src/llm/MtLnnRenderer.ts`：

```typescript
/**
 * MT-LNN 3D渲染器
 * 使用WebGL渲染微管结构和液态神经网络
 */

export class MtLnnRenderer {
    private canvas: HTMLCanvasElement;
    private gl: WebGLRenderingContext;
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl')!;
    }
    
    /**
     * 渲染13根protofilaments为圆柱体
     */
    renderProtofilaments(layout: MtLnnModelLayout, state: MtLnnRunState) {
        const protos = layout.getProtofilamentPositions(state.currentLayer);
        
        protos.forEach((proto, i) => {
            // 绘制圆柱体
            this.drawCylinder(
                proto.position,
                0.2,  // radius
                2.0,  // height
                this.getProtoColor(state.protoActivations[i])
            );
            
            // 绘制时间尺度节点
            proto.timeScales.forEach((scale, s) => {
                const offset = { x: 0, y: s * 0.4, z: 0 };
                const pos = addVec3(proto.position, offset);
                
                this.drawSphere(
                    pos,
                    0.15,
                    this.getScaleColor(scale.tau, state.scaleGates[i][s])
                );
            });
        });
    }
    
    /**
     * 渲染横向耦合（B晶格键）
     */
    renderLateralCoupling(layout: MtLnnModelLayout, state: MtLnnRunState) {
        const edges = layout.getLateralCouplingEdges(state.currentLayer);
        
        edges.forEach(([start, end]) => {
            this.drawLine(
                start,
                end,
                0.05,  // thickness
                [0.5, 0.5, 0.5, 0.8]  // RGBA
            );
        });
    }
    
    /**
     * 渲染GTP水解动画
     */
    renderGTPHydrolysis(state: MtLnnRunState) {
        // 绘制脉冲式的能量波
        const alpha = Math.sin(state.gtpLevel * Math.PI) * 0.5 + 0.5;
        
        // ... GTP可视化逻辑
    }
    
    /**
     * 渲染全局工作空间瓶颈
     */
    renderGWTB(layout: MtLnnModelLayout, state: MtLnnRunState) {
        const pos = layout.getGWTBPosition(state.currentLayer);
        
        // 绘制沙漏形状表示瓶颈
        this.drawHourglass(pos, 2.0, 1.0);
    }
    
    // ... 辅助绘制方法
}
```

---

## 🎨 可视化效果预览

运行成功后，您将看到：

### 1. **微管结构**
- 13根圆柱体围成圆形（类似生物微管）
- 每根圆柱有5个彩色节点（代表时间尺度）
- 实时显示激活强度（颜色深浅）

### 2. **横向耦合**
- 相邻protofilaments之间的连线
- 模拟B晶格键的作用力
- 动态更新权重

### 3. **液态神经元动画**
- 5个时间尺度的激活波纹
- 从快速（蓝色）到慢速（红色）渐变
- Kappa门控的开关动画

### 4. **GWTB瓶颈**
- 沙漏形状的中心结构
- 信息流入/流出动画
- 压缩比可视化

### 5. **GTP水解**
- 周期性的能量脉冲
- 每256 token更新一次
- 类似心跳的动画效果

---

## 📁 项目结构

```
llm-viz/
├── public/
│   └── mt_lnn_weights.json          # 导出的权重
├── src/
│   ├── app/
│   │   └── llm/
│   │       └── page.tsx              # 添加MT-LNN选项
│   └── llm/
│       ├── MtLnnModel.ts             # 新增：MT-LNN模型定义
│       ├── MtLnnRenderer.ts          # 新增：渲染器
│       ├── MtLnnModelLayout.ts       # 新增：布局计算
│       └── GptModel.ts               # 原有：GPT模型
└── export_mt_lnn_for_viz.py          # 导出脚本（放在MT-LNN项目）
```

---

## 🔧 调试技巧

### 检查权重是否正确导出

```bash
# 查看JSON文件大小
ls -lh llm-viz/public/mt_lnn_weights.json

# 检查JSON结构
cat llm-viz/public/mt_lnn_weights.json | jq '.config'
```

### 启用开发者控制台

在 `llm-viz/src/llm/MtLnnRenderer.ts` 中添加：

```typescript
console.log('Rendering proto', i, 'activation:', activation);
console.log('Scale gates:', state.scaleGates);
```

### 性能监控

```typescript
performance.mark('render-start');
this.renderProtofilaments(layout, state);
performance.mark('render-end');
performance.measure('render-time', 'render-start', 'render-end');
```

---

## 🚀 下一步

1. ✅ **运行导出脚本**
   ```bash
   python export_mt_lnn_for_viz.py
   ```

2. ✅ **启动llm-viz**
   ```bash
   cd llm-viz
   npm run dev
   ```

3. ✅ **创建TypeScript文件**
   - 复制上面的代码到对应文件
   - 运行 `npm run typecheck` 检查错误

4. ✅ **访问可视化**
   - 打开 http://localhost:3002/llm
   - 选择 "MT-LNN" 模式
   - 交互探索架构

---

## 💡 高级功能

### 动画控制

```typescript
// 添加时间轴控制
interface AnimationControls {
    speed: number;           // 播放速度
    currentToken: number;    // 当前token位置
    showGTPPulse: boolean;   // 显示GTP脉冲
    highlightProto: number;  // 高亮特定proto
}
```

### 对比模式

```typescript
// 同时显示Transformer vs MT-LNN
<div className="comparison-view">
    <div className="left-panel">
        <GptVisualization />
    </div>
    <div className="right-panel">
        <MtLnnVisualization />
    </div>
</div>
```

### 导出视频

```typescript
// 使用Canvas Recorder
const recorder = new CanvasRecorder(canvas);
recorder.start();
// ... 播放动画
recorder.stop();
recorder.download('mt_lnn_animation.webm');
```

---

## 📚 参考资源

- **llm-viz原项目**: https://github.com/bbycroft/llm-viz
- **在线Demo**: https://bbycroft.net/llm
- **MT-LNN论文**: [AwareLiquid-Web `decks/mt_lnn_arxiv.pdf`](https://github.com/AwareLiquid/AwareLiquid-Web/blob/main/decks/mt_lnn_arxiv.pdf)（decks 已迁至 AwareLiquid-Web 仓库）
- **架构文档**: `ARCHITECTURE.md`

---

## 🐛 常见问题

### Q: npm install失败？
**A**: 确保Node.js版本 ≥ 18.0.0
```bash
node --version  # 应该显示v18.x或更高
```

### Q: 权重文件太大？
**A**: 只导出2层用于可视化
```python
config = MTLNNConfig(n_layers=2)  # 而不是12层
```

### Q: 渲染卡顿？
**A**: 降低protofilaments数量或简化几何体
```typescript
const LOD_LEVEL = 2;  // 降低细节级别
```

---

生成时间：2026-05-29  
版本：v1.0
