# 🎨 llm-viz 3D交互式可视化 - 快速启动指南

## ✅ 已完成设置

llm-viz已安装并启动！

## 🚀 访问方式

### 方法1：在浏览器中打开

```
http://localhost:3002/llm
```

**复制上面的链接，在浏览器中打开**

---

## 🎯 在llm-viz中可以看到：

### 1. **3D交互式模型** 
- 🖱️ 鼠标拖拽旋转视角
- 🔍 滚轮缩放
- 📊 实时token流向

### 2. **逐层动画播放**
- ⏯️ 播放/暂停控制
- 📍 单步调试
- 🎬 慢动作查看

### 3. **详细信息面板**
- 📈 每层的激活值
- 🔢 参数数量
- 📐 矩阵形状

### 4. **交互式高亮**
- 点击任意层查看详情
- 查看连接关系
- 追踪数据流向

---

## 📱 操作说明

### 基本控制
```
左键拖拽   - 旋转视角
右键拖拽   - 平移
滚轮       - 缩放
点击层     - 查看详情
```

### 时间轴控制
```
空格键     - 播放/暂停
→ 键       - 下一步
← 键       - 上一步
```

---

## 🎨 llm-viz默认展示GPT-2架构

当前展示的是**GPT-2的Sorting示例**，包括：
- Transformer层
- 自注意力机制
- FFN前馈网络
- Token embedding

### 与MT-LNN对比参考

| 组件 | GPT-2 | MT-LNN对应 |
|------|-------|-----------|
| Transformer Block | ✅ | MTLNNBlock |
| Self-Attention | ✅ | MicrotubuleAttention |
| FFN | ✅ | **MT-DL (13×5液态层)** ⭐ |
| Layer Norm | ✅ | ✅ |
| 额外组件 | ❌ | GWTB + O(1) WM ⭐ |

---

## 💡 如何想象MT-LNN的3D效果

在看llm-viz的GPT-2可视化时，想象：

### 1. **FFN层 → MT-DL**
GPT-2的FFN是一个简单的MLP，而MT-LNN的是：
```
FFN (2层MLP)
    ↓ 替换为
13根圆柱体（Protofilaments）
├─ 每根5个节点（Time Scales）
├─ 横向连线（Lateral Coupling）
└─ 闪烁的脉冲（GTP Hydrolysis）
```

### 2. **添加GWTB瓶颈**
在所有层之后，想象一个：
```
沙漏形状
├─ 上半部：信息汇聚
├─ 中间：窄瓶颈（8×压缩）
└─ 下半部：信息广播
```

### 3. **O(1)记忆层**
最后添加一个：
```
永久存储球体
├─ 半透明
├─ 指数衰减效果
└─ 脉冲式更新
```

---

## 🔧 技术细节

### llm-viz技术栈
```
Next.js 13     - React框架
WebGL          - 3D渲染
TypeScript     - 类型安全
三维数学        - 矩阵变换
```

### 本地服务器
```
端口：3002
路径：http://localhost:3002/llm
进程：npm run dev (后台运行)
```

### 停止服务器
```bash
# 找到进程
netstat -ano | findstr :3002

# 或直接关闭所有Node进程
taskkill /F /IM node.exe
```

---

## 📚 进阶：适配MT-LNN（可选）

如果想让llm-viz展示MT-LNN架构，需要修改TypeScript代码：

### 需要修改的文件
```
/tmp/llm-viz/src/llm/
├── GptModel.ts           - 添加MT-LNN配置
├── GptModelLayout.ts     - 添加布局计算
└── page.tsx              - 添加模型选择器
```

### 参考文档
```
llm-viz-integration/README.md - 详细集成指南
```

---

## 🌐 也可以访问官方在线Demo

如果本地有问题，直接访问原作者的在线版本：

```
https://bbycroft.net/llm
```

**优点**：
- ✅ 无需安装
- ✅ 性能优化
- ✅ 加载速度快

**缺点**：
- ❌ 无法定制
- ❌ 只有GPT-2
- ❌ 无法展示MT-LNN

---

## 🎯 下一步

1. **立即访问**
   ```
   http://localhost:3002/llm
   ```
   在浏览器中打开，体验3D可视化

2. **对比理解**
   - 看GPT-2的FFN层
   - 想象替换成MT-LNN的13×5液态层
   - 理解整体数据流向

3. **可选深度定制**
   - 参考 `llm-viz-integration/README.md`
   - 修改TypeScript代码
   - 适配MT-LNN架构

---

## ❓ 常见问题

### Q: 浏览器显示无法连接？
```bash
# 检查服务器是否运行
curl http://localhost:3002

# 或重新启动
cd /tmp/llm-viz
npm run dev
```

### Q: 想关闭服务器？
```bash
# Ctrl+C（如果在前台）
# 或
taskkill /F /IM node.exe
```

### Q: 端口被占用？
```bash
# 修改端口
cd /tmp/llm-viz
# 编辑 package.json，改为 "dev": "next dev -p 3003"
npm run dev
# 然后访问 http://localhost:3003/llm
```

---

**现在就在浏览器中打开**: `http://localhost:3002/llm` 🚀

体验3D交互式可视化，这就是您想要的效果！
