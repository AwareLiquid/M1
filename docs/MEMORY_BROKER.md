# MEMORY_BROKER — 一套记忆接口,三个后端

`mt_lnn/memory_broker/` — M1 对外暴露的**唯一**记忆接口层。存在目的:

1. **两仓串接的安全面**。Awareness-SDK(记忆产品分发仓)对本仓的调用只
   认这个包的抽象接口;本仓绝不 import 对方源码,跨仓运行时通道只有一条
   ——对方 local daemon 的 HTTP(`http://127.0.0.1:37800`,由
   `external_broker.py` 薄适配)。依赖方向保持 SDK→M1 单向。
2. **召回链路可换后端**。SDK 侧召回链路与本仓四维基准
   (`benchmarks/persistent_memory/`)用同一份 workload 打三个后端,换后端
   不改调用代码(`factory.create_broker(kind)`)。

## 接口(签名对齐两边的调用习惯)

```python
from mt_lnn.memory_broker import create_broker

broker = create_broker("parametric", d_mem=128, update_rule="sum")
broker.write("s1", key="favorite color", value="blue")   # -> 记录句柄
broker.recall("s1", query="favorite color", top_k=3)     # -> [RecallHit]
broker.forget("s1", key="favorite color")                # 单键代数擦除
broker.snapshot("s1") / broker.restore("s1", snap)
broker.state_bytes("s1")                                 # 记忆体字节数
```

方法形状沿用 `ParametricMemory` 既有契约(session 显式、query 自由文本、
top_k 限量),与 Awareness 的 `awareness_record`/`awareness_recall` 工具
形状(session/query/limit)同构。

## 三后端能力对照

| 能力 | `parametric`(快重权 (F, z)) | `graph`(知识图谱) | `external`(daemon HTTP) |
|---|---|---|---|
| 记忆体 | 每会话 (F, z) 矩阵 | 加权类型边图(逐会话一库) | 对方 daemon 的 markdown+SQLite |
| 存储随写入 | **O(1)**(state_bytes 恒定) | O(n)(db 文件,如实) | O(n)(内容字节,如实) |
| snapshot/restore | **bit-exact**(raw fp32 bytes) | **bit-exact**(库文件字节 + sha256) | ✘ **NotImplementedError**(外部存储无 bit-exact 迁移:daemon 无导出/导入端点) |
| 单键擦除 | **✔ 代数投影**(无索引,状态不变) | ✘ 仅整会话(基底无删节点 API) | ✘ daemon 无记忆删除端点 |
| 召回机制 | q→F 读 + NN 解码(余弦分) | **多跳扩散激活**(累积激活分) | cascade 检索(HTTP,无分数暴露) |
| 生命周期 | —(两条写规则 `sum`/`delta`) | **SUPERSEDES/CONTRADICTS**(基底原样透传) | daemon 自管 |
| key/value 语义 | 原生绑定 | 原生绑定(key 嵌入为节点) | 摊平成文本记录(key 入 tags+metadata) |
| 运行时依赖 | torch(已有) | torch + SQLite(已有) | 零新依赖(stdlib urllib) |

✘ 与 ✔ 都是**设计事实**,不是评价:O(1) 后端给不了无限容量,无限容量
的后端给不了 bit-exact 状态。调用方按需选型,不要靠捕获
`NotImplementedError` 做控制流。

## 诚实边界(承接 docs/PARAMETRIC_MEMORY.md,不新增主张)

- 已测数字只有 RESULTS.md 既有行:四维基准 v0(TinyLlama-1.1B,3000 步,
  3 seeds)——D1 检索 mt_v2 0.9896,D2 跨窗召回 mt_v2 0.0911(唯一非零,
  对照结构零),D4 真实快照子进程回载 0.0677,**D3 冲突消解全配置判负**
  (≤0.0013,负结果,kb/H003 记档;冲突消解不在此接口层实现,由 SDK 检测
  层兜底)。本包是协议适配层,**零算法改动**,不产生任何新数字。
- `external` 的 recall 是 workspace 级检索后按 session 过滤(daemon 搜索
  面无 session 参数),top_k 可能因此不满额——如实标注。
- `parametric` 的 `write` 返回**派生**句柄(key 的规范字符串)而非存储 id:
  (F, z) 状态里没有也不该有 id 索引(索引即 O(n) 状态)。

## 文件

| 文件 | 内容 |
|---|---|
| `base.py` | `MemoryBroker` ABC + `RecallHit` + `BrokerError`(契约) |
| `parametric_broker.py` | 包装 `mt_lnn.parametric_memory`(纯协议适配) |
| `graph_broker.py` | 包装 `mt_lnn.graph_memory`(多跳 + 生命周期) |
| `external_broker.py` | daemon HTTP 薄适配(stdlib) |
| `factory.py` | `create_broker(kind)` 注册表 |
| `consolidation_policy.py` | 固化策略纯函数(见 docs/CONSOLIDATION.md) |

## 安装(供 SDK 侧)

```bash
pip install "mt_lnn @ git+https://github.com/everest-an/M1.git"
```

`pyproject.toml` 只声明 torch+numpy(全仓重型依赖均为惰性导入,不进安装
闭包);`requirements*.lock` 仍是可复现环境的唯一真相,保持不动。
