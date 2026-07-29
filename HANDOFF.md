# MT-LNN / M1 — 会话交接文档 (HANDOFF)

> 新会话开始时：**先读这份 HANDOFF.md**，再读 `docs/ROADMAP_M2.md`（M2 战略 + P0 实验日志）和 `PUBLICATION_READINESS.md`，即可无缝接续。
> 最后更新：2026-07-29 · 分支 `main`（physics-informed-head 已并入）

---

## 0. 目标

**产品/研究方向**：MT-LNN（微管启发的液态神经网络）—— 替代 Transformer 的高效长上下文架构。
核心卖点：**O(1) 恒定工作记忆**（无 KV cache 膨胀）+ 13 通道微管液态层 + 每参数能力密度更高。

### 🎯 总目标（长期北极星）

让 MT-LNN 成为**高效长上下文架构的标杆** —— 「提到高效长上下文架构，绕不开 MT-LNN」。

- **技术上**：证明一个 O(1) 记忆的循环/液态架构，在同参数/同预算下质量-效率优于 Transformer 与主流高效注意力（Mamba/GLA/...），且能 scale。
- **落地上**：O(1) 恒定内存让长上下文能在**端侧/旧设备**跑，大幅砍掉长上下文的训练与推理成本；通过 adapter 嫁接到 Qwen/Llama 生态被采用。

### 🚩 近期目标（本阶段，3-6 个月）

**冲一篇 ICLR / NeurIPS / ICML 主会论文（Track A 工程/架构路线）**。

- **意识 / Φ / 麻醉验证降级**（争议大、证据站不住），聚焦 1-2 个证据充分的硬主张：质量-效率折中 + 长上下文。
- **可交付的证据门槛**：P0-1 多种子和 P0-2 20K 收敛训练已完成；P0-3 已补第一类强 baseline（modern Transformer），结果显示现代 Transformer 明显强于当前 MT-LNN；下一步继续补 Mamba/Mamba-2/GLA/DeltaNet、fp16 根因、scaling law、真实长上下文和效率曲线。
- 完整施工图见 `PUBLICATION_READINESS.md`（P0/P1/P2 清单）。

---

## 1. 当前状态

- 主仓库 `E:\M1`，分支 `physics-informed-head`（**公开** GitHub `everest-an/M1`）。
- 姊妹项目 `E:\O1` / `E:\O1-Anti`（**独立仓库，只读参考、绝不直接搬数**）。
- 本地 GPU：RTX 5060 Laptop **8GB**，torch 2.11.0+cu128，Python `E:\Python311`（`py -3.11`）。
- 远程训练目录：`/root/autodl-tmp/M1`（SSH：`root@tulong91.imwork.net -p 54511`）；结果已同步回本地 `E:\M1\scaling_fp32`。

> **分工现状（2026-07-19）**：本地 8GB 能做的高价值项**已做完**（P0-4 fp16 根因修复、O(1) 证据扩展到 1M、文档全面更正）。剩余项**都需要 AutoDL/A100**：P0-3 剩余强 baseline（Mamba/Mamba-2/GLA/DeltaNet）、大预算验证摘要的 14.7%、scaling law 三规模、真实长上下文任务。本地不要再尝试 20K 级长跑——8GB + 会话中断 + 与 O1 抢卡，实测反复失败。

## 2. 已完成（真实验证过）

| 项 | 结果 / 位置 |
|---|---|
| **P0-1 2K 多种子** | mt_lnn **257.15±4.89** vs transformer **373.68±8.97**（n=3, fp32, 全 stable，mt_lnn 领先约 31%，约 11σ）。JSON 在 `scaling_fp32/train_*_s*.json` |
| **P0-2 20K 收敛多种子** | mt_lnn **88.93±0.33** vs transformer **94.14±0.78** val PPL（seeds 0,1,2；20,000 steps；fp32；全 stable；n=3）。mt_lnn 相对 transformer 平均 PPL 降低约 **5.5%**。JSON 在 `scaling_fp32/converge_probe/train_*_s*.json` |
| **P0-3 modern Transformer 强 baseline** | modern_transformer **78.86±0.25** val PPL（seeds 0,1,2；20,000 steps；fp32；全 stable；n=3；144.1M 参数）。结果强于 mt_lnn **88.93±0.33** 和 simple transformer **94.14±0.78**。JSON 在 `scaling_fp32/converge_probe/train_modern_transformer_s*.json` |
| **P0-2/P0-3 日志与汇总整理** | 已将日志统一放入 `scaling_fp32/converge_probe/`：`scaling_train_20000_mt_lnn.log`、`scaling_train_20000_transformer.log`、`scaling_train_20000_modern_transformer.log`；三模型汇总表：`scaling_fp32/converge_probe/scaling_train_20000_summary.txt` |
| **checkpoint/resume** | `benchmarks/scaling_comparison.py` 已加入 `--ckpt_every N` 与 `--resume/--no-resume`；默认每 500 step 保存 `model + optim + scaler + step + cursor + RNG`，中断后可从 `out_dir/checkpoints/` 恢复 |
| **P0 训练口径固定** | WikiText-103-raw-v1，GPT-2 tokenizer，vocab 50257，seq_len=512，batch=4，lr=3e-4，`--dtype fp32` 关闭 autocast/scaler |
| **P0-4 fp16 发散根因（已修复）** | 根因：`global_coherence.py` 的 `(Q@K)/scale` 在 d_head=64 累加**之后**才缩放 → 中间乘积 ~2e5 溢出 fp16(65504) → `Inf×0`(因果掩码) = NaN → sigmoid 污染整层。修复：`(Q/scale)@K`×4 处 + `_gate_energy` 用 where/fp32累加/clamp_min。**验证**：原失败的 2000 步配方跑满，`stable: true`，val PPL **257.91** vs 同配方 fp32 **257.48**（差 0.17%，在 ±4.89 种子方差内）→ fp16 已回到 fp32 同等质量。审计：其余注意力全用 SDPA，coherence 是唯一手写的。工具：`benchmarks/diagnose_fp16_divergence.py` |
| **A2 端侧 pilot：NASA 电池 SoH（2026-07-24）** | 真实公开数据（NASA PCoE，B0005/6/7/18，与文献吻合），整块电池 B0018 留出。**三个发现**：① 规则采样下四架构精度**统计打平**（pairwise \|t\|<1，n=10），mt_lnn 用 **2.6× 更少参数**达到同等精度；② 流式状态 mt_lnn 恒定 **2.6 KB** vs transformer **34 MB@32K**（13,107×），但 **O(1) 非液态独有**，LSTM/GRU 也恒定且更小；③ **不规则采样是唯一真差异化**：80% 丢弃下 mt_lnn 仅退化 **+7.7%**，lstm **+31.1%**、gru **+32.8%**，**t=+2.40/+2.63 显著**（且已给所有架构喂 Δt，不是让 RNN 蒙眼）。**可卖定位**：唯一同时"抗不规则采样 + 内存恒定"的架构（transformer 精度追得上但内存爆，RNN 内存够但一遇丢采样就垮）。脚本：`benchmarks/battery_{soh_edge,streaming_memory,irregular_sampling}.py` |
| **ONNX 导出打通（B2 浏览器 demo 前提）** | 修了两个真实阻塞：`.contiguous().view()`→`.reshape()`（3 处 attention head-merge）+ `parallel_scan._next_pow2` 强制 int。验证：export/ORT 运行/数值 **3.58e-07** 全过；48M 投影 **~66 MB int8**，可浏览器下发。**注意**：必须用 `dynamo=False`（新导出器仍拒绝）；序列长度固定，浏览器 pad/截断。脚本：`benchmarks/check_onnx_webgpu_feasibility.py`、`export_o1_for_browser.py` |
| **O(1) 恒定内存证据扩展到 1M** | `--mode decode`：上下文 512→1,048,576（**增长 2048×**），ARR 携带状态**恒定 0.381 MB**，llama KV 线性增长到 **3,072 MB** → **8,063×**。ARR 为实测快照字节，KV 为精确解析式。**边界**：推理携带状态、仅无注意力 O 系列（非训练内存、非 hybrid）|

### P0-2/P0-3 20K 收敛结果

```text
========================================================================
SCALING TRAIN | WikiText-103 | 20000 steps | seeds [0, 1, 2]
Protocol: GPT-2 tokenizer | seq_len=512 | batch=4 | lr=3e-4 | fp32
========================================================================
arch                       params  stable     val_ppl (mean±std)   n
mt_lnn                126,041,819    True       88.93 ± 0.33      3
transformer           142,051,520    True       94.14 ± 0.78      3
modern_transformer    144,070,784    True       78.86 ± 0.25      3
========================================================================

Per-seed val_ppl:
mt_lnn               s0=89.2819, s1=88.8849, s2=88.6194
transformer          s0=94.6340, s1=94.5436, s2=93.2441
modern_transformer   s0=79.1465, s1=78.6632, s2=78.7693
```

说明：

- P0-2 已从原来的“seed 0 单跑”扩展为 **seeds 0,1,2 三种子完整结果**。
- mt_lnn 在同一 20K/fp32/WikiText-103 口径下稳定优于自建 simple transformer baseline（PPL 降低约 **5.5%**）。
- P0-3 的 modern_transformer 是 RoPE + RMSNorm + SwiGLU baseline，结果 **78.86±0.25**，比 simple transformer 低约 **16.2%**，比当前 mt_lnn 低约 **11.3%**。这说明原 simple baseline 偏弱，论文主张不能再写成“MT-LNN 在 PPL 质量上优于强 Transformer baseline”；当前更稳妥的主线应转向“质量差距待优化 + O(1) working memory / 长上下文效率优势”。
- O1 的 214.8 是 84M/3000步/AMP/不同项目口径，**只能作为参考锚，不能混入 M1 论文主表做直接对比**。

## 2.5 M2 路线:思考深度研究(2026-07-28/29,进行中)

战略文档 `docs/ROADMAP_M2.md`:**2B 推理引擎,四轴对标 70B base**(数学/代码推理、
长流式记忆、持续学习、端侧延迟)。知识外置 RAG,本体只做推理与记忆控制。
完整实验日志(六轮,含全部负结果)在该文档 §4.5,**数字以文档为准,勿凭记忆复述**。

### 已落地的代码(全部测试通过,已推 main)

| 机制 | 位置 | 状态 |
|---|---|---|
| `core_iterations` 液体核心潜空间循环 | `mt_lnn/model.py` MTLNNBlock + `config.py` | ✅ 零回归,N=1 位等价 |
| `stack_iterations` 块级循环(注意力+LNN 权重绑定重复) | `mt_lnn/model.py` MTLNNModel | ✅ 零回归,use_cache 守卫 |
| `n_global_heads` 全局头配额(架构原则 #1) | `mt_lnn/mt_attention.py` + `config.py` | ✅ 默认 0 位等价,待 sweep 定默认值 |
| 深度敏感基准(单环指针追踪+模运算链) | `benchmarks/reasoning_tasks.py` | ✅ 已封死"抄起点"捷径 |
| 深度实验框架(fixed/anytime/mix/消融旋钮) | `benchmarks/reasoning_depth.py` | ✅ 结果落 `benchmarks/results/reasoning_depth.jsonl` |

### 核心发现(改变 M2 设计方向)

1. **只循环液体核心 ≠ 思考**:组合查找的计算在注意力里,LNN 子层迭代深度全平
2. **GTP 距离衰减 init 是关系推理的阻断器**(主因)+ GQA 协同:修复后 8 节点
   指针追踪 0.25 → **1.0000 满分**(transformer 对照 0.9932)。液体子层本身无罪
3. **随机深度训练教模型无视迭代**;固定深度或深监督才可能起作用
4. **基准必须做作弊者分析**:随机置换图存在 f^k(s)=s 捷径(理论/实测逐位吻合),已改单环封死

## 2.7 官网与产品表面(2026-07-30,已上线)

线上 `awareliquid.ai` 已部署到 `45c4b72`。生产是一台 Vultr 机器(**地址与登录方式不写在
本公开仓库**,见 §3.7),仓库在 `/root/M1`,三容器 `mtlnn_prod` / `mtlnn_adapter` / `caddy_prod`。
**`serve/static` 是只读 bind-mount,改 HTML/CSS/JS 只要 `git pull` 即生效,不用重建;
改 `serve/server.py` 才需要 `docker compose -f deploy/docker-compose.prod.yml up -d --build mtlnn`。**

### 已完成

| 项 | 内容 |
|---|---|
| **撤回泄漏修复** | `llms.txt` / `llms-full.txt` 是 `robots.txt` 指给 GPTBot/ClaudeBot/PerplexityBot 的文件,却仍把 −28.5% 写成 "Key Verified Results / validated / Built / independently verified"。人类在 `/research` 看到已撤回,爬虫读到的却是旧版——而后者才是被模型引用的那份。已同步撤回标注,`+13.3pp` 重新定性为 prompt-template 效应,基座笔误 Qwen-3B → Qwen-1.5B,版本号 bump 到 2.3 |
| **4 个用户可见 bug** | `finally{}` 把 `[error 503]`/`[Model backend offline]`/`[connection lost]` 无条件覆盖成 `[no output]`(return 也走 finally);404 处理器把 JSON API 错误返回整页 HTML,SDK 收到 `SyntaxError: Unexpected token '<'`;`/api` 是死链;`/robots.txt` 和 `/sitemap.xml` 生产上 404(文件在 `static/` 但 `StaticFiles` 只挂 `/static`,**整站 SEO 配置从未生效过**) |
| **编辑式改版(8 页)** | 参照 liquid.ai 的**结构语言**:系统 serif 标题 weight 400 + 负字距、1px 发丝线代替卡片、mono uppercase 元数据标签、控件 4px 圆角其余为 0、只有 hover 动效。**配色刻意不同**——他们单一紫 `#7c3aed`,我们纯黑白灰,这是名称近似下最强的区分手段 |
| **公司主体** | 8 个页面页脚均含 `Shenzhen Santi Anyuan Technology Co., Ltd`;`index` / `demo` 跟随语言切换显示「深圳三体暗源科技有限公司」 |
| **另外 7 个 bug** | privacy/terms 访问一次即把全站主题永久钉死;about/research 语言按钮把 `<html lang>` 谎报成 `zh-CN`;research 侧栏高亮观察全部 22 个 `[id]`(12 个非章节)导致高亮被清空;research 宽表被 `overflow-x:hidden` 裁切;404 `rel=canonical` 指向首页;demo `probe()` 无超时;demo 切模型失败静默(显示 M1 实际打到 O1) |

### 行业案例(6 块,只有 2 块有数据)

| 块 | 状态 | 依据 |
|---|---|---|
| 电池 | **Measured** | NASA PCoE,10 seeds,整颗 B0018 留出。80% 丢采样 +7.7% vs LSTM +31.1% / GRU +32.8%;2.6 KB vs 34 MB@32K |
| 金融文档 | **Measured** | `AwareLiquid/AwareLiquid-M2`,44/48 = 91.7%,约 1.3 次调用 / 2.8k tokens 每题。**M2 是检索+压缩适配器,不是 O 系列,不可混为一谈** |
| 耳机 | Code shipped — untrained | `AwareLiquid/O1-Sound`,见 §2.8 |
| 车机 / 手机 / 工程机器人 | Target application | **无任何实验**,写的是场景 + 架构适配理由,明写 "No pilot data yet" |

## 2.8 O1-Sound(2026-07-30,已推送,无权重)

仓库 `AwareLiquid/O1-Sound`(公开,MIT)。多语种问候唤醒词,O 系列液态核心。

**已实测**:ONNX 导出 **5.03 MB fp32 / 1.27 MB int8**,携带状态 **5,120 B 且不随流长增长**,
流式 `step()` 与批处理路径数值一致 **3.7e-09**,10 个测试通过(含体积门禁超预算会真的 fail)。
默认 `hidden=640, layers=2` = 1.30M params。τ 用 `softplus(log_tau)+tau_min` 几何初始化覆盖 **10–240 ms**。
导出的是**单帧 step 图**而非定长窗口。

**没有的**:权重。所以没有 FRR / FAR,官网标 `Code shipped — untrained`(灰框,非 Measured 黑框)。
`o1sound/keywords.py` 里 20 个语种是 **spec 不是结果**;`train.py` 会按名字警告 spec 中磁盘上不存在的语种。

## 3. 进行中 / 卡点

- **P0-3 强 baseline 正在进行中**：modern_transformer 已完成；Mamba/Mamba-2/GLA/DeltaNet 等现代高效架构仍需继续跑。当前脚本已支持 `mamba`，但 Windows/无 CUDA kernel 环境的速度结果不能用于论文效率对比；强 baseline 建议继续在 Linux + CUDA kernel + A100/AutoDL 上跑。
- **P0-3 modern_transformer 阶段成果已推送**：`benchmarks/baselines.py` 增加 `ModernCausalTransformer`；`benchmarks/scaling_comparison.py` 增加 `modern_transformer` arch；`scaling_fp32/converge_probe/scaling_train_20000_summary.txt` 已更新为三模型对比；modern_transformer 三个 JSON 与三份标准化日志已上传到 `physics-informed-head`。接手时仍需先 `git status` 确认本地是否有新实验结果或远端同步差异。
- ~~**fp16/AMP 根因未解决**~~ → **已解决（2026-07-19）**：根因是 `mt_lnn/global_coherence.py` 的注意力缩放顺序 `(Q@K)/scale`——在 d_head=64 维累加**之后**才缩放，Q/K 增大后中间乘积 ~2e5 在矩阵乘内部溢出 fp16（上限 65504），产生的 Inf 与因果掩码零相遇触发 `Inf*0=NaN`，经 sigmoid 污染整层。修复：改为 `(Q/scale)@K`（4 处）+ `_gate_energy` 用 where 代替乘法、fp32 累加、`clamp_min(1e-6)` 替换在 fp16 下下溢成 0 的 `1e-9`。**验证**：原本第 875 步发散的同配方现已跑满 **2000 步** `stable: true`，val PPL **257.91**，与同配方 fp32 的 257.48 相差仅 0.17%（远小于 ±4.89 种子方差）——fp16 已恢复到 fp32 同等质量。审计确认：其余注意力实现均用 SDPA，coherence 是唯一手写的。诊断工具：`benchmarks/diagnose_fp16_divergence.py`。
- **⛔ O1 48M 权重找不到（阻塞浏览器 demo）**：本地仓库只有 M1 adapter；HuggingFace `EverestAn/MT-LNN` 也只有 `llama_mt_adapter_000500.pt`（4.11 MB）+ PDF，**没有 O1 48M**。`checkpoints/` 和 `*.pt` 被 gitignore。最可能在**已被禁用的 Modal workspace**（`ac-ESq0Y6MGgrCtt67tOwrcDS`，官网 demo 的 `/adapter/v1/model` 因此 404）或服务器上。**拿到权重后一条命令即可**：`py -3.11 benchmarks/export_o1_for_browser.py --ckpt <path> --int8`
- **scaling law 未完成**：还需要至少 3 个模型规模，统一 token budget、训练步数/样本量和 eval 口径，确认优势是否随规模保持。
- **长上下文证据仍需补齐**：O(1) working memory 的核心卖点需要 decode/profile/真实任务支撑，不能只靠 WikiText PPL。

## 3.7 官网 / O1-Sound 未完成项(2026-07-30 交接)

### ⛔ 最高优先:改版从未经人眼看过

8 个页面改版 + 部署全程**没有做过视觉确认** —— 整个会话里 Chrome 工具都连不上。
上线的是**结构正确但没人看过**的版本:标签栈平衡、CSS 括号配平、inline JS 过
`node --check`、i18n 55/55 对称、8 条路由 200,这些都验过;**渲染出来长什么样没验过**。
用户反馈过一次「模块间距不太对」,根因(`.section` 同特异性覆盖)已修,但**修完的效果同样没看过**。

**接手第一件事:开浏览器把 8 个页面走一遍**,尤其间距,以及 serif 标题在 Windows 上的落地
——`--font-serif` 首选 Iowan Old Style 是 macOS 字体,Windows 会落到 Palatino/Georgia,
中文落到宋体/思源宋体,两边观感可能差别很大。

### O1-Sound:训练没跑

官网那块要翻成 Measured,必须先有真数字:

```bash
python scripts/fetch_mswc.py --languages en,de,fr,es,it,pt,pl,ru,tr,id --out data/mswc
python train.py --root data/mswc --epochs 20 --out checkpoints/o1sound.pt
python eval.py  --ckpt checkpoints/o1sound.pt --root data/mswc --split test --out results/test.json
python export_onnx.py --ckpt checkpoints/o1sound.pt --out dist/o1sound.onnx --int8
```

跑在**本地 RTX 5060**(服务器是 4 vCPU 纯 CPU,不合适)。模型只有 1.3M 参数,几十分钟量级。
`eval.py` 报的是**固定 FAR 预算下逐语言的 FRR**,并单独打印最差语种 ——
**多语言声明的上界是最差语种,不是平均值**,填官网时用那个数。

### 官网仍未修的 bug(按严重度)

| 级 | 位置 | 问题 |
|---|---|---|
| **P1** | `serve/server.py:340` | `if token and ...` 短路语义 = **`PARTNER_STATS_TOKEN` 没设就完全不鉴权**。`docker-compose.prod.yml` 里默认空串,`.env` 忘填 → `GET /partners` 全公开。应改成 prod 强制要求 token |
| **P1** | `index.html` i18n | `demo.fullpage` / `footer.brand` / `footer.built` **中文翻译写好了但没进 `I18N_MAP`,永远不会应用**;另有 60+ 可见文案根本不在映射表内(stat 标签、feature card、rh-table、future bullets、vs 对比表、footer 链接、模型下拉)。切中文后半中半英 |
| **P2** | `index.html` / `demo.html` | `marked.parse()` 无 sanitize,CSP 含 `'unsafe-inline'` → 诱导模型输出 `<img src=x onerror=…>` 即执行。fallback shim 更糟:只转义代码块,普通段落直接拼 |
| **P2** | `llms.txt:29` / `research.html:592` | **同一个 O(1) 数字有两个版本**:`4 KB`(正文)vs `0.381 MB`(meta/JSON-LD/首页),差近 100 倍。两者可能指不同模型规模,但**未标注就是自相矛盾**,需确认各自对应什么配置再统一 |
| **P2** | `api.html` | 缺 canonical、og 标签;主题只有 1 处 `prefers-color-scheme`,浅色模式下仍可能突兀 |
| **P2** | `server.py:304` | `counts.json` 非原子写(先 truncate 再 dump),被 `docker stop` 打断 → 下次 `json.load` 抛 ValueError → `except` 静默把累计计数**归零** |
| **P2** | `server.py:172` | 全局单锁串行化所有生成,无超时无队列上限。CPU 2–15 tok/s × 400 token ≈ 单请求最长 200s,第二个用户一直挂着 |
| **P2** | `server.py:600` | `/v1/model` 不返回 `base_model`/`adapter_loaded`/`is_baseline`,前端 `applyModel` 依赖它们 → 状态栏永远显示 `MT-LNN · 48M · cpu`,「· O1」徽标永不出现 |
| **P2** | `demo.html:8` vs `sitemap.xml` | demo 页 `noindex` 但 sitemap 以 priority 0.8 收录 |

完整审计是 4×P0 / 10×P1 / 24×P2,另有 17 类明确验证为干净(childNodes 下标依赖全站 0 处、
重复 id 0 处、JSON-LD 全合法、站内锚点 0 死链、本地资源 0 死链)。上面只列**仍未修**的。

### 部署访问

**部署凭证不在本仓库,也不会进本仓库** —— 这是公开仓库。服务器地址、SSH 密钥路径
和吊销方式记在本地 `DEPLOY_ACCESS.local.md`(已被 `.gitignore` 排除),接手时向
项目所有者索取。

`serve/static` 是只读 bind-mount:改 HTML/CSS/JS 只要在服务器上 `git pull` 即生效;
改 `serve/server.py` 才需要
`docker compose -f deploy/docker-compose.prod.yml up -d --build mtlnn`(耗时数分钟,
SSH 前台会超时,用 `nohup ... &` 丢后台再 `tail /tmp/deploy.log`)。

### 踩过的坑(别再踩)

1. **`.gitignore` 的 `data/` 是任意层级匹配**。O1-Sound 首次推送时它吞掉了源码包
   `o1sound/data/`,远端仓库 `import` 直接失败,而本地工作区一切正常。
   **推送后必须重新 clone 一份跑测试**,不能信本地。要锚定根目录就写 `/data/`。
2. **`.section{padding:96px 0}` 与 `.industries-section{...}` 同特异性**,后者在前、
   前者在后 → 后者胜出,设的节奏静默失效。改 CSS 节奏时把值放在**最通用的那条规则上**,
   不要逐个 section 覆盖。
3. **Edit 工具会把 LF 文件写成 CRLF**(index.html / demo.html 中过招)。批量改完
   `git diff --stat` 若出现整文件改动,先查行尾。

## 3.5 资源缺口分析(2026-07-29 — 回答"M1 目前缺哪块")

按阻塞程度排序:

| # | 缺口 | 现状 | 需要什么 |
|---|---|---|---|
| **1** | **训练算力(最大瓶颈)** | 本地 8GB 只够 200K 级探针实验;P0-3 剩余强 baseline(Mamba-2/GLA/DeltaNet)、14.7% 大预算复核、scaling law 三规模、M2-P1 蒸馏(350M~1B)**全部堵在这里** | AutoDL/A100 预算(估 P0-3 收尾 ~¥300-500;P1 蒸馏首轮 ~$300-500)或同事的卡 |
| **2** | **推理训练数据** | M2-P1 蒸馏需要强教师的推理轨迹(数学/代码 CoT),目前管线代码和数据都是零 | 教师模型 API 预算 + `benchmarks/` 下建蒸馏数据管线(CPU 工作,可先行) |
| **3** | **P0 收尾实验(GPU 排队中)** | 单环任务过夜实验在跑;`n_global_heads` sweep(0/1/2/4)排队——定默认值必需 | 只需本地 GPU 时间,无需外部资源 |
| **4** | **生物模块 ablation 补课** | GWT/PC/睡眠的贡献未按 5-seed 纪律量化;Hebbian 已知惰性待改 fast-weights 或删 | 本地 GPU + 时间,优先级低于 1-3 |
| **5** | **O1 48M 权重仍失踪** | 阻塞浏览器 demo(见 §3 ⛔ 条目) | 人工找回(Modal workspace 或服务器) |

**给人类队友的建议分工**:算力(#1)和教师 API(#2)是钱能解决的;#3/#4 我(CC)在本地
逐个排队跑;#5 需要你找回权重。如果只解锁一项,**先解锁 #1**——它同时打开论文
(P0-3/scaling law)和 M2-P1(蒸馏)两条线。

## 3.6 分工卡(2026-07-29,按人领任务)

三方并行、互不阻塞。**协作纪律**:所有结果落 repo(JSON+log,不落聊天记录);
checkpoint `.pt` 不提交;O1 数字绝不进 M1 主表;本地 8GB 归 CC 排队使用,勿并行抢卡。

### 🧑‍💻 技术同事(AutoDL/A100)— 大预算 GPU 线,按序执行

| 序 | 任务 | 怎么跑 | 交付物 |
|---|---|---|---|
| T1 | **P0-3 mamba 三种子** | §6 现成命令,SSH 后照抄(Linux 有 CUDA kernel,速度数据才可用于论文) | `train_mamba_s{0,1,2}.json` + `run.log` 同步回 `scaling_fp32/`,格式照 converge_probe 现有文件 |
| T2 | **14.7% 大预算复核**(头号学术风险) | 100K 步 A100 口径下加跑 `modern_transformer` 对照(同 tokenizer/seq_len/batch/token budget) | 三种子 JSON;若反转,通知全员改摘要 |
| T3 | Mamba-2 / GLA / DeltaNet | 同 T1 模式逐个补 | 同 T1 |
| T4 | Scaling law 三规模 | 等 T1-T3 完成后统一口径跑 | 均值±标准差 + 效率曲线数据 |

### 👤 老板(Everest)— 资源与钥匙,都是只有你能做的

| 序 | 任务 | 说明 |
|---|---|---|
| B1 | **算力预算拍板** | AutoDL 充值(P0-3 收尾约 ¥300-500)+ M2-P1 蒸馏预算(首轮约 $300-500) |
| B2 | **找回 O1 48M 权重** | 最可能在被禁用的 Modal workspace(`ac-ESq0Y6MGgrCtt67tOwrcDS`)或服务器;找到后一条命令导出浏览器 demo(§3 ⛔ 条目) |
| B3 | 教师 API 选型 | P1 蒸馏的推理轨迹来源(DeepSeek/Qwen API 性价比高);拿到 key 交给 CC 接管线 |
| B4 | (可选)投稿目标确认 | ICLR/NeurIPS/ICML 哪个 deadline,影响 T2-T4 排期 |

### 🤖 CC(Claude Code)— 本地 GPU + 全部代码,自主排队

| 序 | 任务 | 状态 |
|---|---|---|
| C1 | M2-P0 收尾:过夜单环实验 → `n_global_heads` sweep → 结论+曲线图固化 | 进行中 |
| C2 | 蒸馏数据管线(teacher-trace 采集/清洗/SFT 格式,CPU 先行,等 B3 的 key 接通) | 排队 |
| C3 | 生物模块 5-seed ablation 补课(GWT/PC/睡眠;Hebbian 改 fast-weights 或删) | 排队 |
| C4 | 每轮结果同步 README/BENCHMARKS/论文材料 + 本 HANDOFF | 持续 |

**汇合点**:T2 结果决定论文摘要改不改;C1 sweep 结果决定 `n_global_heads` 默认值;
B1 到位后 T1 立即可动。三条线没有互相等待的死锁。

## 4. 下一步（按优先级）

0. **M2-P0 收尾(本地,进行中)**:① 等单环过夜实验出"stack 深度 × 跳数"矩阵;
   ② 跑 `--n_global_heads` sweep(0/1/2/4,探针任务)定配额默认值;③ 把 P0 完整
   结论 + 曲线图固化进 `docs/ROADMAP_M2.md` 与 BENCHMARKS;④ 视结果决定
   深监督(每迭代加 loss)是否立项。命令模板:
   `py -3.11 benchmarks/reasoning_depth.py --task pointer_chase --difficulty 4 --n_values 8 --steps 30000 --seeds 0 --mode fixed --eval_depths 1 2 4 --stack --mix --n_global_heads 2 --tag quota-sweep`
1. **当前主线任务：继续跑强 baseline**：P0-3 modern_transformer 阶段成果已推送；下一步优先补 `mamba`，随后补 Mamba-2/GLA/DeltaNet 或同类高效架构；注意 Windows Mamba 无 CUDA kernel，强 baseline 和效率曲线建议迁到 Linux/A100。
2. **继续归档新 baseline 结果**：Mamba/Mamba-2/GLA/DeltaNet 每跑完一个模型，都同步三 seed JSON、run.log/标准化日志和更新后的 `scaling_train_20000_summary.txt`；checkpoint `.pt` 仍不提交。
3. **更新结果文档和论文材料**（2026-07-19 已完成第一轮）：P0-2 三种子 + P0-3 modern_transformer 结果已写入 README/BENCHMARKS/RESULTS/中英文论文/中英文 deck，并已明确标注 modern_transformer 领先 MT-LNN 11.3%、2K 旧结论已撤回、O1 参考锚限制。
4. **⚠️ 验证论文摘要的 14.7% 主张**（新增，重要）：论文摘要/结论的「比同参数 Transformer 低 14.7% PPL」来自大预算（100K 步 A100）实验，但**几乎肯定也是对着同一个 simple-reference 弱基线测的**。已先加限定语（"vs simple-reference，非现代基线"）作为止血，但**需要在大预算下补一轮 `modern_transformer` 对照**才能确认这个头号主张是否成立。若同样反转，摘要必须重写。建议在 AutoDL 上与其他强 baseline 一起排队。
5. ~~**并行待办：做 fp16 诊断**~~ → **已完成（2026-07-19）**：根因定位 + 修复 + 2000 步验证 + 全仓库审计，见上表。剩余可选：在 AutoDL 上跑 fp16 20K 确认长程（本地已验证 2000 步且与 fp32 质量持平）。
6. **扩 scaling law**：至少 3 个参数规模，固定 tokenizer/data/seq_len/batch/token budget，输出均值±标准差和效率曲线。
7. **补真实长上下文实验**：用 decode state / memory profile / 长上下文任务证明 O(1) working memory 的实际价值。

## 5. ⚠️ 要避免的坑（血泪教训）

1. **绝不信二手结论**：引用任何数据/文件/API 前必须亲自 Read/Grep/检查 JSON；尤其是跨会话、跨项目的结果。
2. **绝不把 O1 的数字搬进 M1 论文主表**：O1（84M/3000步/AMP）和 M1（126M/fp32/20K）不同口径，混用会造成不严谨甚至学术风险。
3. **长跑必须 checkpoint/resume**：无 checkpoint 时机器睡眠/SSH 断开/进程重启都会导致当前 arch 从 0 重跑。现在 `scaling_comparison.py` 已支持每 N 步保存。
4. **不要提交 checkpoint `.pt`**：`scaling_fp32/converge_probe/checkpoints/*.pt` 单个文件可达 1GB+，只用于本地/服务器恢复，不进 GitHub。
5. **8GB GPU 不适合并行训练**：M1 P0-2 约 7GB，占用时不要并行 O1 或其他 GPU 训练。
6. **transformer baseline 硬编码 `n_heads=13`**：`--d_model` 必须能被 13 整除（用 832 或 104，别用 128）。
7. **公开仓库自曝短板**：`PUBLICATION_READINESS.md`、本 HANDOFF 都包含未完成项和风险，提交前确认可以公开。

## 6. 关键命令速查

```powershell
# 本地 P0-1 2K 多种子（已完成）
py -3.11 benchmarks/scaling_comparison.py --mode train --steps 2000 `
  --seeds 0,1,2 --archs transformer,mt_lnn --dtype fp32 `
  --train_token_cap 50000000 --out_dir E:/M1/scaling_fp32

# 本地 P0-2 20K 收敛（已完成；支持 checkpoint/resume）
py -3.11 benchmarks/scaling_comparison.py --mode train --steps 20000 `
  --seeds 0,1,2 --archs mt_lnn,transformer --dtype fp32 `
  --ckpt_every 500 --resume `
  --out_dir E:/M1/scaling_fp32/converge_probe

# 远程 P0-3 modern_transformer（已完成）
ssh root@tulong91.imwork.net -p 54511
cd /root/autodl-tmp/M1
python benchmarks/scaling_comparison.py --mode train --steps 20000 \
  --seeds 0,1,2 \
  --archs modern_transformer \
  --dtype fp32 \
  --ckpt_every 500 --resume \
  --train_token_cap 50000000 \
  --out_dir /root/autodl-tmp/M1/scaling_fp32/p0_3_modern_transformer \
  2>&1 | tee /root/autodl-tmp/M1/scaling_fp32/p0_3_modern_transformer/run.log

# 远程下一步：跑 mamba（进行中/待完成）
cd /root/autodl-tmp/M1
python benchmarks/scaling_comparison.py --mode train --steps 20000 \
  --seeds 0,1,2 \
  --archs mamba \
  --dtype fp32 \
  --ckpt_every 500 --resume \
  --train_token_cap 50000000 \
  --out_dir /root/autodl-tmp/M1/scaling_fp32/p0_3_mamba \
  2>&1 | tee /root/autodl-tmp/M1/scaling_fp32/p0_3_mamba/run.log

# 查看 P0-2/P0-3 汇总
Get-Content E:\M1\scaling_fp32\converge_probe\scaling_train_20000_summary.txt

# 查 GPU 争抢
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## 7. 关键文件

- 训练脚本：`benchmarks/scaling_comparison.py`（`--mode train`，含 `--ckpt_every` / `--resume`）
- 强 baseline 代码：`benchmarks/baselines.py`（新增 `ModernCausalTransformer`：RoPE + RMSNorm + SwiGLU）
- P0-1 结果：`scaling_fp32/train_*_s*.json`
- P0-2/P0-3 结果：`scaling_fp32/converge_probe/train_mt_lnn_s0.json`、`train_mt_lnn_s1.json`、`train_mt_lnn_s2.json`、`train_transformer_s0.json`、`train_transformer_s1.json`、`train_transformer_s2.json`、`train_modern_transformer_s0.json`、`train_modern_transformer_s1.json`、`train_modern_transformer_s2.json`
- P0-2/P0-3 汇总：`scaling_fp32/converge_probe/scaling_train_20000_summary.txt`
- 标准化日志：`scaling_fp32/converge_probe/scaling_train_20000_mt_lnn.log`、`scaling_train_20000_transformer.log`、`scaling_train_20000_modern_transformer.log`
- checkpoint（不提交）：`scaling_fp32/converge_probe/checkpoints/*.pt`
- 施工图：`PUBLICATION_READINESS.md`
- 模型：`mt_lnn/model.py`、`mt_lnn/mt_lnn_layer.py`、`mt_lnn/mt_lnn_v2.py`
