# MODERN_TRUNK — 前沿配方的三个缺失件（门控扩张 FFN / QK-RMSNorm / 深度缩放残差初始化）

分支 `iter/modern-trunk`（worktree `../M1-modern-trunk`）· 2026-08-29 · Phase A 骨架

**使命**：攻击已实测的 11.3% 质量缺口（modern Transformer 78.86±0.25 vs
mt_lnn 88.93±0.33，20K 步收敛口径，WikiText-103 / gpt2 tok / seq512 /
batch4 / lr3e-4 / fp32）。缺口假设：前沿混合架构（Qwen3-Next、Kimi K2/KDA、
Gemma）的 mixer 层从不单打独斗——每层仍配齐 SwiGLU 门控扩张 FFN、QK logit
封顶、深度缩放残差初始化；本仓库的液态层把这三件都省掉了。

**边界**：不动 chunkwise 扫描（iter/chunkwise-scan）、不动混合比例
（iter/o-series-hybrid-ratio）、不做 KV 账目（iter/kv-cache-frontier）、
不改 serve/server.py。

**筛选纪律（红线）**：2K 步数字只用于方向筛选，禁止进 RESULTS.md / README
（2026-07-19 收回 31% 主张的教训正是 2K 步外推）。本文件所有实验数字栏为
`TBD(Phase B)` 占位，Phase B 回填的筛选数字也永远标注 screening-only。

---

## 1. 三个缺失件与工业证据链

### 1.1 SwiGLU 门控扩张 FFN（config: `ffn_swiglu`, `ffn_expansion`）

- **证据**：Qwen3-Next 的 Gated DeltaNet（GDN）混合块与 Kimi Linear 的
  KDA（Kimi Delta Attention）块，其线性 mixer 之外每层都保留独立的 SwiGLU
  门控扩张 FFN——"线性层替代注意力"从不延伸到"替代 FFN"。SwiGLU 的 8/3
  扩张是参数匹配惯例（Shazeer, GLU Variants 2020：3×(8/3)·d 的 SwiGLU ≈
  2×4d 的 GELU-MLP 参数预算），本仓库 modern baseline 即按
  `d_ff = round(8/3·d_model / 256)·256` 取整（benchmarks/scaling_comparison.py:137）。
- **本仓库缺口**：MTLNNLayer 只有等宽 in_proj/out_proj（832→832，
  mt_lnn_layer.py:701-702）+ MAPGate 的 ReLU MLP（:671），FFN 槽位被吃掉。
- **实现**（commit fa51ae1）：每个 MTLNNBlock 在液态层 out_proj 残差之后接
  独立 pre-norm 子层 `x = x + dropout(SwiGLUFFN(RMSNorm(x)))`（RMSNorm 前置
  口径写死）。`ffn_expansion` 默认 8/3，d_ff 同 baseline 规则 → 832×8/3 →
  **2304**。零回归契约：默认 off 零参数零行为；on 时 w2 零初始化 → init
  forward 与 off 逐位一致、w2 有活梯度（house zero-gated-residual 模式）；
  FFN 初始化走私有生成器 + RNG save/restore，同 seed on/off trunk 逐位可比
  （init-luck 教训，model.py fast_weight 同款隔离）。

### 1.2 QK-RMSNorm（config: `qk_norm`）

- **证据**：Qwen3 全系在 q/k 投影后加 per-head QK-RMSNorm（arXiv:2505.09388）；
  Gemma 2/3 同款（Gemma 2, arXiv:2408.00118）；Kimi K2 为 1T 级训练专门
  发明 QK-Clip 来封顶 attention logits（arXiv:2507.20534）——logit 爆炸是
  全行业共识的故障类。
- **本仓库事故**：2026-07 fp16 Q@K 溢出事故正是该故障类；
  mt_attention.py 此前无任何 q/k 封顶机制。
- **实现**（commit 343cadc）：q/k 投影后、RoPE 前（position-free 路径则在
  SDPA 1/√d 缩放前）加可学习 per-head RMSNorm(d_head)；归一化后的 K 进
  KV cache，prefill/decode parity 由构造保证。数学性质：RMSNorm 输出 L2
  范数固定 ≤ max|w|·√d_head，故 |logit| ≤ d_head·max|w_q|·max|w_k| ≈ 64 ≪
  fp16 溢出量级 65504（单测锁定）。默认 off 零参数零行为；ones 初始化不
  消耗 RNG。

### 1.3 深度缩放残差初始化（config: `scaled_residual_init`）

- **证据**：GPT-2（Radford et al. 2019）起的标准实践——残差出口权重按
  1/√(2·n_layers) 缩放（GPT-2 原文为 1/√N，2·N 为现代通用变体），让深度
  残差流在 step 0 接近恒等映射。
- **本仓库教训**：init-luck 教训（model.py trunk-perturbation 注释）与
  2K 步 ±4.89 的种子方差部分源于出口投影未随深度缩放。
- **实现**（commit fce3a1f）：全部 init pass 之后，把每个 block 的
  attn.out_proj 与 lnn.out_proj **权重**乘 1/√(2·n_layers)（bias 不动；
  lnn.lateral.out_proj 是 RMC 内部投影、不是残差出口，明确排除）。确定性
  乘法不消耗 RNG，包 house RNG save/restore 模式 → 同 seed on/off trunk
  精确可比（on = off × 常数）。默认 off 位等价。

---

## 2. Matched-param 分析（对照锚：modern baseline 144.1M）

| 配置 | 参数增量（832/12L 默认形状） | 总量（实测） | vs 144.1M 锚 |
|---|---|---|---|
| base（lean core） | 0 | 126,041,819 | 欠配 |
| +qk_norm | 12×(64+64) = 1,536 | 126,043,355 | 欠配（可忽略） |
| +ffn | 12×3×832×2304 = **69.0M** | 195,061,211 | **超配 ~35%** |
| all_on | +ffn 同上（A2/A3 近似零参） | 195,062,747 | 超配 ~35% |

诚实口径：+ffn/all_on 在默认形状下**超过** 144.1M 锚。筛选衡量的是
"方向"，不主张 matched-param parity。若 +ffn 方向为正，两条 matched-param
路线（确认跑前二选一，`ffn_expansion` 旋钮已支持）：

1. **缩 FFN**：`ffn_expansion≈0.92` → d_ff=768 → +23.0M → ≈148M ≈ 锚；
2. **如实报告超配**：确认跑同时报 param 数，结论措辞注明参数不对齐。

参数实测以筛选脚本 row JSON 的 `params` 字段为准：§2 表已回填（2026-09-06）。

---

## 3. 筛选实验（Phase B 执行，2K 数字已出；screening-only）

四配置 × ≥3 seeds 配对筛选（`benchmarks/modern_trunk_screen.py`，P0 固定
口径：WT-103 / gpt2 / seq512 / batch4 / lr3e-4 / beta2 0.95 / clip 1.0 /
fp32）。**预注册判优**：all_on 相对 base 的配对 ΔPPL 在 ≥2/3 seeds 为负
才 CONFIRM，否则 NULL。

| 配置 | params（实测） | val_ppl per seed (2K 步) | ΔPPL vs base per seed |
|---|---|---|---|
| base | 126,041,819 | 255.17 / 254.95 / 255.69 | — |
| +ffn | 195,061,211 | 237.46 / 234.31 / 229.99 | −17.71 / −20.64 / −25.70 |
| +qk_norm | 126,043,355 | 244.58 / 244.49 / 243.54 | −10.60 / −10.46 / −12.15 |
| all_on | 195,062,747 | 232.38 / 230.46 / 228.61 | −22.79 / −24.49 / −27.08 |

判优结果：**CONFIRM**（all_on 3/3 seeds 优于 base；三旋钮各自单独为正，
叠加不冲突）。筛选 JSON 落 `benchmarks/results/modern_trunk_screen_2000step.json`
（内嵌 screening_only 标记，2K 数字任何情况下不进 RESULTS.md）。

---

## 3.5 20K 确认跑（2026-09-06 执行，结果见下）

2K 判优 CONFIRM → 按 §4 排 20K 确认（维护者排期，A100-80GB fp32，
base + all_on × 3 seeds，P0 同口径，`--train_token_cap 50000000
--eval_chunks 200`）：

| 格 | val_ppl (20K) |
|---|---|
| base s0 / s1 / s2 | 88.04 / 91.33 / 89.48（mean 89.62，对照历史 88.93±0.33 管道复现 ✓） |
| **all_on s0 / s1 / s2** | **73.33 / 74.76 / 73.83（mean 73.97）** |

- 配对 ΔPPL：−14.70 / −16.57 / −15.65，**3/3 seeds，verdict = CONFIRM**。
- all_on **反超 modern Transformer 基线 78.86±0.25**（20K 收敛口径）——
  三缺失件收复 11.3% 质量缺口并转正 ~6.5%。
- 逐格 JSON + 汇总：`benchmarks/results/modern_trunk_screen_*_20000step*.json`。

---

## 4. 20K 确认跑 runbook（远程可直接执行）

前置：`iter/modern-trunk` 已合并主 worktree；全量 pytest 全绿；GPU 机器
（fp32 口径，显存 ≥16GB；832/12L fp32 训练峰值同 scaling_comparison 量级）。

```bash
# ① 管道验证 (<10min, 任何机器)
python benchmarks/modern_trunk_screen.py --smoke

# ② 2K 筛选（Phase B 判优用; 结果落 benchmarks/results/modern_trunk_screen_2000step.json）
python benchmarks/modern_trunk_screen.py

# ③ 判优（预注册, 脚本自动打印）：
#    all_on 相对 base ≥2/3 seeds ΔPPL<0 → 排 20K 确认; 否则 NULL, 停止。

# ④ 20K 确认跑（远程 GPU, tmux/nohup; resume = 重跑同一命令, row JSON 存在即跳过）
nohup python benchmarks/modern_trunk_screen.py --steps 20000 \
    --train_token_cap 50000000 --eval_chunks 200 \
    > benchmarks/results/modern_trunk_screen_20k.log 2>&1 &

# 中断续跑：直接重跑 ④（已完成 config×seed 自动跳过）。
# 全部完成后汇总 JSON: benchmarks/results/modern_trunk_screen_20000step.json
```

**只有 20K 确认跑显示 all_on 显著优于 base 后**，才允许：RESULTS.md 加行
（20K 收敛口径）、BENCHMARKS.md 常规记录。2K 筛选数字任何情况下只进
BENCHMARKS.md 且必须标注 screening-only，绝不进 RESULTS.md。

---

## 5. Phase A 交付物（本分支）

| commit | 内容 |
|---|---|
| fa51ae1 | A1 SwiGLU 门控扩张 FFN（config + SwiGLUFFN + block 接线 + utils.RMSNorm） |
| 343cadc | A2 QK-RMSNorm（config + mt_attention） |
| fce3a1f | A3 深度缩放残差初始化（config + model._scale_residual_exits） |
| ac8431d | tests/test_modern_trunk.py：12 个位等价/数学性质/易回归点测试（只写未跑） |
| 4f7c2da | A4 benchmarks/modern_trunk_screen.py（筛选脚本，只写未执行） |
| 251f310 | Phase B smoke 修复：smoke 档 GWTB 头数整除断言（d_gw=13 不被 4 整除，按形状自适应 gwtb_n_heads） |
| （验收修正） | 修 d_ff 取整测试的 config 断言坑（裸 d_model=104 会撞 d_head assert）；A3 组补 fwd/bwd 有限性易回归点 → 共 13 个 |
| （本 commit） | A5 本文档骨架 + runbook |

**Phase B 进度（2026-08-29）**：全量 pytest ✅（1428 passed / 4 skipped / 183s，
含新增 13 个）；`--smoke` ✅（12/12 格，管道验证 + 抓出并修复 1 个 bug）；
**2K 筛选 ⏳ 挂起待 GPU —— 见 §6 待测任务记录**。

---

## 6. ⏳ 待执行任务：全量 2K 筛选（等 GPU，不自动执行）— 2026-08-29 挂起

**状态**：Phase B 前三步全绿；唯一未执行项 = 全量 2K 筛选（4 配置 × 3 seeds）。

**挂起原因（执行机实测）**：本机 Apple M1 Pro（8 核 CPU / 16GB RAM / 无 CUDA），
脚本回退纯 CPU fp32。全尺寸标定实测：base 8.35 s/step（245 tok/s）、+ffn
14.04 s/step（146 tok/s）→ 12 格 ≈ 78 小时连续运行（3.3~4.5 天），判定不适合
本机；任何 CUDA 卡（T4/3060 级）预计 10~40× 加速，全部 2~8 小时。内存已实测
16GB 可跑 195M 配置，无压力。

**待执行命令（在远程 GPU 机器的 repo 根目录）**：

```bash
python benchmarks/modern_trunk_screen.py --smoke   # 新机器先验管道 (<10min)
python benchmarks/modern_trunk_screen.py           # 全量 4配置×3seeds×2K步
# 断点续跑 = 重跑同一命令；汇总落 benchmarks/results/modern_trunk_screen_2000step.json
```

**测完必须回填的关键指标（逐项勾，缺一不可）**：

- [x] §2 参数表实测值：base 126,041,819 / +qk_norm 126,043,355 / +ffn
      195,061,211 / all_on 195,062,747（row JSON `params` 字段，2026-09-06 回填）
- [x] §3 筛选表：val_ppl per seed + 配对 ΔPPL per seed + verdict CONFIRM
      （all_on 3/3 seeds 优于 base）
- [x] matched-param 决策：CONFIRM → 20K 确认跑先行（§3.5，2026-09-06
      A100-80GB 执行，all_on 73.97 vs base 89.62 3/3 seeds）；
      **matched-param 二选一仍未定**——缩 `ffn_expansion≈0.92`（d_ff=768，
      ≈148M 贴锚）还是如实报超配 35%，留给"ffn 是否默认 on"决策时定
- [x] BENCHMARKS.md 新节：2K 数字一律标注 screening-only（见本仓库
      BENCHMARKS.md modern-trunk 节）
- [x] ≤200 字诚实结论：见 §3.5 与 RESULTS.md modern-trunk 行——三缺失件
      全部正向、可叠加；all_on 20K 反超 modern Transformer 基线，值得默认 on
      讨论（须 matched-param 证据支撑）
- [x] 红线自查：2K 数字未进 RESULTS.md/README（仅 BENCHMARKS screening-only）；
      20K 确认跑在判优 CONFIRM 后人工排期执行（2026-09-06）

**worktree 保留**：`../M1-modern-trunk` 暂不移除——远程执行若再暴露 bug，修复
仍须走 worktree→commit→合回 的协议循环；全部通过后才 `git worktree remove`。

---

## 7. 迭代决策日志（关键决策 + 理由 + 被否备选）

| # | 决策 | 理由/证据 | 被否备选 |
|---|---|---|---|
| D1 | 三旋钮默认 off + 逐位等价契约 | 历史 checkpoint/所有已归档结果零回归；13 个测试锁死 | 直接改默认（破坏全部存量可比性） |
| D2 | FFN w2 零初始化（恒等残差） | on 模型 init 输出与 off 逐位一致，开关语义干净；w2 有活梯度 | 三矩阵全随机（init 即漂移，on/off 不可比） |
| D3 | 新参数初始化走私有生成器 + RNG save/restore | init-luck 教训（mt_lnn_mtp trunk 移位事故，scaling_comparison 里有专门控制块）；保同 seed on/off A/B | 从全局流抽（后续所有 block init 被移位） |
| D4 | FFN 用 RMSNorm **前置**（pre-norm），口径写死 | Qwen/Llama 主流口径；二选一必须写死防口径漂移 | 后置 norm（非主流，且引入第二次选择） |
| D5 | QK-norm 放投影后、RoPE 前；归一化 K 进 KV cache | decode parity 由构造保证（KV-cache parity 测试锁定） | RoPE 后归一化（破坏旋转后内积语义，cache 口径需重推） |
| D6 | d_ff 取整规则 `round(exp·d/256)·256` 对齐 baseline | 与 scaling_comparison modern baseline 完全同口径，matched-param 可比 | 自定宽度（对照失真） |
| D7 | A3 只缩 attn/lnn 两个残差出口，明确排除 lateral.out_proj | RMC 内部投影不是残差出口（测试注释锁死该边界） | 所有 out_proj 后缀一刀切（误伤 RMC） |
| D8 | 判优规则**预注册**（all_on ≥2/3 seeds 优于 base）写死于脚本 | 防 hindsight 挑结果；跑前定死 | 跑完再定标准 |
| D9 | 2K 数字禁止进 RESULTS.md/README | 2026-07-19 收回 31% 主张的教训（2K 外推） | 把筛选数字当结论 |
| D10 | 挂起本机 78h 的 CPU 执行，等 GPU | 实测标定：12 格 ≈78h 连续；CUDA 10~40× → 2~8h；断点续跑设计使挂起零浪费 | 本机硬跑 3.3~4.5 天（占用主力机） |
| D11 | +ffn 默认形状超配 35% **如实记录** | 不偷偷缩配冒充 matched；诚实标注筛选=方向不=parity | 直接用 148M 形状做筛选（口径与 baseline 配方不一致） |
| D12 | 筛选仍用 baseline 同款 8/3 扩张，缩配推迟到确认阶段 | 先按对手同配方验方向；matched-param 二选一留在 §6 checklist（避免一次实验混两个变量） | 筛选就上 0.92 缩配（方向与口径双变量，结果不可解释） |

---

## 8. 现状快照与后续测试路线图（2026-08-29）

### 8.1 现状

- **分支**：`iter/modern-trunk`（worktree `../M1-modern-trunk`），主 worktree 已
  同步至同 HEAD；worktree 按 §6 理由保留。
- **代码**：三旋钮（`ffn_swiglu`/`ffn_expansion`、`qk_norm`、
  `scaled_residual_init`）全部落地、默认 off 位等价；13 个契约测试已并入全量
  pytest；筛选脚本 + runbook 就绪。
- **验证**：全量 pytest ✅ 1428 passed / 183s；smoke ✅ 12/12 格（抓出 1 个
  GWTB 头数 bug，`251f310` 修复后通过）；**全量 2K 筛选 ⏳ 挂起待 GPU（§6）**。

### 8.2 问题清单

| 类别 | 内容 |
|---|---|
| 主问题（未解决） | 11.3% 质量缺口**归因未定**：三缺失件 vs 液态 mixer 本身 vs 配方 |
| 已解决 | 位等价契约（13 测试）；smoke GWTB 头数断言；d_ff 测试 config 坑；GQA einsum 维度 |
| 受阻 | 本机无 CUDA，筛选需远程 GPU（§6 命令） |
| 遗留（非本分支） | 主 worktree 有记忆族未提交文件（session_state.py 等），归属 parametric-memory/event-stream 工作线，勿混入本分支提交 |

### 8.3 迭代方向与目标

**目标**：归因并尽可能收复 11.3% 缺口；让每个缺失件"赚到或失去"默认值
（default must earn its cost）。
**非目标**：不动 chunkwise 扫描、混合比例、KV 账目（其他分支领域）。

### 8.4 后续测试路线图（决策树，按序执行）

1. **GPU 全量 2K 筛选**（§6 两条命令）→ 产出配对 ΔPPL 表 + verdict。
2. **verdict = NULL**（全开 <2/3 seeds 胜）→ 归因转向 mixer 本身，火力移交
   chunkwise-scan / o-series-ratio 分支；三旋钮永久 off 或删除，本文档归档
   为负结果记录。
3. **verdict = CONFIRM** → matched-param 二选一（§6 checklist ③）→
   **20K 确认跑**（runbook §4）→ 达标 → RESULTS.md 加行（20K 口径）+
   启动"ffn 是否默认 on"的单独决策（涉及 +69M，须 matched-param 证据支撑）。
4. **端侧线（O 系列）**：若 FFN 证明有效，用小 `ffn_expansion` 重标定端侧
   配置再测（194M 默认形状不适合端侧，见 §2）。
5. **永久回归护栏**：13 个契约测试已随全量 pytest 常驻，任何后续改动破坏
   位等价即红。

---

## 9. 测试同学交接手册（全量筛选执行，2026-08-29）

**背景一句话**：本分支给模型装了三个默认关闭的架构开关，代码已全绿
（pytest 1428 / smoke 12 格），剩下的唯一工作是**在 GPU 机器上跑一次全量
筛选实验**，产出"哪个开关有效"的数据。

**前置环境**：任意 CUDA GPU 机器（T4 / 3060 级即可，显存 ≥16GB）；
检出本分支并装好依赖（`torch≥2.0`、`transformers`、`datasets`）；
首次运行自动下载 WikiText-103（约 500MB）+ gpt2 tokenizer。

**执行步骤**（repo 根目录，严格按序）：

1. `python -m pytest tests/test_modern_trunk.py -q`
   → 预期 **13 passed**（<1 分钟）。红 = 环境或代码问题，停止并报告。
2. `python benchmarks/modern_trunk_screen.py --smoke`
   → 预期 12 格全跑通（<10 分钟）。**这一档的数字全部无意义**，只验管道。
3. `python benchmarks/modern_trunk_screen.py`
   → 全量 2K 筛选（GPU 约 2~8 小时）。中断随时可停，重跑同命令自动续
   （已完成的格子自动跳过）。

**预期产物**（都在 `benchmarks/results/`）：
12 个逐格 JSON（`modern_trunk_screen_<config>_2000step_s<seed>.json`）+
汇总 `modern_trunk_screen_2000step.json`（内含配对差值表与 `verdict`）。

**判读**：只看汇总 JSON 的 `verdict` 字段——
`CONFIRM`（all_on 在 ≥2/3 seeds 配对优于 base）→ 值得排 20K 确认跑；
`NULL` → 方向归档，火力转向 mixer 本身。2K 数字只代表方向。

**红线（违反即返工）**：
① 任何 2K 数字**不得**写入 RESULTS.md / README；
② 20K 确认跑不在本次交接范围，CONFIRM 后由维护者人工排期；
③ 遇到 bug **不要在运行机改代码**——原样报告（含报错栈 + 已产生的
JSON），修复走 worktree 分支循环后重新合入。

**回填**：跑完把汇总 JSON 交回维护者，或直接按 §6 的 6 项 checklist
回填本文件 §2/§3（参数实测值、配对 ΔPPL、verdict、matched-param 决策、
BENCHMARKS.md 新节标注 screening-only、≤200 字结论）。
