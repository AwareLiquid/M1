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

| 配置 | 参数增量（832/12L 默认形状） | 总量（估） | vs 144.1M 锚 |
|---|---|---|---|
| base（lean core） | 0 | ≈125M（实测 TBD(Phase B)） | 欠配 |
| +qk_norm | 12×(64+64) = 1,536 | ≈125.0M | 欠配（可忽略） |
| +ffn | 12×3×832×2304 = **69.0M** | ≈194M | **超配 ~35%** |
| all_on | +ffn 同上（A2/A3 近似零参） | ≈194M | 超配 ~35% |

诚实口径：+ffn/all_on 在默认形状下**超过** 144.1M 锚。筛选衡量的是
"方向"，不主张 matched-param parity。若 +ffn 方向为正，两条 matched-param
路线（确认跑前二选一，`ffn_expansion` 旋钮已支持）：

1. **缩 FFN**：`ffn_expansion≈0.92` → d_ff=768 → +23.0M → ≈148M ≈ 锚；
2. **如实报告超配**：确认跑同时报 param 数，结论措辞注明参数不对齐。

参数实测以筛选脚本 row JSON 的 `params` 字段为准：TBD(Phase B)。

---

## 3. 筛选实验（Phase B 执行，数字占位）

四配置 × ≥3 seeds 配对筛选（`benchmarks/modern_trunk_screen.py`，P0 固定
口径：WT-103 / gpt2 / seq512 / batch4 / lr3e-4 / beta2 0.95 / clip 1.0 /
fp32）。**预注册判优**：all_on 相对 base 的配对 ΔPPL 在 ≥2/3 seeds 为负
才 CONFIRM，否则 NULL。

| 配置 | params | val_ppl per seed (2K 步) | ΔPPL vs base per seed |
|---|---|---|---|
| base | TBD(Phase B) | TBD(Phase B) | — |
| +ffn | TBD(Phase B) | TBD(Phase B) | TBD(Phase B) |
| +qk_norm | TBD(Phase B) | TBD(Phase B) | TBD(Phase B) |
| all_on | TBD(Phase B) | TBD(Phase B) | TBD(Phase B) |

判优结果：**TBD(Phase B)**（CONFIRM / NULL）。筛选 JSON 落
`benchmarks/results/modern_trunk_screen_*.json`（内嵌 screening_only 标记）。

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

- [ ] §2 参数表实测值：base / +qk_norm / +ffn / all_on 的 params（各 row JSON
      `params` 字段），替换 ≈125M/≈194M 估算
- [ ] §3 筛选表：每配置 val_ppl per seed、配对 ΔPPL per seed、verdict
      （CONFIRM / NULL，预注册规则：all_on ≥2/3 seeds 优于 base）
- [ ] matched-param 决策：若 CONFIRM，20K 确认跑前二选一——缩
      `ffn_expansion≈0.92`（d_ff=768，≈148M 贴锚）或如实报告超配 35%
- [ ] BENCHMARKS.md 新节：筛选数字一律标注 **screening-only**
- [ ] ≤200 字诚实结论：哪个缺失件筛选中最有效、是否值得排 20K 确认跑
- [ ] 红线自查：2K 数字未进 RESULTS.md/README；20K 未自动触发（runbook §4
      就绪，仅在判优 CONFIRM 后人工排期）

**worktree 保留**：`../M1-modern-trunk` 暂不移除——远程执行若再暴露 bug，修复
仍须走 worktree→commit→合回 的协议循环；全部通过后才 `git worktree remove`。
