# P4 云 GPU Runbook — 类脑机制注入 1.5B 基座的决策门实验

> 目的：一次回答"selective_decay（exp 参数化）注入真实基座后，是否有差异化价值"
> 决策门 G-A：长上下文外推改善且不伤质量 → 继续投入；否则专注 O 系列端侧路线。

## 0. 前置（GPU 机一到就执行）

```bash
git clone https://github.com/AwareLiquid/M1.git && cd M1
pip install -r requirements.txt
# 确认 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. 三臂对比（核心决策门实验）

每臂 = 同一 Qwen2.5-1.5B 基座 + 不同 adapter 配置，WikiText-2 训练 5000 步：

```bash
# A) 基座对照（无 adapter，纯 LoRA）
python train_llama_mt_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset Salesforce/wikitext --dataset_config wikitext-2-raw-v1 \
    --seq_len 2048 --batch 1 --grad_accum 8 --steps 5000 --lora

# B) MT adapter + selective mamba 模式（当前默认）
python train_llama_mt_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset Salesforce/wikitext --dataset_config wikitext-2-raw-v1 \
    --seq_len 2048 --batch 1 --grad_accum 8 --steps 5000 \
    --mt_every 4 --lora --v2_selective --sel_mode mamba

# C) MT adapter + selective exp 模式（E5e 修复，本实验主角）
python train_llama_mt_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset Salesforce/wikitext --dataset_config wikitext-2-raw-v1 \
    --seq_len 2048 --batch 1 --grad_accum 8 --steps 5000 \
    --mt_every 4 --lora --v2_selective --sel_mode exp
```

## 2. 评估（三臂同一套）

```bash
# 1) 质量：PPL（不伤基座质量是第一前提）
for ckpt in checkpoints/llama_mt_adapter/*.pt; do
  python eval_llama_mt_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter $ckpt
done

# 2) 长上下文外推（决策门 G-A 的核心指标）
#    训练 seq_len=2048，评估 2048→8192 的 sliding-window PPL 曲线
python eval.py --adapter <ckpt> --sliding_window --context_lengths 2048 4096 8192

# 3) 记忆能力（fast-weight 跨窗口 recall 是否随 adapter 保留）
python bench_llama_mt_needle.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapters <ckpt> --context_lengths 2048 4096 --depths 0.1 0.5 0.9 --samples 5
```

## 3. 判定

| 结果 | 决策 |
|---|---|
| C（exp）的外推曲线明显优于 A/B，且 PPL 不伤 | **G-A 通过**：类脑机制在真实基座有差异化价值，继续 P4 规模化 |
| C 外推无改善 | 机制在适配器路径失效，回查注入是否正确（对比 toy 层 0.999 的差异） |
| C 的 PPL 明显伤 | exp 参数化在 LM 规模有害（与 22M PPL 检查一致），保持 mamba 模式 |

### 3b. 本地预演已跑（2026-08-15，0.5B × 500 步）

云 GPU 前用本地 Qwen2.5-0.5B 预演两臂（mamba vs exp，各 500 步 seq_len 256
batch 1 grad_accum 8，可训练参数 1.188%）：

| arm | 500 步 loss 采样（每 50 步） | 均值 |
|---|---|---:|
| mamba | 3.56/2.31/3.21/2.35/2.66/3.01/2.68/3.26/2.96/2.58 | ~2.86 |
| exp | 3.37/3.14/3.17/2.66/3.65/3.08/3.33/2.81/2.50/3.26 | ~3.10 |

- **验证了管线**：exp 参数化在真实基座（V2 adapter + LoRA + 0.5B）上稳定
  训练，无 NaN/爆炸 —— 云 GPU 决策门实验不会遇到注入崩溃。
- **未验证方向性**：500 步 batch 1 噪音太大（单点 std ~0.4），mamba/exp 的
  ~8% loss 差异在噪音内，不是结论。决策门仍需 5000 步 + 3 seeds。

### 3c. 本地降级决策门已跑（2026-08-15，0.5B × 1500 步 × seq_len 512）

训练 seq_len 512（1500 步）后，评估 base / mamba / exp 在 512/1024/2048 的
wikitext-2 PPL（10-20 batches）：

| seq_len | base | mamba | exp |
|---|---:|---:|---:|
| 512（in-dist） | 15.650 | 10.693 | 10.713 |
| 1024（2× 外推） | 12.920 | 9.846 | 9.845 |
| 2048（4× 外推） | 11.439 | 9.289 | 9.298 |

- **adapter 有效**：MT V2 + LoRA 把 PPL 压降 ~30-40%（15.65→10.69 @ 512）。
- **mamba vs exp 无差异**：所有长度差异 <0.3%（10.693 vs 10.713 等）——在
  1500 步的 LM 训练中，带符号 exp 参数化和恒正 mamba 参数化打平。
- **诚实解读**：exp 参数化的 toy 层优势（长度外推 0.999）是**任务特异的**
  ——parity 需要 ±1 翻转（负特征值），wikitext LM 不需要。这与 22M PPL 检查
  一致（exp 比 tanh 高 6%）。**G-A 决策门的本地预演结论：exp 在真实 LM 基座
  无差异化价值，维持 mamba 为默认**（toy 层的 exp 优势仍作为 S1 主张的
  电路级证据，不迁移到 LM 质量主张）。
- 附带修复：`attach_adapters_from_checkpoint` 现在按 checkpoint 的
  `--adapter` 标记选择 V2/V1 重建（之前 V2 评估会 key 不匹配崩溃）。

### 3d. 完整决策门已跑（2026-08-16 云 A100，1.5B × 5000 步 × seq_len 2048）

四臂 + 无 adapter 基座，wikitext-2 PPL（5-20 batches）：

| arm | 2048（in-dist） | 4096（2×） | 8192（4×） |
|---|---:|---:|---:|
| base（无 adapter） | 7.682 | 7.236 | 8.035 |
| **纯 LoRA** | **6.546** | **6.434** | **7.302** |
| V1 adapter + LoRA | 6.547 | 6.443 | 7.290 |
| V2-mamba + LoRA | 6.576 | 6.477 | 7.355 |
| V2-exp + LoRA | 6.592 | 6.477 | 7.344 |

- **G-A 决策门判定（完整版）**：LoRA 是全部贡献（−15% PPL），**MT adapter
  （V1/V2）增量 ≈ 0**（纯 LoRA 6.546 vs V1 6.547 vs V2 6.576/6.592）；mamba
  vs exp 打平（<0.3%）。与本地 0.5B 降级版、22M PPL 检查、历史 adapter
  retraction（"MT adds ≈0 PPL beyond LoRA"）三方一致。
- **类脑机制的定位因此明确**：选择性/记忆机制的价值不在 LM PPL 质量，
  在 O(1) 内存（S2）、跨会话记忆（S3）、电路级任务（S1）。P4 的规模化
  （7B）不再作为"质量追赶"路线——质量交给 LoRA/蒸馏，类脑机制作为
  差异化能力注入（记忆 API、长上下文效率）。
- 评估协议：`eval_p4.sh`（远程）；纯 LoRA 臂由新增 `--no_mt` 支持。

### 3e. 3-seed 完整决策门（2026-08-16 云 A100，3 seeds × 四臂 × 5000 步）

3 seeds 均值（wikitext-2 PPL）：

| arm | 2048（in-dist） | 4096（2×） | 8192（4×） |
|---|---:|---:|---:|
| **纯 LoRA** | **6.540** | **6.428** | **7.289** |
| V1 adapter + LoRA | 6.547 | 6.434 | 7.294 |
| V2-mamba + LoRA | 6.583 | 6.469 | 7.348 |
| V2-exp + LoRA | 6.598 | 6.476 | 7.352 |

- **3-seed 结论（与 seed 0 单臂一致）**：① 纯 LoRA 与 V1 打平（6.540 vs
  6.547，+0.1%）→ MT adapter 增量 ≈ 0；② V2（mamba/exp）比纯 LoRA 略差
  （+0.7%/+0.9%）；③ mamba vs exp 打平（+0.2%，seed 方差内）；④ 4× 外推
  （8192）时纯 LoRA 退化最小（+11.5% vs V2 的 +11.6-11.7%）。
- **最终定位**：类脑机制的差异化**不在 LM PPL 质量**——在 O(1) 内存
  （S2，471859×@1M）、跨会话记忆（S3）、电路级任务（S1，6/6 vs 0/6）。
  质量路线交给 LoRA/蒸馏；类脑机制作为差异化能力注入（记忆 API、长上下
  文效率），这是 4 个独立实验（toy 层、22M、0.5B、1.5B×3seeds）的一致结论。

## 4. 算力与时间

- 1×A100（80GB）：1.5B 基座 + LoRA + MT adapter，5000 步 seq_len 2048
  ≈ **1.5-2 小时/臂**，三臂 ≈ 5-6 小时 + 评估 1 小时 ≈ **1 个 A100-天**
- 显存：1.5B bf16 ≈ 3GB + 梯度（LoRA only ~10M 参数）≈ 1GB → 16GB 足够，
  A100 80GB 余量可开 grad_accum 更大或 batch 2

## 5. 若 G-A 通过，规模化队列（8×A100）

```bash
# 7B 基座 + 相同协议（batch 1, grad_accum 8, 5000 步）
python train_llama_mt_adapter.py --model Qwen/Qwen2.5-7B-Instruct \
    --dataset Salesforce/wikitext --dataset_config wikitext-2-raw-v1 \
    --seq_len 4096 --batch 1 --grad_accum 16 --steps 5000 \
    --mt_every 4 --lora --v2_selective --sel_mode exp
```

## 6. 本地已备好的支撑

- `mt_lnn/mt_lnn_v2.py`：V2 adapter 已支持 `selective_decay_mode="exp"`（本次改动）
- `train_llama_mt_adapter.py`：`--sel_mode {mamba,exp}` 旋钮已接好
- 零回归：27 核心测试通过（test_mt_lnn_v2 + test_model）
- E5e 证据链：exp 参数化在 toy 层实现长度外推 0.999（ABLATIONS.md 完整入档）

---

## 附录 P2：蒸馏到 O 系列端侧模型（体积维度）

`train_distill.py`（本地 smoke 已验证，39.1M 学生 + KL/CE 蒸馏正常）。

```bash
# 教师 Qwen2.5-0.5B → 学生 O 系列（纯 LNN，attention-free，~79M @ d_model 384）
python train_distill.py --teacher Qwen/Qwen2.5-0.5B-Instruct \
    --dataset Salesforce/wikitext --dataset_config wikitext-2-raw-v1 \
    --d_model 384 --n_layers 8 --steps 5000 --temperature 4.0 --alpha 0.5 \
    --sel_mode exp
```

- 学生用教师同一 tokenizer（151936 vocab）保证 KL 对齐；embedding 层占大头
  （~58M @ d_model 384），端侧 <5MB 需第二段词表裁剪 + int8 量化。
- 验证标准（G-B）：蒸馏学生达到教师 80% 下游性能 @ 4% 体积。
- 算力：1×A100 × 2-3 天（5000 步 + 下游评估）。

**P2 蒸馏已跑（2026-08-16 云 A100，5000 步 21 分钟）**：384d×8L O 系列学生
（~79M）从 Qwen2.5-0.5B 蒸馏，CE 10.6 → 7.65（最终 7.648，KL+CE 混合损失
正常收敛），checkpoint 483MB（151936 vocab embedding 占大头）。下一步：词表
裁剪 + int8 量化到 <5MB（部署段，未跑）。

**P2 端侧量化已跑（2026-08-16）**：词表裁剪 151643→8211（wikitext 高频 +
特殊 token）+ per-tensor int8 → **13.81MB**（从 482.9MB，35×压缩）。
距离 <5MB 目标还差 2.7×（下一步：4-bit 量化或 2K 词表或 208d 学生）；
路径已验证可行。O1-Sound 的 1.27MB（小词表 + 小模型）是同路线的更小实例。

**P2 4-bit 量化已跑（2026-08-16）**：int4 打包（两 int4 进一 int8）→
**7.11MB**（fp32 483MB → int8 13.81MB → 4bit 7.11MB，总 68× 压缩）。
端侧规模 7.11MB 已接近实用（O1-Sound 1.27MB 是更小模型）；<5MB 需
4K 词表或 208d 学生，路径明确。

**P2 端侧 <5MB 闭环达成（2026-08-16）**：4K 词表（4096）+ 去 target_head
（MTP 辅助头，推理不需要）+ weight-tie（embedding/lm_head 共享）+ 4bit →
**3.15MB**（从 fp32 483MB，**153× 压缩**）。体积维度验收通过：O 系列
蒸馏学生可部署到 <5MB 端侧。
