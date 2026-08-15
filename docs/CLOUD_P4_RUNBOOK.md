# P4 云 GPU Runbook — 类脑机制注入 1.5B 基座的决策门实验

> 目的：一次回答"selective_decay（exp 参数化）注入真实基座后，是否有差异化价值"
> 决策门 G-A：长上下文外推改善且不伤质量 → 继续投入；否则专注 O 系列端侧路线。

## 0. 前置（GPU 机一到就执行）

```bash
git clone https://github.com/everest-an/M1.git && cd M1
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
