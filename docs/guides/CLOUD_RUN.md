# Cloud Run: Llama/Qwen + MT Adapter

This is the first serious experiment: keep a real pretrained LM frozen, train
MT-LNN residual adapters plus optional LoRA, then compare base vs adapter on
perplexity and needle retrieval.

## Recommended Machine

Start small:

| GPU | Use |
|---|---|
| RTX 4090 24GB | First 1B-1.5B adapter run |
| RTX A6000 48GB | Longer context or safer batch sizes |
| A100 40/80GB | Parameter sweeps and 2048-4096 context |

Avoid H100 for the first pass. The question is whether the adapter has signal,
not whether we can spend money quickly.

## Fresh Machine Setup

```bash
git clone <your-repo-url> O1
cd O1

# Optional if the model is gated.
huggingface-cli login

bash scripts/cloud_llama_mt_experiment.sh
```

Default model:

```bash
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

Override settings:

```bash
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct \
SEQ_LEN=2048 \
STEPS=2000 \
MT_EVERY=4 \
NEEDLE_CONTEXTS="1024 2048 4096" \
bash scripts/cloud_llama_mt_experiment.sh
```

## What It Produces

The script writes a timestamped result directory:

```text
benchmarks/cloud_YYYYMMDD_HHMMSS/
  train.log
  adapter.txt
  ppl_ablation.log
  ppl_ablation.json
  needle.log
  needle.json
```

The adapter checkpoint is saved under:

```text
checkpoints/llama_mt_adapter/llama_mt_adapter_*.pt
```

## Pass/Fail Criteria

The adapter is worth pursuing only if:

| Metric | Pass |
|---|---|
| PPL | No meaningful regression vs base |
| Needle | Better exact/contains rate at long context or deep needle positions |
| Speed | Adapter overhead is acceptable for the target use |
| Stability | Training loss decreases without NaNs |

If PPL worsens and needle does not improve, the adapter is not helping yet.

## First Sweep

Run these three experiments before touching architecture:

```bash
# 1. Short, cheap sanity run
STEPS=300 SEQ_LEN=512 NEEDLE_CONTEXTS="512 1024" bash scripts/cloud_llama_mt_experiment.sh

# 2. Main 1.5B run
STEPS=1000 SEQ_LEN=1024 NEEDLE_CONTEXTS="1024 2048 4096" bash scripts/cloud_llama_mt_experiment.sh

# 3. Longer context
STEPS=2000 SEQ_LEN=2048 NEEDLE_CONTEXTS="2048 4096 8192" bash scripts/cloud_llama_mt_experiment.sh
```

## Notes

- `train_llama_mt_adapter.py` freezes the base LM.
- `--lora` trains LoRA plus MT adapter weights.
- Checkpoints save only adapter/LoRA weights, not the base model.
- The CPU-only local results are not a valid speed estimate for cloud GPU runs.
