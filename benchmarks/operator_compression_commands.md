# Operator Compression Commands

## State-Only Streaming Demo

```powershell
python demo.py --ckpt checkpoints/final.pt --prompt "Hello" --state_only --session_id demo_session
python demo.py --ckpt checkpoints/final.pt --prompt "Continue from memory" --state_only --session_id demo_session
```

## Direct Target Demo

```powershell
python demo.py --ckpt checkpoints/final.pt --prompt "copy targets:" --direct_target_len 4
```

## Benchmark

```powershell
python benchmarks/state_only_streaming.py --steps 100 1000
python benchmarks/state_only_streaming.py --steps 100 --metrics_jsonl benchmarks/state_only_metrics.jsonl
```

## Sparse Resonance Prototype

```powershell
python benchmarks/state_only_streaming.py --steps 100 --sparse_resonance_kernel --sparse_resonance_top_k 1
python benchmarks/sparse_resonance_ablation.py --steps 100 --top_k 1 2 3
```

## Report

```powershell
python benchmarks/operator_compression_report.py --steps 100 1000
```

Notes:

- Use `requirements.lock` when rerunning published smoke benchmarks.
- State-only mode keeps recurrent `h_prev` and drops historical KV tensors.
- Direct target mode is useful only after the direct head has task-specific supervision.
- Scale-gate masking is blend-level masking unless `--sparse_resonance_kernel` is enabled.
- Sparse resonance skips inactive tau-scale projection/scan work, but top-k selection is chunk-dependent and can change exact KV parity.
- SQLite session persistence is local-first; production API serving needs a dedicated state-store adapter.
