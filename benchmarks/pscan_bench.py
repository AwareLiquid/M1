"""pscan_bench.py — pscan (Blelloch log-depth) vs pscan_chunkwise (SSD 式) 微基准

背景 (iter/chunkwise-scan): 递归训练路径的 scan 是训练吞吐的关键项。
Mamba-2 SSD / GLA / DeltaNet / KDA 的训练内核全部是 chunkwise 分解
(块内并行 matmul + 块间 carry)。本脚本量化两条路径在同一 liquid 递归
    h_t = A_t * h_{t-1} + X_t
上的 墙钟 / 峰值显存，并给出 chunk_size ∈ {32, 64, 128} 敏感性。

用法:
  python benchmarks/pscan_bench.py --smoke                 # 快档 T ∈ {1K, 4K}
  python benchmarks/pscan_bench.py                         # auto (GPU 优先), 全档 1K→64K
  python benchmarks/pscan_bench.py --device cuda --iters 10
  python benchmarks/pscan_bench.py --device cpu            # 纯 CPU (64K 档 fwd+bwd 可能分钟级)

输出: benchmarks/results/pscan_bench.json
  每行: {impl, T, chunk_size, mode, wall_ms, peak_mem_bytes}
  mode ∈ {fwd, fwd_bwd} — 训练相关的是 fwd_bwd (chunkwise 的文献收益是训练提速)。

诚实边界:
  * 微基准，只测 scan 算子本身，不是端到端训练吞吐;
  * 形状取生产调用 (B,P,S,T,D) 的缩小版 (leading=2·4·2, D=64);
  * CPU 不测峰值显存 (torch.cuda.max_memory_allocated 无 CPU 对应物,
    ru_maxrss 是进程级单调量, 无法按 case 归零);
  * A 取 (0.90, 0.99) — 训练后 decay 的真实工作区, 而非测试用的宽区间。
"""

import argparse
import json
import os
import sys
import platform
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mt_lnn.parallel_scan import pscan, pscan_chunkwise

SEQ_LENS = [1024, 4096, 16384, 65536]
SMOKE_SEQ_LENS = [1024, 4096]
CHUNK_SIZES = [32, 64, 128]
LEADING_DIMS = (2, 4, 2)          # (B, P, S) of the production call, shrunk
D = 64                            # per-protofilament state width
DEFAULT_OUT = "benchmarks/results/pscan_bench.json"
MODES = ("fwd", "fwd_bwd")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    lens = SMOKE_SEQ_LENS if args.smoke else args.seq_lens
    print(f"[pscan_bench] device={device} smoke={args.smoke} "
          f"T={lens} chunks={args.chunk_sizes} iters={args.iters}")
    rows = []
    for T in lens:
        rows.extend(bench_length(T, device, args))
    write_results(rows, args, device, lens)
    print_summary(rows, args.out)


def bench_length(T, device, args):
    """All impl × chunk × mode cells for one sequence length; prints rows."""
    torch.manual_seed(0)
    A = (torch.rand(*LEADING_DIMS, T, device=device) * 0.09 + 0.90).requires_grad_()
    X = torch.randn(*LEADING_DIMS, T, D, device=device).requires_grad_()
    impls = [("pscan", None)] + [("chunkwise", c) for c in args.chunk_sizes]
    rows = []
    for name, chunk in impls:
        fn = (lambda a, x: pscan(a, x)) if chunk is None else \
             (lambda a, x, c=chunk: pscan_chunkwise(a, x, chunk_size=c))
        for mode in MODES:
            row = bench_cell(name, chunk, mode, fn, A, X, T, args)
            rows.append(row)
            print(f"  T={T:>6} {name:>9} chunk={str(chunk):>4} {mode:>7}: "
                  f"{row['wall_ms']:>10.2f} ms   peak={row['peak_mem_bytes']}")
    del A, X
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def bench_cell(name, chunk, mode, fn, A, X, T, args):
    """One (impl, chunk, mode) measurement; OOM is a result, not a crash."""
    need_bwd = mode == "fwd_bwd"
    try:
        wall_ms = timed_run(fn, A, X, args.iters, args.warmup, need_bwd)
        peak = peak_mem_bytes(fn, A, X, need_bwd)
        return dict(impl=name, T=T, chunk_size=chunk, mode=mode,
                    wall_ms=round(wall_ms, 3), peak_mem_bytes=peak)
    except RuntimeError as e:                     # torch.cuda.OutOfMemoryError
        if "out of memory" not in str(e).lower():  # and its pre-1.13 alias
            raise
        if A.device.type == "cuda":
            torch.cuda.empty_cache()
        return dict(impl=name, T=T, chunk_size=chunk, mode=mode,
                    wall_ms=None, peak_mem_bytes=None, error="cuda_oom")


def timed_run(fn, A, X, iters, warmup, need_bwd):
    """Median wall-clock ms over `iters` measured runs (after `warmup`)."""
    times = []
    for i in range(warmup + iters):
        if need_bwd:
            A.grad = None
            X.grad = None
        sync_like(A)
        t0 = time.perf_counter()
        H = fn(A, X)
        if need_bwd:
            H.sum().backward()
        sync_like(A)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)[len(times) // 2]


def peak_mem_bytes(fn, A, X, need_bwd):
    """torch.cuda.max_memory_allocated for one call; None on CPU."""
    if A.device.type != "cuda":
        return None
    A.grad = None
    X.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    H = fn(A, X)
    if need_bwd:
        H.sum().backward()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def sync_like(t):
    if t.device.type == "cuda":
        torch.cuda.synchronize()


def make_meta(args, device, lens):
    return dict(
        script="benchmarks/pscan_bench.py",
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        torch=torch.__version__,
        device=device,
        gpu_name=(torch.cuda.get_device_name(0) if device == "cuda" else None),
        dtype="float32",
        leading_dims=list(LEADING_DIMS),
        d_state=D,
        seq_lens=lens,
        chunk_sizes=args.chunk_sizes,
        iters=args.iters,
        warmup=args.warmup,
        smoke=args.smoke,
        note="wall_ms is median; peak_mem_bytes is torch.cuda.max_memory_allocated "
             "(None on CPU); wall_ms=None + error=cuda_oom marks an OOM cell.",
        hostname=platform.node(),
    )


def write_results(rows, args, device, lens):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(meta=make_meta(args, device, lens), results=rows), f, indent=2)
    print(f"[pscan_bench] wrote {len(rows)} rows -> {args.out}")


def print_summary(rows, out_path):
    """Headline: chunkwise/pscan wall-clock ratio per T (fwd_bwd, chunk=64)."""
    for T in sorted({r["T"] for r in rows}):
        base = next((r for r in rows if r["T"] == T and r["impl"] == "pscan"
                     and r["mode"] == "fwd_bwd" and r["wall_ms"]), None)
        best = next((r for r in rows if r["T"] == T and r["impl"] == "chunkwise"
                     and r["mode"] == "fwd_bwd" and r["chunk_size"] == 64
                     and r["wall_ms"]), None)
        if base and best:
            print(f"[summary] T={T}: chunkwise(C=64)/pscan fwd_bwd wall ratio = "
                  f"{best['wall_ms'] / base['wall_ms']:.3f}x")
    print(f"[summary] full JSON: {out_path}")


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help=f"fast档: T ∈ {SMOKE_SEQ_LENS} only")
    p.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=CHUNK_SIZES)
    p.add_argument("--iters", type=int, default=3, help="measured runs per cell")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


if __name__ == "__main__":
    main()
