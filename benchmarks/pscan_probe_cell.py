"""pscan_probe_cell.py — 探针单格 runner（入仓存档版，两轮实验的原始仪器）。

每格一个独立子进程运行，使峰值 RSS 可归因（进程内混跑会互相污染）。
默认加载本仓库 mt_lnn/parallel_scan.py（新旧实现同文件同源对比）；
--impl 选组：
  baseline     只加载不计算，测进程基线 RSS
  pscan / chunk                  通用路径（第一轮矩阵：逐 token 乘子）
  pscan_const / chunk_const      constant-A 路径（第二轮矩阵：生产默认路径）

数值口径：fp32，A∈(0.90,0.99)（训练后 decay 真实工作区），形状 (2,4,2,T,64)
为生产调用 (B,P,S,T,D) 的缩小版；wall_ms 取 median of iters（warmup 1）；
等价性 diff 为 chunk 类实现 vs 对应 pscan 参照的 max 相对误差（免费副产品）。

原始数据：benchmarks/results/pscan_probe_round{1,2}.json
记录文档：docs/CHECKWISE_EXPERIMENT_LOG.md
官方基准（Phase B）：benchmarks/pscan_bench.py —— 本探针是它的低成本前哨。
"""
import argparse
import importlib.util
import json
import os
import resource
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SCAN = os.path.join(_REPO, "mt_lnn", "parallel_scan.py")


def load_scan_module(path):
    spec = importlib.util.spec_from_file_location("pscan_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rss_mb():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1048576.0 if sys.platform == "darwin" else v / 1024.0


def sync(device, torch):
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser(description="单格探针（详见模块 docstring）")
    ap.add_argument("--scan", default=_DEFAULT_SCAN,
                    help="parallel_scan.py 路径（默认仓库内）")
    ap.add_argument("--impl", required=True,
                    choices=["baseline", "pscan", "chunk",
                             "pscan_const", "chunk_const"])
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd_bwd")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()

    import torch
    mod = load_scan_module(a.scan)

    if a.impl == "baseline":
        print(json.dumps(dict(impl="baseline", device=a.device,
                              wall_ms=0.0, rss_mb=rss_mb(), max_rel_diff=0.0)))
        return

    torch.manual_seed(0)
    B, P, S, D = 2, 4, 2, 64
    is_const = a.impl.endswith("_const")
    A = (torch.rand(B, P, S, device=a.device) * 0.09 + 0.90).requires_grad_() \
        if is_const else \
        (torch.rand(B, P, S, a.T, device=a.device) * 0.09 + 0.90).requires_grad_()
    X = torch.randn(B, P, S, a.T, D, device=a.device).requires_grad_()

    def ref_fn():
        return mod.pscan_constant_A(A, X) if is_const else mod.pscan(A, X)

    def fn():
        if a.impl in ("pscan", "pscan_const"):
            return ref_fn()
        if a.impl == "chunk":
            return mod.pscan_chunkwise(A, X, chunk_size=a.C)
        return mod.pscan_chunkwise_constant_A(A, X, chunk_size=a.C)

    mrd = 0.0
    if a.impl.startswith("chunk"):
        with torch.no_grad():
            ref = ref_fn()
            out = fn()
            mrd = ((out - ref).abs().max()
                   / ref.abs().max().clamp_min(1e-12)).item()
            del ref, out  # 计时前释放，否则虚增 chunk 格的峰值 RSS

    need_bwd = a.mode == "fwd_bwd"
    times = []
    for i in range(a.warmup + a.iters):
        A.grad = None
        X.grad = None
        sync(a.device, torch)
        t0 = time.perf_counter()
        H = fn()
        if need_bwd:
            H.square().mean().backward()
        sync(a.device, torch)
        if i >= a.warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()

    print(json.dumps(dict(
        impl=a.impl, T=a.T, C=(a.C if a.impl.startswith("chunk") else None),
        mode=a.mode, device=a.device,
        wall_ms=round(times[len(times) // 2], 2),
        rss_mb=round(rss_mb(), 1),
        max_rel_diff=float(f"{mrd:.2e}"),
    )))


if __name__ == "__main__":
    main()
