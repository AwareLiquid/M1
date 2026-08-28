"""pscan_probe2.py — 第二轮探针驱动器（入仓存档版）。

矩阵：constant-A 生产默认路径 pscan_constant_A vs 特化版
pscan_chunkwise_constant_A；T ∈ {1K, 4K, 16K} × {fwd_bwd, fwd}，
C 敏感性 {32, 128} @16K，CPU/MPS 双档，每格子进程隔离。

判定规则（2026-08-29 实验前写死，与 docs/CHECKWISE_EXPERIMENT_LOG.md §D2 一致）：
  等价性  每个 chunk_const 格 max_rel_diff < 1e-3（vs pscan_constant_A）
  翻正    speedup = pscan_const/chunk_const，要求 16K fwd_bwd > 1 且随 T 递增
  观察    C ∈ {32, 64, 128}（调度开销消除后最优 C 可能回落）

复现第一轮（通用路径）矩阵：直接循环调用 pscan_probe_cell.py
  --impl pscan / chunk（见 CHECKWISE_EXPERIMENT_LOG.md §5）。
注意：T=512 生产区间补测格（log §5 表 3）为临时执行，未入本驱动矩阵。
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
SCAN = os.path.join(_REPO, "mt_lnn", "parallel_scan.py")
CELL = os.path.join(_HERE, "pscan_probe_cell.py")
OUT = os.path.join(_REPO, "benchmarks", "results", "pscan_probe_round2.json")


def run_cell(device, impl, T, C=64, mode="fwd_bwd"):
    cmd = [sys.executable, CELL, "--scan", SCAN, "--impl", impl, "--T", str(T),
           "--C", str(C), "--mode", mode, "--device", device]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  [FAIL] {device}/{impl}/T={T}/C={C}/{mode}\n{r.stderr[-600:]}")
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    results = []
    for dev in ("cpu", "mps"):
        print(f"\n=== device: {dev} (constant-A: production default path) ===")
        for T in (1024, 4096, 16384):
            for impl in ("pscan_const", "chunk_const"):
                row = run_cell(dev, impl, T)
                if row:
                    results.append(row)
                    print(f"  T={T:>5} {impl:>11} fwd_bwd  {row['wall_ms']:>8.1f} ms  "
                          f"rss={row['rss_mb']:>7.1f} MB  diff={row['max_rel_diff']}")
        for impl in ("pscan_const", "chunk_const"):
            row = run_cell(dev, impl, 16384, mode="fwd")
            if row:
                results.append(row)
                print(f"  T=16384 {impl:>11} fwd      {row['wall_ms']:>8.1f} ms  "
                      f"rss={row['rss_mb']:>7.1f} MB  diff={row['max_rel_diff']}")
        for C in (32, 128):
            row = run_cell(dev, "chunk_const", 16384, C=C)
            if row:
                results.append(row)
                print(f"  T=16384 chunk_const C={C:>3} fwd_bwd  {row['wall_ms']:>6.1f} ms  "
                      f"rss={row['rss_mb']:>7.1f} MB  diff={row['max_rel_diff']}")

    with open(OUT, "w") as f:
        json.dump(dict(results=results), f, indent=2)

    print("\n" + "=" * 70)
    print("对照判定 (speedup = pscan_const/chunk_const, >1 即特化版更快)")
    print("=" * 70)
    for dev in ("cpu", "mps"):
        for mode in ("fwd_bwd", "fwd"):
            for T in (1024, 4096, 16384):
                p = next((r for r in results if r["device"] == dev
                          and r["impl"] == "pscan_const"
                          and r["T"] == T and r["mode"] == mode), None)
                c = next((r for r in results if r["device"] == dev
                          and r["impl"] == "chunk_const"
                          and r["T"] == T and r["mode"] == mode
                          and r["C"] == 64), None)
                if p and c:
                    sp = p["wall_ms"] / max(c["wall_ms"], 1e-9)
                    mem = c["rss_mb"] / max(p["rss_mb"], 1e-9)
                    print(f"  [{dev}] T={T:>5} {mode:>7}: speedup={sp:5.2f}x  "
                          f"rss_ratio={mem:5.2f}  diff={c['max_rel_diff']}")
    eq_ok = all(r["max_rel_diff"] < 1e-3
                for r in results if r["impl"] == "chunk_const")
    print(f"\n等价性门槛 (max_rel_diff < 1e-3): {'PASS' if eq_ok else 'FAIL'}")
    print(f"原始数据: {OUT}")


if __name__ == "__main__":
    main()
