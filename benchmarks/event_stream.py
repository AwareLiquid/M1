"""event_stream.py — DVS 物理原理的异步事件流生成器（纯 numpy）。

事件相机语义：连续潜信号（多通道，模拟多个像素/区域的辐照度）在积分
网格上演化；某通道相对**上次触发参考值**的变化量 |Δv| 超过阈值 θ 即
触发事件 (t, channel, polarity)，极性为变化方向。

Δt 跨数量级的机制（真实 DVS 的核心现象）：场景在"运动"与"静默"之间
交替——运动期（burst）事件密集，静默期一个事件都没有，静默时长
log-uniform 分布在 [FINE_DT·10, FINE_DT·10^(1+span)] 秒。span_decades
从 1 拉到 4 即把流内 Δt 分布的跨度从 ~1 个数量级拉到 ~4 个数量级，
而**事件总数大体不变**（跨度拉宽不稀释信息密度——公平前提）。

旋钮：
  * theta        — 触发阈值，控制 burst 内事件密度
  * bandwidth    — 潜信号谐波带宽（事件时间结构的复杂度）
  * n_channels   — 潜信号通道数
  * span_decades — 静默间隙的十进制数量级上限（Task 3 的 Δt 跨度旋钮）

被测试锁定的不变量（tests/test_event_stream.py）：
  1. 时间戳严格递增（同一步内的跨通道并列用通道序号 ε 偏移保证）
  2. |Δv| < θ 不触发（峰峰值小于 θ 的信号零事件）
  3. θ→0 退化为近似规则采样（每个网格步都触发，Δt 恒等于网格步长）
  4. realized Δt 跨度（p98/p5，排除同格同时事件）随 span_decades 单调扩大

输出张量 (T, C+2)：C 维通道 one-hot + 极性 ±1 + Δt 通道。Δt 为真时间
差（秒），经 log10(1 + dt/FINE_DT) 压缩（跨数量级可线性化），**喂给
所有架构同一变换**（沿用 battery_irregular_sampling 的公平护栏口径）。
"""

from __future__ import annotations

import argparse

import numpy as np

FINE_DT = 1e-3          # 积分网格步长（秒）
GAP_MIN = 10.0          # 最短静默间隙 = 10 个网格步；realized span ≈ 1 + s/2
BURST_LEN = 24.0        # 每个 burst 的目标事件数
_MAX_BURSTS = 200       # θ 过大时防死循环的上限（不足 min_events → 丢弃）


def main():
    args = _cli()
    ds = make_dataset(n_episodes=8, n_channels=args.channels, seq_len=64,
                      theta=args.theta, span_decades=args.span,
                      bandwidth=args.bandwidth, seed=0)
    X = ds["X"]
    print(f"episodes {len(X)}  tensor {X.shape}  "
          f"kept_fraction {ds['kept_fraction']:.2f}")
    print(f"dt (raw s): p5 {ds['dt_p5_s']:.2e}  median {ds['dt_med_s']:.2e}  "
          f"p98 {ds['dt_p98_s']:.2e}")
    print(f"realized dt span {ds['dt_span_decades']:.2f} decades "
          f"(requested gap span {args.span})")
    return 0


def make_dataset(n_episodes, n_channels, seq_len, theta, span_decades,
                 bandwidth, seed):
    """生成 n_episodes 条事件流并张量化；不足 seq_len 个事件的流丢弃。

    返回 dict：X (N, T, C+2) float32、y (N,) 末时刻潜信号通道均值（状态
    估计目标，回归，由 bench 侧用训练统计标准化）、以及 realized 统计。
    """
    rng = np.random.default_rng(seed)
    Xs, ys, dts, n_raw = [], [], [], 0
    for _ in range(n_episodes):
        n_raw += 1
        ep = generate_episode(rng, n_channels, theta, span_decades,
                               bandwidth, min_events=seq_len)
        if ep is None:
            continue
        t, c, p, target = ep
        dts.append(np.diff(t))
        Xs.append(events_to_tensor(t, c, p, n_channels, seq_len))
        ys.append(target)
    if not Xs:
        raise ValueError(f"theta={theta}: no episode reached {seq_len} events")
    raw_dt = np.concatenate(dts)
    # 同格"同时事件"（dt < 0.1 网格步）不参与跨度统计——它们的 Δt=0 是
    # 语义上的同时，不是物理间隙
    raw_dt = raw_dt[raw_dt > 0.1 * FINE_DT][:100_000]
    return {"X": np.stack(Xs), "y": np.array(ys, dtype=np.float32),
            "kept_fraction": len(Xs) / n_raw,
            "dt_p5_s": _pct(raw_dt, 5), "dt_med_s": _pct(raw_dt, 50),
            "dt_p98_s": _pct(raw_dt, 98),
            "dt_span_decades": float(np.log10(_pct(raw_dt, 98)
                                              / _pct(raw_dt, 5)))}


def generate_episode(rng, n_channels, theta, span_decades, bandwidth,
                     min_events):
    """一条事件流：运动 burst（阈值触发）与静默间隙交替推进绝对时钟。

    间隙 g = GAP_MIN·FINE_DT·10^U(0, span_decades)，U 均匀。返回
    (times, channels, polarity, target)；总事件数 < min_events 返回 None。
    """
    ts, cs, ps, clock = [], [], [], 0.0
    for _ in range(_MAX_BURSTS):
        if sum(len(a) for a in ts) >= min_events:
            break
        t_local = np.arange(0.0, _burst_seconds(n_channels, theta, bandwidth),
                            FINE_DT)
        v = _latent(rng, clock + t_local, n_channels, bandwidth)
        for ch in range(n_channels):
            idx, pol = _emit(v[ch], theta)
            ts.append(clock + t_local[idx])
            cs.append(np.full(len(idx), ch))
            ps.append(pol)
        clock += t_local[-1] + _gap(rng, span_decades)
    if sum(len(a) for a in ts) < min_events:
        return None
    t = np.concatenate(ts)
    c = np.concatenate(cs)
    p = np.concatenate(ps)
    order = np.lexsort((c, t))
    # 同一网格步的跨通道并列事件用通道序号 ε 偏移 → 时间戳严格递增
    eps = FINE_DT * 1e-3 / max(n_channels, 2)
    return (t[order] + c[order] * eps, c[order].astype(np.int64),
            p[order], float(v[:, -1].mean()))


def events_to_tensor(times, channels, polarity, n_channels, seq_len):
    """(T',) 事件 → 截断/补齐到 seq_len 的 (T, C+2) 张量。

    不足 T 的事件用零填充（one-hot 全零 + Δt=0 标记 padding 位）。
    Δt 通道：log10(1 + dt/FINE_DT)，首事件 Δt 取其自身时间戳。
    """
    T = min(seq_len, len(times))
    X = np.zeros((seq_len, n_channels + 2), dtype=np.float32)
    X[np.arange(T), channels[:T]] = 1.0
    X[:T, n_channels] = polarity[:T]
    dt = np.diff(times[:T], prepend=0.0)
    X[:T, n_channels + 1] = np.log10(1.0 + dt / FINE_DT)
    return X


def _latent(rng, t, n_channels, bandwidth):
    """多通道潜信号：每通道 3 个谐波、随机相位，各通道等幅。"""
    freqs = bandwidth * np.array([0.6, 1.0, 1.7])
    phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 3))
    w = 2 * np.pi * freqs
    return np.sin(w[None, None, :] * t[None, :, None]
                  + phase[:, None, :]).sum(axis=2)


def _emit(v, theta):
    """逐通道阈值触发：|v - ref| ≥ θ → 事件并重置 ref=v。

    θ=0 时每个网格步都触发（含零变化步）——θ→0 退化不变量由此保证。
    """
    theta = max(theta, 0.0)
    ref, i, idx, pol = v[0], 1, [], []
    while i < len(v):
        hit = np.nonzero(np.abs(v[i:] - ref) >= theta)[0]
        if hit.size == 0:
            break
        j = i + int(hit[0])
        idx.append(j)
        pol.append(1 if v[j] > ref else -1)
        ref, i = v[j], j + 1
    return np.asarray(idx, dtype=np.int64), np.asarray(pol, dtype=np.int64)


def _burst_seconds(n_channels, theta, bandwidth):
    """burst 时长：约 BURST_LEN 个事件跨所有通道的期望秒数（解析估计）。

    单通道单步期望 |Δv| ≈ 0.8·(2π·bw·1.23)·FINE_DT（3 谐波信号的
    E|v'|·FINE_DT）；每步总事件率 ≈ C·E|Δv|/θ。
    """
    mean_step = 0.8 * 2 * np.pi * bandwidth * 1.23 * FINE_DT
    per_step = n_channels * mean_step / max(theta, 1e-9)
    steps = max(BURST_LEN / max(per_step, 1e-6), 2.0)
    return min(steps, 500.0) * FINE_DT


def _gap(rng, span_decades):
    """静默间隙：log-uniform 于 [GAP_MIN, GAP_MIN·10^span] 个网格步。"""
    return GAP_MIN * FINE_DT * 10.0 ** rng.uniform(0.0, span_decades)


def _pct(raw_dt, pct):
    """原始秒 Δt 的分位数（realized 跨度报告用）。"""
    return float(np.percentile(raw_dt, pct))


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, default=0.05)
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--bandwidth", type=float, default=2.0)
    ap.add_argument("--channels", type=int, default=6)
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
