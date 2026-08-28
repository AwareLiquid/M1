"""event_stream 生成器的数学/语义不变量测试（锁定 docstring 承诺）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.event_stream import (FINE_DT, _emit, _latent,
                                     events_to_tensor, generate_episode,
                                     make_dataset)


def test_realized_dt_span_grows_with_span_knob():
    spans, realized = [1.0, 3.0], []
    for s in spans:
        ds = make_dataset(6, 6, 64, 0.05, s, 2.0, seed=0)
        realized.append(ds["dt_span_decades"])
    assert realized[1] > realized[0] + 0.5       # 跨度随旋钮显著扩大


def test_timestamps_strictly_increasing():
    rng = np.random.default_rng(0)
    t, c, p, _ = generate_episode(rng, 6, 0.05, 2.0, 2.0, min_events=10)
    assert (np.diff(t) > 0).all()          # 生成器已排序 + ε 偏移


def test_emit_threshold_semantics():
    # |Δv| < θ 永不触发：bandwidth=0 的潜信号精确常数（峰峰值 0）+ 微幅正弦
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 1.0, FINE_DT)
    v = _latent(rng, t, 4, 0.0)
    assert np.allclose(v, v[:, :1], atol=1e-12)
    assert len(_emit(v[0], 1e-6)[0]) == 0
    tiny = 1e-4 * np.sin(np.linspace(0, 100, 5000))   # 峰峰值 2e-4 << 0.5
    assert len(_emit(tiny, 0.5)[0]) == 0
    # 极性跟随变化方向：单调升全 +1，单调降全 -1
    ramp = np.linspace(0.0, 10.0, 100)
    assert (_emit(ramp, 0.5)[1] == 1).all()
    assert (_emit(-ramp, 0.5)[1] == -1).all()


def test_theta_zero_degenerates_to_regular_sampling():
    v = np.sin(np.linspace(0, 100, 2000))  # 平滑信号, 每步变化非零
    idx, _ = _emit(v, 0.0)
    assert len(idx) == len(v) - 1          # 每个网格步都触发
    assert (np.diff(idx) == 1).all()       # Δt 恒等于网格步长


def test_events_to_tensor_layout_and_padding():
    n_ch, T = 5, 16
    times = np.arange(1, 9) * FINE_DT
    chans = np.array([0, 3, 3, 1, 1, 1, 2, 4])
    pol = np.array([1, -1, 1, 1, -1, 1, -1, 1])
    X = events_to_tensor(times, chans, pol, n_ch, T)
    assert X.shape == (T, n_ch + 2)
    assert X[8:].sum() == 0                                   # padding 全零
    assert (X[:8, :n_ch].argmax(axis=1) == chans).all()       # one-hot 通道
    assert (X[:8, n_ch] == pol).all()                         # 极性
    assert (X[1:8, -1] > 0).all() and X[0, -1] > 0            # Δt 为正


def test_make_dataset_shapes_and_finiteness():
    ds = make_dataset(4, 4, 32, 0.1, 1.0, 2.0, seed=0)
    assert ds["X"].shape == (4, 32, 6)
    assert np.isfinite(ds["X"]).all() and np.isfinite(ds["y"]).all()
    assert 0 < ds["kept_fraction"] <= 1
