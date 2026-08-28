"""event_real_data 解析器的不变量测试（不依赖 4GB 下载数据）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.event_real_data import (US_GRID, _samples_by_class,
                                        _to_tensor, parse_bin)


def _pack(x, y, pol, ts):
    return bytes([x & 0xFF, y & 0xFF,
                  ((pol & 1) << 7) | ((ts >> 16) & 0x7F),
                  (ts >> 8) & 0xFF, ts & 0xFF])


def test_parse_bin_decodes_orchard_events():
    buf = b"".join([_pack(10, 20, 1, 5), _pack(200, 100, 0, 300)])
    ev = parse_bin(io_like(buf))
    assert ev["x"].tolist() == [10, 200] and ev["p"].tolist() == [1, 0]
    assert ev["t"].tolist() == [5, 300] and ev["y"].tolist() == [20, 100]


def test_parse_bin_handles_timestamp_overflow():
    # y==240 是溢出标记：其后所有时间戳 += 2^13
    buf = b"".join([_pack(1, 1, 0, 100), _pack(0, 240, 0, 0),
                    _pack(2, 2, 1, 7)])
    ev = parse_bin(io_like(buf))
    assert len(ev["t"]) == 2                       # 溢出标记本身被丢弃
    assert ev["t"][1] == 7 + 2 ** 13


def test_to_tensor_layout_and_positive_dt():
    ev = {"x": np.array([0, 100, 239, 50]), "y": np.zeros(4, int),
          "p": np.array([1, 0, 1, 1]), "t": np.array([0, 500, 900, 1200])}

    class A:  # 最小 args 替身
        seq_len, channels = 4, 4
    X = _to_tensor(ev, A)
    assert X.shape == (4, 6)
    assert (X[:, :4].argmax(1) == [0, 1, 3, 0]).all()   # x 分箱 one-hot
    assert (X[:, 4] == [1, -1, 1, 1]).all()
    assert X[0, 5] == 0 and (X[1:, 5] > 0).all()        # 首事件 Δt=0，其余 >0


def io_like(raw):
    import io as _io
    return _io.BytesIO(raw)


def test_samples_by_class_picks_second_path_segment():
    # 回归点：类目是路径第 2 段（Caltech101/<class>/x.bin），且按
    # max_per_class 截断、丢弃不在候选类里的成员
    import io as _io
    import zipfile

    buf = _io.BytesIO()
    names = ["Caltech101/accordion/image_0001.bin",
             "Caltech101/accordion/image_0002.bin",
             "Caltech101/faces/image_0001.bin",
             "Caltech101/faces/image_0002.bin",
             "Caltech101/faces/image_0003.bin",
             "meta.csv"]
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            z.writestr(n, b"\x00" * 5)
    z = zipfile.ZipFile(_io.BytesIO(buf.getvalue()))
    by = _samples_by_class(z, {"accordion", "faces"}, 2)
    assert set(by) == {"accordion", "faces"}
    assert len(by["faces"]) == 2                      # max_per_class 截断
    assert all(n.split("/")[1] in by for ns in by.values() for n in ns)
