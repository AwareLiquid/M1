"""event_real_data.py — 真实事件数据集接入（尽力项）：N-Caltech101。

数据：N-Caltech101（Orchard et al. 2015，事件相机拍摄 Caltech101 图片，
saccade 扫视生成事件）。无鉴权下载（Mendeley 公开文件，URL 与 md5 来自
tonic 库源码），~4 GB。嵌套 zip（archive → Caltech101.zip → 每类每样本
一个 .bin）：外层 zip 只抽内层 zip 到磁盘，内层 zip **不落盘解压**，
直接从 zip 成员内存读取（磁盘峰值 = 4 GB zip 本体）。

.bin 格式（Orchard 5 字节事件，与 tonic.io.read_mnist_file 同口径）：
x, y, pol(bit7)+ts[18:16], ts[15:8], ts[7:0]；y==240 为时间戳溢出标记
（其后所有 ts += 2^13 µs）。

协议：held-out 类别。任务 = 下一事件极性预测（二分类，chance=0.5）——
极性序列由物理（亮/暗变化交替）驱动，可跨类别迁移；通道 = x 坐标
分箱（C=6）。Δt（真 µs→s，log 压缩）喂所有架构（护栏沿用）。
对比：mt_lnn vs Task 2 最强离散基线（gru）。

下载/解析失败 → JSON 记 SKIP + 原因，不造假数字。

    python benchmarks/event_real_data.py --smoke
    python benchmarks/event_real_data.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks.battery_soh_edge import build
from benchmarks.experiment_protocol import publishable

URL = ("https://data.mendeley.com/public-files/datasets/cy6cvx3ryv/files/"
       "36b5c52a-b49d-4853-addb-a836a8883e49/file_downloaded")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "n_caltech101")
ARCHIVE = os.path.join(DATA_DIR, "N-Caltech101-archive.zip")
INNER = os.path.join(DATA_DIR, "Caltech101.zip")
N_CLASSES_USE = 20          # 前 N 个类别（字典序），一半训练一半 held-out
US_GRID = 1e-6              # Δt 压缩的参考网格：1 µs


def main():
    args = _cli()
    try:
        tr, te = load_real_tensors(args)
    except Exception as e:                       # 尽力项：失败如实 SKIP
        _skip(args, e)
        return 0
    rows = {}
    for arch in ("mt_lnn", "gru"):
        vals = [_train_eval(arch, tr, te, args, s) for s in args.seeds]
        rows[arch] = vals
        print(f"  {arch:<8} acc {np.mean(vals):.4f} ± "
              f"{np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}")
    ok, rep = publishable(rows["mt_lnn"], rows["gru"],
                          tag="ncaltech-polarity|mt_lnn vs gru")
    _save(args.out, tr["meta"], te["meta"], rows, ok, rep)
    return 0


def load_real_tensors(args):
    """外层 zip → 内层 zip（不落盘解压）→ 每类样本 .bin → 张量。"""
    _ensure_archive()
    if not os.path.exists(INNER):
        _extract_inner()
    classes = _pick_classes(N_CLASSES_USE)
    tr_cls, te_cls = classes[:len(classes) // 2], classes[len(classes) // 2:]
    tr = _class_tensors(tr_cls, args)
    te = _class_tensors(te_cls, args)
    print(f"train classes {len(tr_cls)}, held-out {len(te_cls)}; "
          f"samples {len(tr['X'])}/{len(te['X'])}")
    return tr, te


def _class_tensors(classes, args):
    """多类样本 → (N, T, C+2) 张量；每样本截前 T 个事件。"""
    with zipfile.ZipFile(INNER) as z:
        by_cls = _samples_by_class(z, classes, args.max_per_class)
        Xs, ch_meta = [], []
        for cls, names in by_cls.items():
            for n in names:
                ev = parse_bin(io.BytesIO(z.read(n)))
                if len(ev) < args.seq_len:
                    continue
                Xs.append(_to_tensor(ev, args))
                ch_meta.append(cls)
    return {"X": np.stack(Xs), "cls": ch_meta,
            "meta": {"n": len(Xs), "classes": sorted(set(ch_meta))}}


def _samples_by_class(z, classes, max_per_class):
    """内层 zip 里按类目（路径首段）挑前 max_per_class 个 .bin 样本。"""
    by = {}
    for n in sorted(z.namelist()):
        if not n.endswith(".bin"):
            continue
        cls = n.split("/")[0]
        if cls in classes and len(by.get(cls, [])) < max_per_class:
            by.setdefault(cls, []).append(n)
    return {c: ns for c, ns in by.items() if ns}


def _to_tensor(events, args):
    """(T, C+2)：x 分箱 one-hot + 极性 ±1 + log10(1+Δt/µs)。

    .bin 事件本身按时间序；取前 T 个（saccade 开扫段）。
    """
    T = args.seq_len
    x, p, ts = events["x"][:T], events["p"][:T], events["t"][:T]
    C = args.channels
    X = np.zeros((T, C + 2), dtype=np.float32)
    bins = np.clip((x * C / 240).astype(int), 0, C - 1)
    X[np.arange(T), bins] = 1.0
    X[:, C] = np.where(p > 0, 1.0, -1.0)
    dt = np.diff(ts, prepend=ts[0]).astype(np.float64)
    X[:, C + 1] = np.log10(1.0 + dt / US_GRID)
    return X


def parse_bin(buf):
    """Orchard 5 字节事件流 → x/y/p/t（含 y==240 时间戳溢出处理）。"""
    raw = np.frombuffer(buf.getvalue(), dtype=np.uint8).astype(np.uint32)
    x, y5 = raw[0::5], raw[1::5]
    pol = (raw[2::5] & 128) >> 7
    ts = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]
    over = np.where(y5 == 240)[0]
    for i in over:
        ts[i:] += 2 ** 13
    keep = y5 != 240
    return {"x": x[keep].astype(np.int64), "y": y5[keep].astype(np.int64),
            "p": pol[keep].astype(np.int64), "t": ts[keep].astype(np.int64)}


def _ensure_archive():
    if not os.path.exists(ARCHIVE):
        raise RuntimeError(
            f"missing {ARCHIVE} — 先运行: curl -L -C - -o {ARCHIVE} '{URL}'")


def _extract_inner():
    with zipfile.ZipFile(ARCHIVE) as outer:
        member = [n for n in outer.namelist() if n.endswith("Caltech101.zip")]
        if not member:
            raise RuntimeError("archive 内未找到 Caltech101.zip")
        with outer.open(member[0]) as src, open(INNER, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)


def _pick_classes(n_use):
    with zipfile.ZipFile(INNER) as z:
        classes = sorted({n.split("/")[0] for n in z.namelist()
                          if n.endswith(".bin")})
    if len(classes) < n_use:
        raise RuntimeError(f"classes {len(classes)} < {n_use}")
    return classes[:n_use]


def _train_eval(arch, tr, te, args, seed):
    """下一事件极性二分类：输入前 T-1 事件预测第 T 个极性。"""
    torch.manual_seed(seed)
    m = build(arch, args.d_model, args.n_layers, args.seq_len)
    m.inp = nn.Linear(tr["X"].shape[-1], args.d_model)
    m.head = nn.Linear(args.d_model, 2)
    _train(m, *_views(tr), args)
    with torch.no_grad():
        pred = m(torch.from_numpy(_views(te)[0])).argmax(1).numpy()
    return float((pred == _views(te)[1]).mean())


def _train(m, Xtr, ytr, args):
    """AdamW + 梯度裁剪的交叉熵训练循环。"""
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for _ in range(args.epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = nn.functional.cross_entropy(m(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()


def _views(ds):
    X = ds["X"]
    C = X.shape[-1] - 2
    return (X[:, :-1, :], (X[:, -1, C] > 0).astype(np.int64))


def _skip(args, err):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"status": "SKIP", "reason": str(err),
                   "gru_d": "pending merge (iter/irregular-streaming-edge)"}, f,
                  indent=2)
    print(f"SKIP: {err}\nresults -> {args.out}")


def _save(path, tr_meta, te_meta, rows, ok, rep):
    with open(path, "w") as f:
        json.dump({"status": "OK", "task": "next-polarity, held-out classes",
                   "train_meta": tr_meta, "test_meta": te_meta, "rows": rows,
                   "e0": {"ok": ok, "reasons": rep["reasons"],
                          "sign_test": rep.get("sign_test", {})},
                   "gru_d": "pending merge (iter/irregular-streaming-edge)"},
                  f, indent=2)
    print(f"results -> {path}")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--max-per-class", type=int, default=5)
    ap.add_argument("--d_model", type=int, default=52)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="benchmarks/results/event_ncaltech101.json")
    args = ap.parse_args()
    args.seeds = [0] if args.smoke else [0, 1, 2]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
