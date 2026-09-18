#!/usr/bin/env python3
"""t4/t5 + t1 难格判读：汇总 parametric/micro_fw/micro_rls/rag/e5/oracle 的
结果 JSON，输出每系统关键格均值。无参数，直接跑。"""
import json, statistics, os

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load(path):
    p = os.path.join(RES, path)
    if not os.path.exists(p):
        return None
    return json.load(open(p))

def cell(doc, task, match):
    if doc is None:
        return None
    t = doc["results"][task]
    for c in t.get("grid", t.get("curve", [])):
        if all(c.get(k) == v for k, v in match.items()):
            return c["recall"]
    return None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None

SYSTEMS = [
    # (显示名, 结果目录, t1/t4/t5 文件名模板中的系统段)
    # micro_* 映射到 TOKEN_RE 修复后权威族（eb01dfc）；修复前族（microfw/
    # microrls，8892f40）已被勘验条目废止，勿再引用。
    ("micro_rls", "pmb_t45", "micro_rls"),
    ("micro_fw", "pmb_t45", "micro_fw"),
    ("param(falcon)", "pmb_t45", "param_falcon_nlms"),
    ("param(sum)", "pmb_t45", "param_sum"),
    ("rag(hash)", "pmb_t45", "rag"),
    ("rag_2hop", "pmb_t45", "rag_2hop"),
    ("rag(e5)", "pmb_e5", "rag"),
    ("oracle", "pmb_t45", "oracle"),
    ("none", "pmb_t45", "none"),
]

CHECKS = [
    ("t1(4,0)", "t1", {"n": 4, "k": 0}),
    ("t1(64,16)", "t1", {"n": 64, "k": 16}),
    ("t4(4,0)", "t4", {"m": 4, "k": 0}),
    ("t4(8,4)", "t4", {"m": 8, "k": 4}),
]

rows = []
for disp, folder, sysseg in SYSTEMS:
    row = {"系统": disp}
    for label, task, match in CHECKS:
        vals = []
        for s in [0, 1, 2]:
            doc = load(os.path.join(folder, f"pmb_{task}_{sysseg}_s{s}.json"))
            vals.append(cell(doc, task, match))
        row[label] = mean(vals)
    # t5
    rk, af = [], []
    for s in [0, 1, 2]:
        doc = load(os.path.join(folder, f"pmb_t5_{sysseg}_s{s}.json"))
        if doc:
            t5 = doc["results"]["t5"]
            rk.append(t5.get("recall_known"))
            af.append(t5.get("abstain_false_rate"))
        else:
            rk.append(None); af.append(None)
    row["t5known"] = mean(rk)
    row["t5幻答"] = mean(af)
    rows.append(row)

hdr = ["系统"] + [c[0] for c in CHECKS] + ["t5known", "t5幻答"]
print(f"{'系统':14} " + " ".join(f"{h:>10}" for h in hdr[1:]))
for r in rows:
    print(f"{r['系统']:14} " + " ".join(
        (f"{r[h]:>10}" if isinstance(r[h], float) else f"{'-':>10}") for h in hdr[1:]))
