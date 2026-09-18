#!/usr/bin/env python3
"""通用 Kaggle CPU sweep 模板（L010 合规）— 把免费 CPU 聚合成可用吞吐.

设计（从 B-2 V15/V16 教训泛化）：
- 4 lane × 单线程（OMP/MKL/OPENBLAS/NUMEXPR 环境钉死，spawn 前导出）
- 配置队列 = commands 列表（每条命令独立、原子落盘自己的输出 → resume-safe）
- 轮询工作窃取：某 lane 空闲即领下一条未完成的命令（12h 硬窗下吞吐最大化）
- 每条命令完成后立即落盘 + 打印进度（L011：日志可事后审计）

用法：sweep.json = {"commands": ["python x.py --a 1 --out o1.json", ...]}
    每条命令须自带 --out 输出到 /kaggle/working/out/<name>.json
"""
import json, os, subprocess, time, glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

WORK = "/kaggle/working"
OUT = f"{WORK}/out"
LANES = 4
spec = json.load(open(f"{WORK}/sweep.json"))
commands = spec["commands"]
os.makedirs(OUT, exist_ok=True)

def out_path(cmd):
    # 约定：命令里必须含 "--out <path>.json"；缺失则用序号
    if "--out" in cmd:
        return cmd.split("--out")[1].strip().split()[0]
    return f"{OUT}/cmd_{abs(hash(cmd)) % 10**8}.json"

pending = [c for c in commands if not os.path.exists(out_path(c))]
print(f"[sweep] 总 {len(commands)} 条，已完成跳过 {len(commands)-len(pending)}，待跑 {len(pending)}", flush=True)

t0 = time.time()
running = {}   # proc -> (cmd, out_path)
next_i = 0
while pending or running:
    # 填满空闲 lane
    while len(running) < LANES and pending:
        cmd = pending.pop(0)
        op = out_path(cmd)
        log = open(f"{WORK}/lane_{len(running)}_{next_i}.log", "w")
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
        running[p] = (cmd, op, log, next_i)
        print(f"[launch #{next_i}] {cmd[:90]}", flush=True)
        next_i += 1
    # 收割完成的
    done = [p for p in running if p.poll() is not None]
    for p in done:
        cmd, op, log, idx = running.pop(p)
        el = (time.time() - t0) / 60
        ok = "ok" if (p.returncode == 0 and os.path.exists(op)) else f"FAIL(rc={p.returncode})"
        print(f"[done #{idx}] {ok} @{el:.0f}min", flush=True)
    if running:
        time.sleep(10)

n_out = len(glob.glob(f"{OUT}/*.json"))
print(f"[sweep done] 输出 {n_out} 个 JSON @ {(time.time()-t0)/60:.0f}min", flush=True)
