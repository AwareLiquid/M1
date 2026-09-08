"""从逐臂原始 arr_result.json 重建 ARR 账本（绕过子集运行的 control=None bug）。

背景：arr_ratio_sweep.py 的 _ratio_row 无条件计算对 ratio-0 对照的增益，
任何不含 ratio 0 的子集调用（如分片跑）在收尾写盘时必然 TypeError
（traceback 存 w_arr_pbA.log）。本脚本用 sweep 自己的 _record/_ledger/_write
把逐臂原始产物重建成规范账本 —— 判定逻辑零改动，幂等可重跑。
用法: python benchmarks/rebuild_arr_ledger.py [repo_root]
  repo_root 默认 = 本文件所在仓库（本机跑用 worktree 路径即可）。
"""
import sys, os, json, glob, importlib.util

repo = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "sweep", os.path.join(repo, "benchmarks", "arr_ratio_sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

sys.argv = [os.path.basename(sys.argv[0]),
            "--ratios", "0,1:4,1:8,1:2", "--seeds", "0,1,2"]
args = sweep._parse_args()

runs, skipped = [], []
for f in sorted(glob.glob(os.path.join(
        repo, "benchmarks", "arr_ratio_out", "ratio_*_seed*", "arr_result.json"))):
    base = os.path.basename(os.path.dirname(f))[len("ratio_"):]
    tag, seed = base.rsplit("_seed", 1)
    res = json.load(open(f))
    v = res.get("student_ppl_final")
    if v != v:  # NaN 臂剔除（发散如实记录于 commit/日志，不进均值）
        skipped.append(f"{tag} seed {seed}")
        continue
    ratio = tag if tag.isdigit() else tag.replace("_", ":", 1)
    runs.append(sweep._record(ratio, int(seed), res, args))
if skipped:
    print("[rebuild] NaN 臂已剔除:", ", ".join(skipped))

ledger = sweep._ledger(runs, args)
out = os.path.join(repo, "benchmarks", "results", "rebuilt")
sweep._write(ledger, out)
print("[rebuild] promotion:", json.dumps(ledger["promotion"], ensure_ascii=False))
for r in ledger["rows"]:
    print(f"[rebuild] ratio {r['ratio']:>4} seeds={r['seeds']} "
          f"mean={r['val_ppl']} gain={r['ppl_gain_vs_control']}")
print("[rebuild] 写入", out)
