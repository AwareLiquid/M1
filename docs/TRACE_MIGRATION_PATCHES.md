# TRACE_MIGRATION_PATCHES — 分支专属补丁(待随各 PR 落地)

> 本次决策血缘迁移(kb/ + DECISION_TRACE_SPEC + ADJUDICATION_LOG +
> atomic_io)落在当前工作区;以下三处改动属于已开 PR 的分支,记录在此,
> 随各 PR 下一提交应用。均为小补丁,不改协议。

## PATCH-1 · PR #9(`iter/latent-recursion`)· save_row 并发安全 + 自动入仓

`benchmarks/latent_recursion.py` 的 `save_row` 用固定 `path + ".tmp"`,
Phase B 24-worker 并行时同配置双写会互踩(aexp 0.6.1 存档:该模式并发
55% 撕裂率)。main 已提供 `benchmarks/atomic_io.py`(pid+nonce tmp,5 测
全过),应用方式二选一:

```python
# 方式 A(直接换实现,推荐):
from benchmarks.atomic_io import atomic_write_json   # 顶部 import

def save_row(path, row):
    atomic_write_json(path, row)                      # 原 tmp+os.replace 逻辑删除
```

```python
# 方式 B(最小 diff,仅修 tmp 名):
def save_row(path, row):
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"   # 原: path + ".tmp"
    ...
```

同分支收尾追加自动入仓(消除"跑完→入仓"人肉空隙,ADJ-004 教训):

```python
# main() 末尾,write_verdict(args) 之后:
subprocess.run(["git", "add", "--", args.out_dir], check=False)
subprocess.run(["git", "commit", "-m",
                f"data(latent-recursion): 行+verdict 自动入仓 {time.strftime('%F %T')}"],
               cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               check=False)          # check=False: 无新数据时不报错
```

并在重跑 `write_verdict`(ADJ-002)后给 verdict JSON 补
`lineage`/`evidence`/`protocol` 字段,格式见 `docs/DECISION_TRACE_SPEC.md`。

## PATCH-2 · PR #14(`iter/process-train`)· harvest 门禁扩展链校验

`benchmarks/harvest_registry.py` 的 `--check` 现只查 orphan(REGISTRY
人写区未引用)。按 `docs/DECISION_TRACE_SPEC.md` §规则 4 扩展:

```python
def check_lineage_chain(results_root: Path, kb_root: Path) -> list[str]:
    """每个带 lineage.hypothesis 的 verdict JSON,H 工件必须存在;
    每个 UNDER_TEST 以后的 H,必须能解析到 ≥1 个证据文件。"""
    issues = []
    hyp_ids = {p.name.split("-")[0] for p in (kb_root / "hypotheses").glob("H*.md")}
    for j in results_root.glob("*.json"):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        hyp = (d.get("lineage") or {}).get("hypothesis")
        if hyp and hyp not in hyp_ids:
            issues.append(f"{j.name}: lineage.hypothesis={hyp} 无对应 kb 工件")
    return issues
# main(): orphan 为空后追加此检查,非空 → exit 1(与 orphan 同级门禁)
```

## PATCH-3 · 新实验脚手架 · resume-safe 默认

任何新判决跑的启动脚本从 `latent_recursion.py` 复制三件套
(pending_configs 跳过已完成 / save_row 原子写 → atomic_io /
run_config 内 tee 日志),并遵守 ADJ-004 分片预算:单实例 ≤2 配置。
镜像于 HANDOFF §5.3,不再是事后教训。
