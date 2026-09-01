# Pull Request 模板

> 每条必须填写。缺节会被 CI 拒绝。参考 vLLM 贡献政策：AI 生成内容必须披露并人工审查。

---

## 验收标准

<!-- 合入门槛写死。例如：等价性 max_rel_diff < 1e-3；全量 pytest 必须全绿；红线清单 -->

- [ ] 全量 pytest 通过（CI `full-test` 绿）
- [ ] 结果 JSON 通过结构校验（CI `audit-results` 绿）
- [ ] 数值主张可由 CI 一键复算（禁手抄数字）
- [ ] 红线：____

## 实测结果

<!-- 数字必须来自脚本产物，能由 CI 复算。不要手抄。 -->

- 等价性/精度：____（门槛：____）
- 性能：____（复现命令：____）
- pytest：____ passed / ____ skipped（CI 实测）

## 复现路径

<!-- 一键命令 + 产物 JSON 路径 -->

```bash
# 例如：
python benchmarks/kv_frontier.py          # 产物与提交版逐字节一致
python benchmarks/xxx.py --smoke           # 冒烟
python -m pytest tests/test_xxx.py -q      # 定向验收
```

产物：`benchmarks/results/____.json`

## 未完成项

<!-- 待 GPU / 待执行的必须显式声明，否则视为谎报。留空 = 无未完成项 -->

- [ ] 无
- [ ] ____（原因：____）

---

> **AI 协助声明**：本 PR 是否含 AI 生成内容？`是` / `否`
>
> 若含：必须披露生成范围、由人类作者审查确认，并保留 `Co-authored-by:` 标记。
> 未披露的 AI 内容按 vLLM 政策视为不合规，PR 将被关闭。
