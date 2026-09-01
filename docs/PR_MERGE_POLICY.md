# 合入门禁政策

> 生效日期：2026-09-01 · 替代 HANDOFF.md 中散落的口头纪律

## 总则

`main` 分支是已验证过的主干（生产从它部署）。实验分支一条一个假设，
null 结果入档不入主干。任何对 main 的变更必须走 PR，不直推。

## 合入门禁

每次 PR → main 必须满足以下条件，**全部**满足才能合并：

### 1. CI 机器门禁（自动）

| 检查 | 触发条件 | 通过标准 |
|------|---------|---------|
| lint-and-import | PR 创建/更新 | 全部 .py 文件语法正确 + 核心模块可导入 |
| full-test | lint 通过 | pytest 全量绿（当前 ~1400+ tests），无新增失败 |
| audit-results | lint 通过 | kv_frontier 复算逐字节一致 + 结果 JSON 结构校验通过 |

top 3 all green = CI 门禁通过。

### 2. 人工审查

- 至少 1 个 approval（严禁自合）
- 数值主张必须有 CI 可复现的脚本产物（禁手抄数字到 PR 描述）
- AI 协助内容必须披露，标注 `Co-authored-by:`，并已经人类审查

### 3. PR 描述完整性

PR body 必须包含以下四节（模板见 `.github/PULL_REQUEST_TEMPLATE.md`）：

- **验收标准**：合入门槛写死（等价性门槛、pytest 目标、红线清单）
- **实测结果**：须由脚本产出并注明复现命令
- **复现路径**：一键命令 + 产物 JSON 路径
- **未完成项**：待 GPU / 待执行的必须显式声明

缺任何一节 = 视为不完整 PR，不合并。

### 4. 红线（违反即返工）

- 数字手抄 → 脚本重产
- 等价性测试不绿 → 修复后再合
- 2K 数字进 RESULTS.md/README → 删除 （见 MODERN_TRUNK.md §5）
- 未经声明的 AI 内容 → 重新提交

## 合并方式

- 使用 **Squash merge**（每个 PR 压成一个 commit），保持 main 历史干净
- 禁止普通 merge（feature 分支的中间 commit 不进入 main 历史）

## 例外

纯文档/注释变更（无代码行为变化）可免行 full-test，但仍需 lint + audit + 1 approval。

## 参考

本政策参考 vLLM 的 PR 门禁模式（CI 强制 + 数值验证 + 人工审查 + AI 披露）。