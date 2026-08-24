# M1-2B 发布管线（权重未就绪，先备好流程）

> 状态：**准备完成，待权重达标** · 2026-08-25
> 触发条件：data_slim 续训后 val PPL 进入可用区间，由你拍板发布。
> 当前 checkpoint（ckpt_120000，val PPL 77.09）**不建议发布**——远未收敛，发出去是负资产。

## 0. 前置确认

- [ ] 训练机可达（`py -3.11 scripts/download_remote.py --list /home/user/M1/checkpoints/`）。
      **当前 65.49.232.182:10012 SSH 不通**——需先确认开机/换 IP，并更新
      `scripts/download_remote.py` 的 HOST。
- [ ] 本地磁盘：每个 2B checkpoint ≈ **20.3 GB**（fp32），下载到 `E:\M1\checkpoints\`
      （E 盘，C 盘空间紧张）。
- [ ] 选一个里程碑 checkpoint（以 val PPL 为准，参考 archive_main.py 的清单）。

## 1. 下载

```powershell
py -3.11 scripts/download_remote.py /home/user/M1/checkpoints/ckpt_XXXXXX.pt E:\M1\checkpoints\ckpt_XXXXXX.pt
# 或整批归档（含 curriculum jsonl / 日志）：
py -3.11 scripts/archive_main.py
```

## 2. 校验（发布前必跑）

```powershell
py -3.11 scripts/verify_2b_checkpoint.py E:\M1\checkpoints\ckpt_XXXXXX.pt --json
# 通过标准: config 2912d×35L + 参数 1.7-2.2B + missing 关键层为空
# 可选冒烟生成 (CPU 慢, 每 token 数十秒):
py -3.11 scripts/verify_2b_checkpoint.py E:\M1\checkpoints\ckpt_XXXXXX.pt --tokens 8
```

## 3. GitHub Release

- Tag：`m1-2b-v1`（后续里程碑递增 v2/v3…）
- 资产：checkpoint `.pt` + `docs/release-m1-2b-notes.md` 渲染后的说明
- 发布命令（替代 gh release create 的资产上传）：

```powershell
gh release create m1-2b-v1 E:\M1\checkpoints\ckpt_XXXXXX.pt `
  -R AwareLiquid/M1 -F E:\M1\docs\release-m1-2b-notes.md -t "M1-2B (milestone)"
```

## 4. Hugging Face

```powershell
# dry-run 先看计划与 README 模板渲染
py -3.11 scripts/upload_hf_m1_2b.py E:\M1\checkpoints\ckpt_XXXXXX.pt --dry-run
# 实跑（需 HF_TOKEN 或已 login; 仓库 AwareLiquid/M1-2B, MIT, public）
py -3.11 scripts/upload_hf_m1_2b.py E:\M1\checkpoints\ckpt_XXXXXX.pt
```

## 5. 代码镜像同步

本次发布若带了代码改动（trainer/配置），跑 `scripts/sync_oss.ps1` 同步公开镜像。
**注意**：`download_remote.py` / `archive_main.py` 已加入排除名单（含训练机凭据），
不会进公开镜像。

## 6. 官网更新（AwareLiquid-Web 仓库，不是 M1 仓库）

流程：改 `web/` → 提交推送 → 服务器 `cd /root/AwareLiquid-Web && git pull --ff-only
origin main && docker restart mtlnn_prod`。

**待发布时替换的卡片文案**（现为 Training 状态，发布后切 download）：

- `web/index.html` 卡片 mc.3（HTML + EN i18n + ZH i18n 三处）与 `web/models.html`
  m4（HTML + EN + ZH 三处）：
  - desc 替换为：`1.9B-parameter hybrid (2912d × 35L). Trained from scratch on the
    data_slim corpus. [发布时填最终 PPL/步数]. Weights on GitHub Releases + HF.`
  - meta：`converging · PPL -30%` → `<最终 PPL>`，状态徽章 Training → Download
  - cta：`Weights when it converges` → 下载链接（Release + HF）
- `web/models.html` 对比表 M1-2B 行：状态 ◐ Training → ↓ Download，Window/State 补实值
- `web/research.html`：训练曲线小节补最终数字（照 S2 小节样式）

## 7. 演示部署（评估后再定）

- **不建议在现有 VPS 上跑**：1.9B fp32 ≈ 7.6 GB 权重，CPU 推理每 token 数十秒，
  且 VPS 内存可能不足。
- 可选路径：租短时 GPU 实例起 `serve/server.py` 做限时演示；或等量化（int8/int4）
  后再上 VPS。**发布 ≠ 必须上线演示**——128M 与 adapter 演示已在线上。

## 8. 风险与边界（红线）

- 发布前必须满足 §0 触发条件；"训练中里程碑" 发布要显式标注
  （README 模板已内置 honest boundaries）。
- M3 私有材料（P0' 裁决、MoE/P1/L3 结果）**不进入任何 2B 发布文案**。
- 官网卡片不得在权重未上传前切 download 状态。
- 训练机密码已写死在 download_remote.py（私有仓库内）——轮换后记得同步更新。
