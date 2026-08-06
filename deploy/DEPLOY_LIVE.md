# 上线 Checklist：让用户打开网页即可试用

> 目标：用一台服务器（或现有 VPS）运行 MT-LNN 推理服务 + 官网 UI，用户访问
> `https://awareliquid.ai` 即可在网页上对话试用。本地已验证：O1-48M 在纯 CPU 上
> 以 ~24 tok/s 运行，API + UI 全部可用（见下方"本地验证记录"）。

---

## 0. 你需要的东西

| 项 | 说明 |
|---|---|
| 服务器 | Vultr / 任意 Linux VPS，**4 vCPU / 8 GB RAM 起**（跑 adapter 需 ~8GB） |
| 域名 | `awareliquid.ai`（已有） |
| checkpoint | `checkpoints/m2_final.pt`（48M O1，从 Kaggle M2 run 精简的 serve.pt） |
| 可选 adapter | `checkpoints/llama_mt_adapter/llama_mt_adapter_v2s_003000.pt`（24MB，TinyLlama-1.1B + MT v2s，多语言指令跟随） |

---

## 1. 服务器准备

```bash
# Ubuntu 22.04/24.04
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo systemctl enable --now docker
```

---

## 2. 拉代码 + 放 checkpoint

```bash
cd ~
git clone https://github.com/everest-an/M1.git
cd M1

# 上传 checkpoint（本地已跑通的是 o1_48m_serve.pt；线上用 m2_final.pt 同构）
# 本地: scp checkpoints/o1_48m_serve.pt user@server:~/M1/checkpoints/m2_final.pt
mkdir -p checkpoints
# (把 .pt 文件放进 checkpoints/，命名为 m2_final.pt)

# 可选: adapter checkpoint
mkdir -p checkpoints/llama_mt_adapter
# (把 llama_mt_adapter_v2s_003000.pt 放进去)
```

---

## 3. 启动（核心推理服务）

```bash
cd ~/M1
docker compose -f deploy/docker-compose.prod.yml up -d --build
docker compose -f deploy/docker-compose.prod.yml logs -f   # 等 "Application startup complete"
```

验证：

```bash
curl http://localhost:8000/health          # -> {"status":"ok",...}
curl -X POST localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"prompt":"Tell me a short story about a robot.","max_new_tokens":80}'
```

---

## 4. 可选：启用 M1 适配器（体验更好，多语言）

```bash
docker compose -f deploy/docker-compose.prod.yml --profile adapter up -d adapter
curl -X POST localhost:8001/v1/completions -H 'content-type: application/json' \
     -d '{"prompt":"讲一个关于机器人的小故事。","max_new_tokens":80}'
```

前端模型菜单可切换：48M 原生 O1（快） vs TinyLlama-1.1B + MT v2s（会聊天）。

---

## 5. DNS + HTTPS（Caddy 自动证书）

1. 域名提供商加两条 A 记录 → 服务器 IP：
   - `awareliquid.ai`
   - `www.awareliquid.ai`
2. 防火墙放行 80/443：
   ```bash
   sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw enable
   ```
3. Caddy（compose 内已配置）会自动申请 Let's Encrypt 证书。

完成 → **https://awareliquid.ai 用户打开即可试用**。

---

## 6. 上线后更新（改 UI / 图 / 代码）

```bash
cd ~/M1
git pull                        # 拉新代码（含 serve/static/ 的 UI 改动）
docker compose -f deploy/docker-compose.prod.yml restart mtlnn     # 静态 UI 无需重建镜像
docker compose -f deploy/docker-compose.prod.yml --profile adapter restart adapter
```

换 checkpoint：

```bash
# 替换 checkpoints/m2_final.pt 后
docker compose -f deploy/docker-compose.prod.yml restart mtlnn
```

---

## 7. 备份与回滚

- **checkpoint**：`checkpoints/` 整个目录定期 rsync 到本地/对象存储
- **adapter 回滚**：compose 里 `ADAPTER_CKPT` 指回 `llama_mt_adapter_003000.pt`（v1）重启即可
- **数据**：`data/partners/` 是运行时数据（推荐计数），bind-mount 在宿主机上，重建容器不丢

---

## 8. 资源预算参考（经验值）

| 服务 | RAM | CPU | 说明 |
|---|---|---|---|
| mtlnn（48M O1） | ~1 GB | 1-2 vCPU | 24 tok/s CPU（本地实测） |
| adapter（1.1B+MT） | ~6-7 GB | 4 vCPU | 慢但可用；与 mtlnn 同机需 ≥8GB |
| Caddy | 小 | 忽略 | TLS 反代 |

> 若用户量大：adapter 单独上 GPU 实例；O1-48M 保持 CPU 即可。
> 若追求体验：把 `m2_final.pt` 换成 125M checkpoint（`_o1_ckpt/final.pt`）质量更好，CPU 仍可跑。

---

## 本地验证记录（2026-08-07）

Windows 本地验证通过（与线上同代码）：

```
CKPT_PATH=checkpoints/o1_48m_serve.pt TOKENIZER=gpt2 DEVICE=cpu \
  python -m uvicorn serve.server:app --host 127.0.0.1 --port 8123
```

| 检查项 | 结果 |
|---|---|
| `GET /health` | `{"status":"ok","device":"cpu","multimodal":false}` |
| `POST /v1/completions` | 40 tokens / 1.63s = **24.5 tok/s** |
| `GET /` 首页 | 200 · 107 KB · 标题 "AwareLiquid — Bio-inspired Cognitive AI Framework" |
| `GET /research` | 200 · 含新图引用（selective_copy_samesize / cost_capability_positioning） |
| 新 SVG 静态图 | 200（15 KB / 33 KB） |
| about / demo / api 页 | 全 200 |

> ⚠️ Windows 注意：8000/8010 端口可能被系统占用（本机实测 10048 绑定失败），
> 本地验证用了 8123。服务器（Linux）无此问题，用默认 8000 即可。
