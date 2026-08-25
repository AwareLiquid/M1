# API Key 鉴权（轻量版）

`serve/server.py` 的可选接入层：API key + 匿名限流。默认 **off**（不鉴权，
行为与之前完全一致），需要开放给客户时再开启。

## 三种模式（环境变量 `API_AUTH_MODE`）

| 模式 | 无 key 请求 | 带 key 请求 |
|---|---|---|
| `off`（默认） | 放行 | 放行 |
| `soft` | 放行但按 IP 限流（`API_ANON_PER_MIN`，默认 20/分钟） | 按 key 配额 |
| `strict` | 401 | 按 key 配额 |

key 通过请求头 `X-API-Key` 传入。`/health`、`/v1/model`、静态页面不受鉴权影响
（strict 下 `/v1/model` 也要 key，因为它也在 `/v1/` 前缀内——如需放开可调整）。

## 签发 / 管理

在服务器主机上运行（明文 key 只在 add 时打印一次，库内仅存 SHA-256）：

```bash
# 签发：10000 次配额，90 天有效
py -3.11 scripts/api_key_admin.py add --label acme-corp --quota 10000 --ttl-days 90

# 列出全部 key（用量/过期/吊销状态）
py -3.11 scripts/api_key_admin.py list

# 吊销
py -3.11 scripts/api_key_admin.py revoke --label acme-corp
```

`API_KEYS_DB` 环境变量指定库位置（默认 `<repo>/data/api_keys.db`）。
**生产容器注意**：容器内 `/app/data/api_keys.db` 不持久（重建即丢），
建议设 `API_KEYS_DB=/app/data/partners/api_keys.db`（RW 挂载树内，跟随宿主机）。

## 生产开启步骤（示例：soft 模式）

1. 签发 key：`API_KEYS_DB=/app/data/partners/api_keys.db py -3.11 scripts/api_key_admin.py add --label <客户名> --quota <N>`
2. compose 环境加 `API_AUTH_MODE: soft`（`API_ANON_PER_MIN` 可选）
3. `docker compose -f deploy/docker-compose.prod.yml up -d mtlnn`（或重启容器）
4. 验证：
   ```bash
   # 无 key：限流内应正常
   curl -s -X POST https://awareliquid.ai/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"hello","max_new_tokens":4}'
   # 带 key：正常
   curl -s -X POST https://awareliquid.ai/v1/completions -H 'X-API-Key: al_…' -H 'Content-Type: application/json' -d '{"prompt":"hello","max_new_tokens":4}'
   # strict 模式无 key：应 401
   ```

## 边界（诚实）

- 这是**单进程内存限流 + SQLite 配额**，为少量企业客户设计；不是多副本
  分布式限流，不适合大规模公开 API 场景。
- 匿名限流按 IP（x-forwarded-for 首项），NAT 后多人共用 IP 会被一起限。
- key 配额按**请求数**计，不按 token 数。
- 完整账号体系（注册/登录/自助 key/用量仪表盘）见量再上。
