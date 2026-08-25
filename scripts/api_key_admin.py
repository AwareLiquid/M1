"""api_key_admin.py — API key 签发/列表/吊销（在服务器主机上运行）

明文 key 只在 add 时打印一次; 库内仅存 SHA-256。

用法:
  py -3.11 scripts/api_key_admin.py add --label acme-corp --quota 10000 --ttl-days 90
  py -3.11 scripts/api_key_admin.py list
  py -3.11 scripts/api_key_admin.py revoke --label acme-corp

环境: API_KEYS_DB (默认 <repo>/data/api_keys.db; 生产容器建议
       /app/data/partners/api_keys.db —— RW 挂载树内, 重建容器不丢)
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.api_auth import ApiKeyStore  # noqa: E402


def _store() -> ApiKeyStore:
    db = os.environ.get("API_KEYS_DB", "").strip()
    if not db:
        db = str(Path(__file__).resolve().parents[1] / "data" / "api_keys.db")
    return ApiKeyStore(db)


def _fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "—"


def cmd_add(args):
    store = _store()
    key = store.issue(args.label, max_requests=args.quota,
                      ttl_days=args.ttl_days)
    print("═══ 新 API key（明文仅显示这一次，请立即复制给客户）═══")
    print(key)
    print("══════════════════════════════════════════════════════")
    print(f"label={args.label} quota={args.quota or 'unlimited'} "
          f"ttl_days={args.ttl_days or 'never'}")


def cmd_list(args):
    store = _store()
    rows = store.list_keys()
    if not rows:
        print("(空)")
        return
    print(f"{'label':<24} {'used/max':<12} {'expires':<18} "
          f"{'last_used':<18} revoked  key_hash")
    for r in rows:
        used = r["requests_used"]
        mx = r["max_requests"] if r["max_requests"] is not None else "∞"
        print(f"{r['label']:<24} {used}/{mx:<10} "
              f"{_fmt_ts(r['expires_at']):<18} {_fmt_ts(r['last_used_at']):<18} "
              f"{'YES' if r['revoked'] else 'no':<8} {r['key_hash'][:12]}…")


def cmd_revoke(args):
    store = _store()
    n = store.revoke(args.label)
    print(f"revoked {n} key(s) for label={args.label!r}")
    if n == 0:
        print("(label 不存在或已吊销)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="签发新 key（明文打印一次）")
    p_add.add_argument("--label", required=True)
    p_add.add_argument("--quota", type=int, default=None, help="最大请求数 (默认不限)")
    p_add.add_argument("--ttl-days", type=float, default=None, help="有效天数 (默认不过期)")
    p_add.set_defaults(fn=cmd_add)

    p_list = sub.add_parser("list", help="列出全部 key")
    p_list.set_defaults(fn=cmd_list)

    p_rev = sub.add_parser("revoke", help="按 label 吊销")
    p_rev.add_argument("--label", required=True)
    p_rev.set_defaults(fn=cmd_revoke)

    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
