"""api_auth.py — API key 鉴权 + 匿名限流（serve/server.py 的可选接入层）

纯逻辑模块，不依赖 torch/FastAPI，可独立单测。

模式（环境变量 API_AUTH_MODE，默认 off 保持现状零变化）:
  off    - 不鉴权（demo 友好，默认）
  soft   - 匿名请求允许但按 IP 限流；带 key 走配额
  strict - /v1/* 必须带 X-API-Key，否则 401

Key 格式: al_ + 32 hex（secrets.token_hex(16)），库内只存 SHA-256。
"""
import hashlib
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from typing import Optional, Tuple

KEY_PREFIX = "al_"

# 检查失败的判定原因（同时作为 HTTP 响应的 detail 文案）
REASON_OK = "ok"
REASON_INVALID = "invalid API key"
REASON_REVOKED = "API key revoked"
REASON_EXPIRED = "API key expired"
REASON_QUOTA = "API key quota exceeded"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ApiKeyStore:
    """SQLite 存储: key_hash -> (label, quota, usage, expiry, revoked)。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()  # 序列化读写; 请求量小, 单锁足够
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_hash      TEXT PRIMARY KEY,
                        label         TEXT NOT NULL,
                        max_requests  INTEGER,          -- NULL = 不限
                        requests_used INTEGER NOT NULL DEFAULT 0,
                        expires_at    REAL,             -- NULL = 永不过期
                        created_at    REAL NOT NULL,
                        last_used_at  REAL,
                        revoked       INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )

    def issue(self, label: str, max_requests: Optional[int] = None,
              ttl_days: Optional[float] = None) -> str:
        """签发新 key, 返回明文 (仅此一次可见)。"""
        plaintext = KEY_PREFIX + secrets.token_hex(16)
        now = time.time()
        expires = now + ttl_days * 86400.0 if ttl_days else None
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO api_keys "
                    "(key_hash, label, max_requests, expires_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (hash_key(plaintext), label, max_requests, expires, now),
                )
        return plaintext

    def check(self, key: str, increment: bool = True) -> Tuple[bool, str]:
        """校验 key 并 (可选) 计数。返回 (ok, reason)。"""
        if not key or not key.startswith(KEY_PREFIX):
            return False, REASON_INVALID
        h = hash_key(key)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM api_keys WHERE key_hash = ?", (h,)
                ).fetchone()
                if row is None:
                    return False, REASON_INVALID
                if row["revoked"]:
                    return False, REASON_REVOKED
                if row["expires_at"] is not None and row["expires_at"] < time.time():
                    return False, REASON_EXPIRED
                if (row["max_requests"] is not None
                        and row["requests_used"] >= row["max_requests"]):
                    return False, REASON_QUOTA
                if increment:
                    conn.execute(
                        "UPDATE api_keys SET requests_used = requests_used + 1, "
                        "last_used_at = ? WHERE key_hash = ?",
                        (time.time(), h),
                    )
        return True, REASON_OK

    def list_keys(self):
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT label, key_hash, max_requests, requests_used, "
                    "expires_at, created_at, last_used_at, revoked "
                    "FROM api_keys ORDER BY created_at DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def revoke(self, label: str) -> int:
        """按 label 吊销, 返回影响行数。"""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE api_keys SET revoked = 1 WHERE label = ? AND revoked = 0",
                    (label,),
                )
        return cur.rowcount


class AnonymousRateLimiter:
    """匿名请求的滑动窗口限流 (按 IP)。进程内计数, 重启清零——够用。"""

    def __init__(self, per_minute: int = 20):
        self.per_minute = max(1, per_minute)
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            dq = self._hits[ip]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.per_minute:
                return False
            dq.append(now)
        return True
