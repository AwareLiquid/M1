"""test_api_auth.py — mt_lnn/api_auth 的纯逻辑单测（不依赖 torch）。"""
import time

from mt_lnn.api_auth import (KEY_PREFIX, REASON_EXPIRED, REASON_INVALID,
                             REASON_QUOTA, REASON_REVOKED, AnonymousRateLimiter,
                             ApiKeyStore, hash_key)


def test_issue_and_check_ok(tmp_path):
    store = ApiKeyStore(str(tmp_path / "keys.db"))
    key = store.issue("acme", max_requests=10)
    assert key.startswith(KEY_PREFIX)
    assert len(key) == len(KEY_PREFIX) + 32
    ok, reason = store.check(key)
    assert ok and reason == "ok"
    # 计数生效
    assert store.list_keys()[0]["requests_used"] == 1


def test_wrong_key_rejected(tmp_path):
    store = ApiKeyStore(str(tmp_path / "keys.db"))
    store.issue("acme")
    ok, reason = store.check("al_" + "0" * 32)
    assert not ok and reason == REASON_INVALID
    ok, reason = store.check("nonsense")
    assert not ok and reason == REASON_INVALID


def test_quota_exhausted(tmp_path):
    store = ApiKeyStore(str(tmp_path / "keys.db"))
    key = store.issue("acme", max_requests=3)
    for _ in range(3):
        ok, _ = store.check(key)
        assert ok
    ok, reason = store.check(key)
    assert not ok and reason == REASON_QUOTA


def test_expiry(tmp_path):
    store = ApiKeyStore(str(tmp_path / "keys.db"))
    key = store.issue("acme", ttl_days=1)
    # 直接改库模拟过期
    import sqlite3
    conn = sqlite3.connect(store.db_path)
    conn.execute("UPDATE api_keys SET expires_at = ? WHERE label = 'acme'",
                 (time.time() - 10,))
    conn.commit()
    conn.close()
    ok, reason = store.check(key)
    assert not ok and reason == REASON_EXPIRED


def test_revoke(tmp_path):
    store = ApiKeyStore(str(tmp_path / "keys.db"))
    key = store.issue("acme")
    assert store.revoke("acme") == 1
    ok, reason = store.check(key)
    assert not ok and reason == REASON_REVOKED
    assert store.revoke("acme") == 0  # 幂等


def test_hash_deterministic():
    assert hash_key("al_abc") == hash_key("al_abc")
    assert hash_key("al_abc") != hash_key("al_abd")


def test_rate_limiter_window():
    rl = AnonymousRateLimiter(per_minute=3)
    assert all(rl.allow("1.2.3.4") for _ in range(3))
    assert not rl.allow("1.2.3.4")          # 第 4 次拒绝
    assert rl.allow("5.6.7.8")              # 其他 IP 不受影响


def test_rate_limiter_recovers_after_window(monkeypatch):
    rl = AnonymousRateLimiter(per_minute=2)
    assert rl.allow("1.2.3.4") and rl.allow("1.2.3.4")
    assert not rl.allow("1.2.3.4")
    # 时间前进 61s 后窗口滑动恢复
    future = time.time() + 61.0
    monkeypatch.setattr(time, "time", lambda: future)
    assert rl.allow("1.2.3.4")


# ---------------------------------------------------------------------------
# 生产化 (P2-10): count_active + strict 模式启动体检
# ---------------------------------------------------------------------------

def test_count_active_filters_revoked_expired_exhausted(tmp_path):
    store = ApiKeyStore(str(tmp_path / "keys.db"))
    assert store.count_active() == 0

    store.issue("healthy")                       # 可用
    store.issue("revoked"); store.revoke("revoked")
    store.issue("expired", ttl_days=-1.0)        # 已过期
    capped = store.issue("capped", max_requests=1)
    store.check(capped)                          # 用掉唯一一次配额
    assert store.count_active() == 1
