import json
import threading
import time
import urllib.request

from mt_lnn.awareliquid_daemon import serve


def _start(tmp_path, port):
    server = serve(tmp_path, "127.0.0.1", port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)
    return server


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
        return json.loads(r.read())


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def test_daemon_health_and_evidence_roundtrip(tmp_path):
    port = 8771
    server = _start(tmp_path, port)
    try:
        assert _get(port, "/health") == {"ok": True}
        assert _get(port, "/sessions") == {"sessions": []}

        r1 = _post(port, "/sessions/abc/evidence",
                   {"source": "mock", "query": "what is m-theory", "fact_len": 12})
        assert r1["n_evidence"] == 1

        r2 = _post(port, "/sessions/abc/question", {"text": "follow up"})
        assert r2["n_open"] == 1

        full = _get(port, "/sessions/abc")
        assert full["session_id"] == "abc"
        assert full["evidence_log"][0]["query"] == "what is m-theory"

        clusters = _get(port, "/meta/clusters?k=1")
        assert clusters["n_rows"] == 1
    finally:
        server.shutdown()


def test_daemon_404_for_unknown_session(tmp_path):
    port = 8772
    server = _start(tmp_path, port)
    try:
        try:
            _get(port, "/sessions/missing")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
