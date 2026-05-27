"""Phase 8 — AwareLiquid capsule daemon (stdlib only).

A tiny HTTP daemon that persists capsules across processes and surfaces
meta-learned patterns. Deliberately stdlib-only (``http.server``,
``json``) so the daemon adds zero runtime deps on top of the existing
mt_lnn package.

Endpoints
---------
GET  /health                      -> {"ok": true}
GET  /sessions                    -> list of known session_ids
GET  /sessions/<id>               -> full HFSessionState as JSON
POST /sessions/<id>/evidence      -> append one evidence row (json body)
POST /sessions/<id>/question      -> append one open question (json body: {"text": "..."})
GET  /meta/clusters?k=4           -> cluster evidence across ALL sessions

Run
---
    python -m mt_lnn.awareliquid_daemon --root capsules/ --port 8765

The LM-side demo (``demo_awareliquid_v2.py``) still owns
session.json files directly; the daemon is an *aggregation* layer that
reads the same directory. No locking — single-writer assumption.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from mt_lnn.meta_learning import cluster_evidence
from mt_lnn.session_state import HFSessionState, load_session, save_session


def _list_sessions(root: Path):
    return sorted(p.stem for p in root.glob("*.json"))


def _load_or_init(root: Path, sid: str) -> HFSessionState:
    p = root / f"{sid}.json"
    if p.exists():
        return load_session(str(p))
    return HFSessionState(session_id=sid)


def _all_evidence(root: Path):
    rows = []
    for sid in _list_sessions(root):
        s = load_session(str(root / f"{sid}.json"))
        rows.extend(s.evidence_log)
    return rows


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet default access log
            return

        def _send(self, code: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0") or "0")
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            if parts == ["health"]:
                return self._send(200, {"ok": True})
            if parts == ["sessions"]:
                return self._send(200, {"sessions": _list_sessions(root)})
            if len(parts) == 2 and parts[0] == "sessions":
                sid = parts[1]
                p = root / f"{sid}.json"
                if not p.exists():
                    return self._send(404, {"error": "no such session"})
                return self._send(200, asdict(load_session(str(p))))
            if parts == ["meta", "clusters"]:
                q = parse_qs(u.query)
                k = int(q.get("k", ["4"])[0])
                return self._send(200, cluster_evidence(_all_evidence(root), k=k))
            return self._send(404, {"error": "unknown route"})

        def do_POST(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            body = self._read_body()
            if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "evidence":
                sid = parts[1]
                s = _load_or_init(root, sid)
                s.evidence_log.append(body)
                save_session(s, str(root / f"{sid}.json"))
                return self._send(200, {"ok": True, "n_evidence": len(s.evidence_log)})
            if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "question":
                sid = parts[1]
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._send(400, {"error": "missing text"})
                s = _load_or_init(root, sid)
                s.open_questions.append(text)
                save_session(s, str(root / f"{sid}.json"))
                return self._send(200, {"ok": True, "n_open": len(s.open_questions)})
            return self._send(404, {"error": "unknown route"})

    return Handler


def serve(root: Path, host: str, port: int) -> ThreadingHTTPServer:
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), make_handler(root))
    return server


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="capsules")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    root = Path(args.root)
    server = serve(root, args.host, args.port)
    print(f"[awareliquid-daemon] {args.host}:{args.port} root={root.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
