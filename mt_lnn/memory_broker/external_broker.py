"""ExternalBroker — thin HTTP adapter over the Awareness-SDK local daemon.

THE ONLY sanctioned cross-repo runtime channel: this repo never imports
Awareness-SDK source; the SDK's local daemon (Awareness-SDK ``local/``,
bound to ``127.0.0.1:37800``) is reached over plain HTTP from a stdlib
client — zero new dependencies, and the SDK→M1 dependency direction stays
one-way (the daemon calls into M1's installed interface; M1 only speaks its
published HTTP).

Endpoint mapping ( Awareness-SDK local daemon, ``/api/v1`` REST surface):

* ``write``   → ``POST /memories``  ``{content, session_id, tags, metadata}``
* ``recall``  → ``GET  /memories/search?q=...&limit=...`` (client-side filter
  to the requested session — the daemon's search is workspace-scoped)
* ``state_bytes`` → ``GET /memories`` paged, summing content bytes of the
  session's rows
* ``forget`` / ``snapshot`` / ``restore`` → **no daemon endpoint exists**:
  the REST surface has no memory-row deletion (only a knowledge-card regex
  cleanup) and no export/import of the store. Raising NotImplementedError
  with that reason IS the honest adapter; see docs/MEMORY_BROKER.md.

Key/value flattening: the daemon stores free-text records, not bindings, so
``value`` is stringified into ``content`` and ``key`` rides in ``tags``
(``m1:key=...``) plus ``metadata.m1_key`` — enough for a caller to re-find
what it wrote, not a key/value substrate. Dedup: the daemon dedups identical
content per source and answers ``{status: "duplicate", id}``; this adapter
returns that id like any other write.

Project scoping: pass ``project_dir`` to send ``X-Awareness-Project-Dir`` on
every call (the daemon answers 409 ``project_mismatch`` if it is bound
elsewhere — surfaced as :class:`BrokerError`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, List, Optional

from .base import BrokerError, MemoryBroker, MemoryKey, RecallHit, Snapshot

DEFAULT_BASE_URL = "http://127.0.0.1:37800"
_PAGE = 100      # page size for the state_bytes listing walk
_MAX_PAGES = 100  # hard stop: 10k rows walked per state_bytes call


class ExternalBroker(MemoryBroker):
    """MemoryBroker face over the Awareness local daemon's REST surface."""

    kind = "external"

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 project_dir: Optional[str] = None,
                 timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.project_dir = project_dir
        self.timeout = timeout

    # ------------------------------------------------------------- transport

    def _request(self, method: str, path: str, *, params=None,
                 body: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.project_dir:
            req.add_header("X-Awareness-Project-Dir", self.project_dir)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise BrokerError(
                f"daemon {method} {path} -> HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BrokerError(
                f"daemon unreachable at {self.base_url}: {exc}") from exc

    # ---------------------------------------------------------------- broker

    def write(self, session_id: str, key: MemoryKey, value, *,
              meta: Optional[Any] = None) -> str:
        content = value if isinstance(value, str) else str(value)
        metadata: dict = {"m1_key": key if isinstance(key, str) else str(key)}
        if isinstance(meta, dict):
            metadata.update(meta)
        elif meta is not None:
            metadata["m1_meta"] = meta
        out = self._request("POST", "/api/v1/memories", body={
            "content": content,
            "session_id": session_id,
            "tags": [f"m1:key={key}"],
            "metadata": metadata,
        })
        record_id = out.get("id")
        if not record_id:
            raise BrokerError(f"daemon write returned no id: {out}")
        return str(record_id)

    def recall(self, session_id: str, query: MemoryQuery,
               top_k: int = 5) -> List[RecallHit]:
        out = self._request("GET", "/api/v1/memories/search",
                            params={"q": str(query), "limit": int(top_k)})
        hits: List[RecallHit] = []
        for item in out.get("items", []):
            if item.get("session_id") != session_id:
                continue   # daemon search is workspace-scoped; honour sessions
            hits.append(RecallHit(
                value=item.get("content") or item.get("fts_content") or "",
                score=None,   # the cascade returns ranked items, no score
                id=item.get("id"),
                meta=item.get("metadata"),
            ))
            if len(hits) >= top_k:
                break
        return hits

    def forget(self, session_id: str, key: Optional[MemoryKey] = None) -> bool:
        raise NotImplementedError(
            "the external daemon exposes no memory deletion: its REST surface "
            "has no DELETE /memories route (only a knowledge-card regex "
            "cleanup, which is a different store) — selective deletion is the "
            "parametric backend's differentiator (docs/MEMORY_BROKER.md)")

    def snapshot(self, session_id: str) -> Snapshot:
        raise NotImplementedError(
            "外部存储无 bit-exact 迁移: the daemon has no export endpoint; its "
            "state lives in its own SQLite index + markdown files and is only "
            "as portable as its own tooling. A cross-process bit-exact "
            "snapshot is the parametric backend's property.")

    def restore(self, session_id: str, snap: Snapshot) -> None:
        raise NotImplementedError(
            "外部存储无 bit-exact 迁移: the daemon has no import endpoint; see "
            "snapshot().")

    def state_bytes(self, session_id: str) -> int:
        """Sum of the session's stored content bytes (UTF-8), walked over the
        paged list endpoint. Storage-usage honesty for the O(n) backend."""
        total = 0
        for page in range(_MAX_PAGES):
            out = self._request("GET", "/api/v1/memories",
                                params={"limit": _PAGE, "offset": page * _PAGE})
            items = out.get("items", [])
            if not items:
                break
            total += sum(len((it.get("content") or "").encode("utf-8"))
                         for it in items if it.get("session_id") == session_id)
            if len(items) < _PAGE:
                break
        return total

    def health(self) -> dict:
        """Daemon ``/healthz`` passthrough (unauthenticated, project-free)."""
        return self._request("GET", "/healthz")
