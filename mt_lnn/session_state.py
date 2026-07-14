"""HFSessionState — Capsule v2 schema for HuggingFace backbones.

MT-LNN's :mod:`mt_lnn.capsule` persists ``ModelCacheStruct.layers`` (the
recurrent h_states). HF causal LMs don't expose that, so we carry only
the *thinking context*: open_questions, evidence_log, conversation
history. Same field names as Capsule v2 so callers don't fork.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class HFSessionState:
    session_id: str
    open_questions: List[str] = field(default_factory=list)
    evidence_log: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)


def save_session(session: HFSessionState, path: str) -> None:
    """Atomically persist *session* as JSON.

    Writes to a temp file in the SAME directory, then ``os.replace``s it over
    the target — atomic on POSIX and on Windows (os.replace overwrites an
    existing destination atomically). A plain ``Path.write_text`` truncates
    first, so a concurrent reader (or a crash mid-write) could observe a
    torn/empty file; the daemon serves these files from multiple threads.
    """
    target = Path(path)
    payload = json.dumps(asdict(session), ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_session(path: str) -> HFSessionState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return HFSessionState(
        session_id=data["session_id"],
        open_questions=list(data.get("open_questions", [])),
        evidence_log=list(data.get("evidence_log", [])),
        history=list(data.get("history", [])),
    )
