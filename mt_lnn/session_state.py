"""HFSessionState — Capsule v2 schema for HuggingFace backbones.

MT-LNN's :mod:`mt_lnn.capsule` persists ``ModelCacheStruct.layers`` (the
recurrent h_states). HF causal LMs don't expose that, so we carry only
the *thinking context*: open_questions, evidence_log, conversation
history. Same field names as Capsule v2 so callers don't fork.
"""

from __future__ import annotations

import json
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
    Path(path).write_text(
        json.dumps(asdict(session), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_session(path: str) -> HFSessionState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return HFSessionState(
        session_id=data["session_id"],
        open_questions=list(data.get("open_questions", [])),
        evidence_log=list(data.get("evidence_log", [])),
        history=list(data.get("history", [])),
    )
