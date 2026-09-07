"""consolidation_policy — WHAT to write / WHAT is stale, as pure functions.

Two deterministic, IO-free decisions that the sleep-cycle consolidation path
and the SDK-C event hooks share:

* :func:`policy_write` — map one incoming record to the ``(key, value)``
  bindings it should sediment into long-term memory (an explicit key when the
  record carries one, content-addressable otherwise, with an optional
  salience gate);
* :func:`policy_forget` — map one ALREADY-DETECTED conflict event to the old
  key(s) whose bindings are stale, for the caller to forget/supersede.

HONEST BOUNDARY — conflict RESOLUTION does not live here (kb/H003 D3):
the four-competency bench judged D3 (conflict resolution through the write
rule) NEGATIVE for every config at v0 budget (RESULTS.md; kb/H003 verdict
log) — no sum/delta blending rule resurrected it, and per preregistration the
negative stands. This module therefore NEVER decides which side of a conflict
is true, never merges or blends values, and never rewrites memory itself: it
only names candidate keys. Detecting conflicts is the caller's job (the SDK
detection layer is the designated owner); acting on the returned keys
(forget / supersede / ignore) is the caller's decision.

Purity contract: both functions are deterministic, total over their input
domain (missing fields are handled, never raised on), perform ZERO IO, never
mutate their inputs, and import nothing heavier than typing — so the SDK-C
event hooks can call them from any thread or process without importing torch
or touching a store.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

MemoryKey = Any   # str / int here; the sleep path may pass embedding tensors

__all__ = ["policy_write", "policy_forget"]


def _content_of(record: Mapping) -> Any:
    """The record's payload: explicit 'value', else 'content', else None."""
    value = record.get("value")
    if value is None:
        value = record.get("content")
    return value


def _salience_of(record: Mapping) -> Optional[float]:
    sal = record.get("salience")
    if sal is None:
        return None
    try:
        return float(sal)
    except (TypeError, ValueError):
        return None   # non-numeric salience = ungated, never a crash


def policy_write(
    record: Mapping,
    *,
    min_salience: Optional[float] = None,
) -> List[Tuple[MemoryKey, Any]]:
    """Bindings to sediment for one record: ``[(key, value), ...]``.

    Rules, in order (first match wins):

    1. **Salience gate** — with ``min_salience`` set, a record whose numeric
       ``salience`` is below it yields ``[]`` (nothing consolidates; the
       brain consolidates a subset of the night's replay). Records without a
       numeric salience are ungated.
    2. **Explicit keys** — ``record["keys"]`` (a list) or the single
       ``record["key"]``: one binding per key, all carrying the same payload.
       This is the rule the sleep cycle rides: ``nrem_replay(write_policy=...)``
       pre-fills ``"key"``/``"content"`` from its key/content fields, so the
       derived bindings land on the same keys the default path would use.
    3. **Content-addressable fallback** — no key anywhere: the content keys
       itself, ``[(content, content)]``, matching how the knowledge/graph
       stores treat free text (embed the text, store the text).

    The empty record yields ``[]``. Keys/values are returned as-is — no
    normalisation, no encoding (encoding is the store/broker's job).
    """
    if min_salience is not None:
        sal = _salience_of(record)
        if sal is not None and sal < min_salience:
            return []

    value = _content_of(record)
    if value is None:
        return []

    keys = record.get("keys")
    if keys is None:
        key = record.get("key")
        keys = [] if key is None else [key]
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
        keys = [keys]   # a bare non-list keys field = one key
    keys = list(keys)

    if not keys:
        # Content-addressable: text keys itself (free-text record, no key).
        return [(value, value)]
    return [(k, value) for k in keys]


def policy_forget(conflict_event: Mapping) -> List[MemoryKey]:
    """Old keys made stale by an ALREADY-DETECTED conflict: ``[old_key, ...]``.

    Input (all fields optional — the function is total):

      ``old`` / ``new`` : mappings with ``key`` and ``value``/``content``
      ``type``          : ``"update"`` | ``"contradiction"`` | ``"duplicate"``
                          (unknown types are treated as updates)

    Rules:

    * no old key, or old == new content (a re-statement, not a conflict),
      or an explicit ``"duplicate"`` → ``[]``: nothing is stale;
    * otherwise → ``[old["key"]]``: the incoming binding takes the new
      record's own key; the old binding is what a caller may forget or
      supersede (graph lifecycle) — the CALL of this policy does not itself
      touch any store.

    See the module docstring for why resolution/judgment is out of scope
    (kb/H003 D3 negative; detection is the SDK layer's job).
    """
    old = conflict_event.get("old")
    old = old if isinstance(old, Mapping) else {}
    new = conflict_event.get("new")
    new = new if isinstance(new, Mapping) else {}

    old_key = old.get("key")
    if old_key is None:
        return []
    if conflict_event.get("type") == "duplicate":
        return []

    old_content = _content_of(old)
    new_content = _content_of(new)
    if old_content is not None and old_content == new_content:
        return []   # same statement twice — no stale binding

    return [old_key]
