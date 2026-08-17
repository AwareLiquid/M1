"""
mt_lnn/graph_linking.py -- reference extraction + matching for graph memory.

A faithful Python port of the Awareness-SDK memory graph's primary edge-building
mechanism (``local/src/core/link-discovery.mjs`` in everest-an/Awareness-SDK):
scan a text node for code-identifier *references* (backtick names, file paths,
PascalCase identifiers) and match them against known node titles to create
weighted, typed ``reference`` edges. Zero LLM -- all matching is regex + exact /
fuzzy string comparison -- and zero model coupling (pure stdlib).

This complements the semantic-cosine linking already in graph_memory.py: cosine
linking connects nodes that are *embedding-similar*, reference linking connects a
note to the entities it *explicitly mentions*. Awareness uses both; together they
populate the graph from content, not just from vector geometry.

Confidence weights mirror Awareness (used directly as the edge weight):
    exact file path  -> 1.0
    backtick ident   -> 0.8
    PascalCase ident -> 0.7
    path basename    -> 0.6
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

__all__ = ["extract_references", "match_references"]


# Common words that look like PascalCase but are not code references.
IGNORE_PASCAL = {
    "README", "TODO", "FIXME", "NOTE", "WARNING", "HACK", "XXX",
    "API", "URL", "HTTP", "HTTPS", "HTML", "CSS", "JSON", "XML",
    "SQL", "CLI", "SDK", "CDN", "DNS", "SSH", "TLS", "SSL",
    "EOF", "NULL", "TRUE", "FALSE", "OK", "PR", "CI", "CD",
    "UI", "UX", "ID", "IP", "OS", "DB", "AWS", "GCP", "PDF",
    "DOCX", "XLSX", "CSV", "UTF", "ASCII", "NPM", "YAML",
    "Docker", "GitHub", "GitLab", "MongoDB", "PostgreSQL", "Redis",
    "Python", "JavaScript", "TypeScript", "Markdown", "Rust", "Golang",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Phase", "Table", "Section", "Chapter", "Figure", "Summary",
    "Added", "Changed", "Fixed", "Removed", "Updated", "Created",
}

# Backtick code: `identifier` or `obj.method()`.
_RE_BACKTICK = re.compile(r"`([^`]+?)`")
# File paths: foo/bar.ext or ./foo/bar.ext (must have an extension).
_RE_FILE_PATH = re.compile(r"(?:\.?\.?/?)?(?:[\w@-]+/)+[\w.\-]+\.\w{1,10}")
# PascalCase identifiers: at least two words (e.g. WorkspaceScanner).
_RE_PASCAL = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
# A bare identifier (for pulling a name out of `obj.method(arg)`).
_RE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _extract_identifiers(raw: str) -> List[str]:
    """Pull a meaningful identifier out of backtick content.

    ``obj.method(arg)`` -> ``method``; ``Foo`` -> ``Foo``; non-identifiers (pure
    punctuation, shell snippets) -> none.
    """
    core = raw.split("(", 1)[0].strip()
    if not core:
        return []
    seg = core.split(".")[-1].strip()
    return [seg] if _RE_IDENT.match(seg) else []


def extract_references(text: str) -> List[Dict[str, Any]]:
    """Extract potential code references from a text node.

    Returns a list of ``{"name": str, "type": "backtick"|"path"|"pascal",
    "line": int}``. Skips fenced code blocks; de-duplicates by name across the
    whole text (first occurrence wins, matching the JS source).
    """
    if not text:
        return []

    seen = set()
    refs: List[Dict[str, Any]] = []
    for i, line in enumerate(text.split("\n")):
        line_num = i + 1
        if line.lstrip().startswith("```"):
            continue

        # Backtick references.
        for m in _RE_BACKTICK.finditer(line):
            raw = m.group(1).strip()
            for name in _extract_identifiers(raw):
                if name in seen:
                    continue
                seen.add(name)
                if "/" in raw and "." in raw:
                    refs.append({"name": raw, "type": "path", "line": line_num})
                else:
                    refs.append({"name": name, "type": "backtick", "line": line_num})

        # File paths (outside backticks).
        for m in _RE_FILE_PATH.finditer(line):
            p = m.group(0)
            if p in seen:
                continue
            seen.add(p)
            refs.append({"name": p, "type": "path", "line": line_num})

        # PascalCase names.
        for m in _RE_PASCAL.finditer(line):
            name = m.group(1)
            if name in seen or name in IGNORE_PASCAL:
                continue
            seen.add(name)
            refs.append({"name": name, "type": "pascal", "line": line_num})

    return refs


def match_references(
    refs: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Match extracted references against known nodes by title.

    ``nodes`` is a list of ``{"id": Any, "title": str}`` (an optional
    ``"node_type"`` is accepted and ignored here). Returns a list of
    ``{"target_id", "ref_name", "ref_type", "confidence", "line"}`` links,
    de-duplicated per (ref_name, target_id). Confidence weights mirror
    Awareness-SDK.
    """
    if not refs or not nodes:
        return []

    by_title: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        title = node.get("title")
        if title is None:
            continue
        by_title.setdefault(title, []).append(node)

    links: List[Dict[str, Any]] = []
    seen = set()
    for ref in refs:
        name = ref["name"]
        rtype = ref["type"]
        matches: List[tuple] = []

        if rtype == "path":
            normalized = name[2:] if name.startswith("./") else name
            # Exact full-path title match (strongest), else basename.
            if normalized in by_title:
                for node in by_title[normalized]:
                    matches.append((node, 1.0))
            else:
                basename = name.split("/")[-1]
                for node in by_title.get(basename, []):
                    matches.append((node, 0.6))
        else:
            confidence = 0.8 if rtype == "backtick" else 0.7
            for node in by_title.get(name, []):
                matches.append((node, confidence))

        for node, confidence in matches:
            key = (name, node["id"])
            if key in seen:
                continue
            seen.add(key)
            links.append({
                "target_id": node["id"],
                "ref_name": name,
                "ref_type": rtype,
                "confidence": confidence,
                "line": ref["line"],
            })

    return links
