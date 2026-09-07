"""awareness.py — awareness.market 知识库客户端（KaaS：小模型的云端大脑）

REST 检索：POST /api/v1/public/retrieve
- 匿名预览：标题 + 摘要 + DOI + 被引数 + 置信度（免费）
- API key：全文
- 1027+ 条知识，物理/科学领域 844 条（凝聚态/量子信息/材料/ML）

返回结构（预览）：
  {"results": [{"id", "title", "confidence", "source_type", "source_ref",
                "citation_count", "domain", "topic", "snippet"}], "count", "usage"}
"""
from __future__ import annotations

import json
import urllib.request
from typing import Dict, List, Optional

API_URL = "https://awareness.market/api/v1/public/retrieve"
_DEFAULT_TOP_K = 5


def retrieve(query: str, top_k: int = _DEFAULT_TOP_K,
             api_key: Optional[str] = None,
             domain: Optional[str] = None) -> List[Dict]:
    """检索知识库，返回归一化的结果列表。网络失败返回 []（优雅降级）。

    domain: 限定领域（如 "science" / "business"）。KaaS 的 science 领域是
    物理/材料/ML/量子信息等 844 条论文，business 是供应链/财务/交易帖 183 条。
    不传 domain 时中文 query 极易命中 business/deals（交易帖），把物理问题
    检索成"液冷数据中心/A100 显卡现货"——这是"回答很差"的直接根因。
    """
    body: dict = {"query": query, "top_k": top_k}
    if domain:
        body["domain"] = domain
    data_bytes = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(API_URL, data=data_bytes, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:  # noqa: BLE001 - 检索失败不应中断对话
        print(f"[awareness] retrieve failed: {e}")
        return []
    return data.get("results", [])


def format_context(results: List[Dict], max_chars: int = 3000) -> str:
    """检索结果 → 带来源引用的上下文。摘要截断保护。"""
    parts: List[str] = []
    used = 0
    for i, r in enumerate(results, 1):
        snippet = (r.get("snippet") or "").strip()
        if len(snippet) > 5:
            snippet = ("".join(snippet.split("\n")[1:])  # 摘要通常首行是标题重复
                       if "\n" in snippet else snippet)
        take = snippet[:min(len(snippet), max_chars - used)]
        if not take:
            continue
        cite = r.get("source_ref", "") or ""
        title = r.get("title", "") or ""
        parts.append(f"[S{i}] {title} :: {take}\n   (来源: {cite})")
        used += len(take) + 60
        if used >= max_chars:
            break
    return "\n".join(parts)


def has_evidence(result_ctx: str) -> bool:
    return bool(result_ctx and result_ctx.strip())


GROUNDED_PROMPT = (
    "You are a knowledgeable assistant. Answer using ONLY the provided sources.\n\n"
    "Sources:\n{ctx}\n\n"
    "Question: {q}\n\n"
    "Rules:\n"
    "1. Answer based only on the sources — cite the source ID like [S1] when you use it.\n"
    "2. If the sources do not contain the answer, answer exactly: "
    "I don't have reliable sources for this — I won't guess.\n"
    "3. Keep the answer concise, 2-4 sentences.\n"
    "Answer:")
