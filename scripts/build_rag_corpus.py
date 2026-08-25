"""build_rag_corpus.py — 构建并保存检索索引（roadmap Phase 2 基础设施）

用法:
  # 从本地文本文件建索引（每行一段或一个文档, 自动切块）
  py -3.11 scripts/build_rag_corpus.py --files corpus/*.txt --out data/rag_index.pkl

  # 从 HF 数据集下载（复用 datasets-server API, 规避 datasets 5.0 的
  # canonical 数据集 URI bug —— 见 P1 本地验证记录）
  py -3.11 scripts/build_rag_corpus.py --hf wikitext --limit-docs 50000 --out data/rag_index.pkl

保存格式: pickle (BM25Index 对象), 可被 serve 栈直接加载。
"""
import argparse
import json
import pickle
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.rag import BM25Index, build_index_from_texts  # noqa: E402

HF_TEMPLATES = {
    # datasets-server rows API; wikitext 用 parquet 镜像 train split
    "wikitext": ("https://datasets-server.huggingface.co/rows"
                 "?dataset=Salesforce/wikitext&config=wikitext-103-raw-v1"
                 "&split=train&offset={off}&length=100"),
}


def fetch_hf_texts(name: str, limit_docs: int):
    tpl = HF_TEMPLATES[name]
    texts, off = [], 0
    while len(texts) < limit_docs:
        url = tpl.format(off=off)
        try:
            data = json.loads(urllib.request.urlopen(url, timeout=120).read())
        except Exception as e:  # noqa: BLE001
            print(f"fetch 失败 (offset {off}): {e}")
            break
        rows = data.get("rows") or []
        if not rows:
            break
        for r in rows:
            text = r["row"].get("text", "")
            if isinstance(text, str) and len(text.strip()) > 100:
                texts.append(text.strip())
                if len(texts) >= limit_docs:
                    break
        off += 100
        print(f"  fetched {len(texts)}/{limit_docs} docs ...")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", help="本地文本文件")
    ap.add_argument("--hf", choices=list(HF_TEMPLATES), help="HF 数据集名")
    ap.add_argument("--limit-docs", type=int, default=50000)
    ap.add_argument("--chunk-words", type=int, default=100)
    ap.add_argument("--out", required=True, help="输出 pickle 路径")
    args = ap.parse_args()

    texts: list[str] = []
    if args.hf:
        print(f"从 HF 下载 {args.hf} (limit {args.limit_docs}) ...")
        texts = fetch_hf_texts(args.hf, args.limit_docs)
    elif args.files:
        for f in args.files:
            p = Path(f)
            raw = p.read_text(encoding="utf-8-sig", errors="ignore")
            texts.append(raw)
            print(f"read {f}: {len(raw)} chars")
    if not texts:
        print("无输入 (--hf 或 --files 至少一个)")
        return 1

    print(f"建索引 ({len(texts)} docs, chunk={args.chunk_words}) ...")
    idx = build_index_from_texts(texts, chunk_words=args.chunk_words)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(idx, fh)
    print(f"完成: {out} ({len(idx)} chunks, {out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
