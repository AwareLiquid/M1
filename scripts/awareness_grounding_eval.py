"""awareness_grounding_eval.py — 验证 awareness.market 知识库对小模型准确率的提升

命题: 小模型（1.7B）回答物理/科学问题 =
  - 无检索: 靠参数内记忆 (幻觉/错答率高)
  - +awareness 检索: 有来源摘要 grounding (准确率应显著提升)

用法:
  py -3.11 scripts/awareness_grounding_eval.py --questions 10
"""
import argparse
import json
import sys
import time
import urllib.request

import torch

sys.path.insert(0, r"E:\M1")

API = "https://awareness.market/api/v1/public/retrieve"


def retrieve(query: str, top_k: int = 5):
    req = urllib.request.Request(
        API, data=json.dumps({"query": query, "top_k": top_k}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return d.get("results", [])


def build_context(results):
    """检索结果 → 带引用的上下文。"""
    parts = []
    for i, r in enumerate(results, 1):
        snippet = (r.get("snippet") or "")[:1500]
        parts.append(f"[{i}] {r['title']}\n{snippet}\n"
                     f"来源: {r['source_ref']} (被引 {r.get('citation_count', 0)})")
    return "\n\n".join(parts)


PROMPT = ("You are answering facts about physics and science. "
          "Answer ONLY based on the provided sources.\n\n"
          "Sources:\n{ctx}\n\n"
          "Question: {q}\n"
          "Answer with the exact answer text only — a few words, no explanation. "
          "If the sources do NOT contain the answer, answer exactly "
          "'NOT_FOUND'.\nAnswer:")


def generate(model, tok, prompt, device):
    ids = tok.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, do_sample=False,
                             eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    # 物理/科学问题 + 标准短答案 (从公开题库常识取, 用于核对)
    QA = [
        ("What is the value of the fine-structure constant approximately?",
         "1/137"),
        ("What particle mediates the electromagnetic force?",
         "photon"),
        ("What is the critical temperature of BCS superconductors?",
         "39 K"),
        ("What is the Fermi energy?",
         "highest occupied energy level at absolute zero"),
        ("What does the uncertainty principle relate?",
         "position and momentum"),
        ("What is a quasiparticle in a crystal?",
         "phonon"),
        ("What is the Hall effect?",
         "voltage across a conductor"),
        ("What is the Meissner effect?",
         "expulsion of magnetic field"),
        ("What is the band gap?",
         "energy gap between valence and conduction band"),
        ("What is entanglement entropy?", "measure of quantum entanglement"),
    ][: args.questions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen3-1.7B"
    print(f"load {name} on {device} (local cache) ...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True,
                                                 local_files_only=True).to(device).eval()

    hit_base = hit_rag = nf_base = nf_rag = 0
    rows = []
    for q, gold in QA:
        # 无检索
        t0 = time.time()
        ans_base = generate(model, tok, PROMPT.format(ctx="(none)", q=q), device)
        # +检索
        results = retrieve(q, args.top_k)
        ctx = build_context(results)
        ans_rag = generate(model, tok, PROMPT.format(ctx=ctx, q=q), device)
        ok_base = gold.lower() in ans_base.lower() or ans_base.lower() in gold.lower()
        ok_rag = gold.lower() in ans_rag.lower() or ans_rag.lower() in gold.lower()
        hit_base += int(ok_base)
        hit_rag += int(ok_rag)
        nf_base += int("not_found" in ans_base.lower())
        nf_rag += int("not_found" in ans_rag.lower())
        rows.append({"q": q, "gold": gold, "base": ans_base, "rag": ans_rag,
                     "n_sources": len(results), "ok_base": ok_base,
                     "ok_rag": ok_rag})
        print(f"[{q[:40]}...]")
        print(f"  无检索: {ans_base[:60]!r} {'✓' if ok_base else '✗'}")
        print(f"  +awareness: {ans_rag[:60]!r} {'✓' if ok_rag else '✗'} "
              f"({len(results)} 来源, {time.time()-t0:.1f}s)")

    n = len(QA)
    print("\n═══════ 命题验证 ═══════")
    print(f"无检索     命中 {hit_base}/{n} ({hit_base/n:.0%})  拒答 {nf_base}")
    print(f"+awareness 命中 {hit_rag}/{n} ({hit_rag/n:.0%})  拒答 {nf_rag}")
    print(f"提升: {hit_rag-hit_base} 题")
    json.dump(rows, open("awareness_grounding_results.json", "w"),
              ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
