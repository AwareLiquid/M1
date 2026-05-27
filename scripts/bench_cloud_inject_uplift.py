"""bench_cloud_inject_uplift.py — quantify the gain from cloud inject.

For each question in ``benchmarks/cloud_inject_questions.json``, generate
two answers:

  (A) no-inject  : pass the bare question to the backbone
  (B) inject     : prepend ``[Absorbed fact] {fact}\\nContinuing: ``
                   (the same template ``demo_awareliquid_v2.py`` uses)

Score = normalized substring match of ``answer`` in the generated text.
Report uplift = inject_acc - no_inject_acc.

Backends
--------
- ``hf``    : real HuggingFace model (+ optional MT adapter checkpoint)
- ``echo``  : deterministic stub used in CI tests / smoke runs.
              Returns "I don't know" if no fact is in the prompt,
              otherwise echoes the fact verbatim. Lets us prove the
              scoring + scaffolding works without a 4 GB model download.

Usage
-----
    # smoke (10 s, no model needed)
    python scripts/bench_cloud_inject_uplift.py --backend echo

    # real (TinyLlama + adapter)
    python scripts/bench_cloud_inject_uplift.py --backend hf \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List


# ----------------------------------------------------------------- scoring

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    return _PUNCT.sub(" ", s.lower()).strip()


def hit(reference: str, generated: str) -> bool:
    return _norm(reference) in _norm(generated)


# ----------------------------------------------------------------- prompts

INJECT_TEMPLATE = "[Absorbed fact] {fact}\nContinuing: Question: {q}\nAnswer:"
PLAIN_TEMPLATE = "Question: {q}\nAnswer:"


# --------------------------------------------------------------- backends

class EchoBackend:
    """Deterministic stub: echoes any absorbed fact, else says 'I do not know'."""

    name = "echo"

    def generate(self, prompt: str, *, max_tokens: int = 80) -> str:
        if "[Absorbed fact]" in prompt:
            after = prompt.split("[Absorbed fact]", 1)[1]
            fact = after.split("\nContinuing:", 1)[0].strip()
            return fact
        return "I do not know."


class HFBackend:
    """Real HuggingFace + optional MT adapter backend."""

    name = "hf"

    def __init__(self, model_name: str, adapter_path: str | None = None,
                 max_tokens: int = 60, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state

        self._torch = torch
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=None
        )
        if adapter_path:
            ckpt = torch.load(adapter_path, map_location="cpu")
            attach_adapters_from_checkpoint(self.model, ckpt)
            load_adapter_state(self.model, adapter_path, strict=False)
        self.model.to(self.device).eval()

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        n = max_tokens or self.max_tokens
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=n,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-6),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return text.strip()


# --------------------------------------------------------------- runner

def run(questions: List[Dict], backend, max_n: int | None = None) -> Dict:
    if max_n:
        questions = questions[:max_n]
    rows = []
    n_hit_plain = n_hit_inject = 0
    t0 = time.time()
    for q in questions:
        p_plain = PLAIN_TEMPLATE.format(q=q["question"])
        p_inj = INJECT_TEMPLATE.format(fact=q["fact"], q=q["question"])
        a_plain = backend.generate(p_plain)
        a_inj = backend.generate(p_inj)
        h_plain = hit(q["answer"], a_plain)
        h_inj = hit(q["answer"], a_inj)
        n_hit_plain += int(h_plain)
        n_hit_inject += int(h_inj)
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "reference_answer": q["answer"],
            "no_inject_generated": a_plain,
            "inject_generated": a_inj,
            "no_inject_hit": h_plain,
            "inject_hit": h_inj,
        })
    n = len(rows)
    acc_plain = n_hit_plain / n if n else 0.0
    acc_inj = n_hit_inject / n if n else 0.0
    return {
        "backend": backend.name,
        "n_questions": n,
        "no_inject_accuracy": round(acc_plain, 4),
        "inject_accuracy": round(acc_inj, 4),
        "uplift_abs": round(acc_inj - acc_plain, 4),
        "uplift_rel": round((acc_inj - acc_plain) / acc_plain, 4) if acc_plain else None,
        "wall_s": round(time.time() - t0, 2),
        "per_question": rows,
    }


# ---------------------------------------------------------------- CLI

def build_backend(args) -> object:
    if args.backend == "echo":
        return EchoBackend()
    if args.backend == "hf":
        return HFBackend(args.model, args.adapter, max_tokens=args.max_tokens,
                         temperature=args.temperature)
    raise ValueError(args.backend)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="benchmarks/cloud_inject_questions.json")
    p.add_argument("--backend", choices=["echo", "hf"], default="echo")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default=None)
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_n", type=int, default=None)
    p.add_argument("--out", default="benchmarks/cloud_inject_uplift.json")
    args = p.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    backend = build_backend(args)
    report = run(questions, backend, max_n=args.max_n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[{report['backend']}] n={report['n_questions']}  "
          f"no_inject={report['no_inject_accuracy']:.3f}  "
          f"inject={report['inject_accuracy']:.3f}  "
          f"uplift=+{report['uplift_abs']:.3f}  "
          f"({report['wall_s']}s)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
