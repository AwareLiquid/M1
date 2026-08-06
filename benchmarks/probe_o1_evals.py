"""Quick capability probe for O1-48M on standard evals (subset, CPU).

Purpose: get REAL numbers before building any chart. 48M-125M models
score near random on AGI benchmarks (MMLU ~25% random, GSM8K ~0%); this
script confirms that honestly, so the deck chart can be framed correctly.

Usage: python benchmarks/probe_o1_evals.py --ckpt checkpoints/o1_48m_serve.pt --shots 0 --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# DLL-order guard (Windows): datasets/pyarrow before transformers' tokenizers.
import datasets as _datasets  # noqa: F401


def load_model(ckpt_path: str, device: str):
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        raise RuntimeError(f"{ckpt_path} has no embedded config")
    valid = {f.name for f in __import__("dataclasses").fields(MTLNNConfig) if f.init}
    cfg = MTLNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = MTLNNModel(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[probe] loaded {ckpt_path} step={ckpt.get('step','?')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/o1_48m_serve.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=20, help="items per task")
    ap.add_argument("--shots", type=int, default=0)
    args = ap.parse_args()

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    model = load_model(args.ckpt, args.device)

    results = {}
    for task in ["mmlu", "gsm8k"]:
        # small local subsets; datasets may download on first run
        try:
            if task == "mmlu":
                ds = _datasets.load_dataset("cais/mmlu", "college_computer_science",
                                            split="test", trust_remote_code=True)
            else:
                ds = _datasets.load_dataset("gsm8k", "main", split="test",
                                            trust_remote_code=True)
        except Exception as e:  # pragma: no cover
            print(f"[probe] {task} dataset error: {e}")
            results[task] = {"error": str(e)[:200]}
            continue

        rows = list(ds)[: args.limit]
        correct = 0
        total = 0
        for r in rows:
            if task == "mmlu":
                prompt = r["question"] + "\nA. " + r["choices"][0] + "\nB. " + r["choices"][1] + \
                         "\nC. " + r["choices"][2] + "\nD. " + r["choices"][3] + "\nAnswer:"
                answer_letter = "ABCD"[r["answer"]]
            else:
                prompt = r["question"] + "\nAnswer:"
                answer_letter = None
            ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=24, do_sample=False)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if task == "mmlu":
                pred = gen[0].upper() if gen else ""
                ok = pred == answer_letter
            else:
                # gsm8k: compare final number loosely
                import re
                nums = re.findall(r"-?\d[\d,]*", gen)
                ans_nums = re.findall(r"-?\d[\d,]*", r["answer"].split("####")[-1])
                ok = bool(nums) and bool(ans_nums) and nums[0].replace(",", "") == ans_nums[0].replace(",", "")
            correct += int(ok)
            total += 1
        results[task] = {"acc": correct / total if total else 0.0,
                         "n": total, "correct": correct}
        print(f"[probe] {task}: {correct}/{total} = {correct/total:.3f}", flush=True)

    out = os.path.join("benchmarks", "results", "o1_evals_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "device": args.device,
                   "results": results}, f, indent=2)
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
