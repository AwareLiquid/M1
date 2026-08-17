"""Quick capability probe for M1 (TinyLlama-1.1B + MT v2s adapter) on
standard eval subsets, CPU.

Purpose: get REAL numbers for the "cost-vs-capability" deck chart.
M1 is a 1.1B base + ~1% adapter; expect MMLU ~25-30%, GSM8K low.
This is honest positioning: we do NOT claim AGI-benchmark parity; we
chart cost-per-task vs capability on the tasks we CAN do.

Usage: python benchmarks/probe_m1_evals.py --limit 20 --shots 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# DLL-order guard (Windows): datasets/pyarrow before transformers' tokenizers.
import datasets as _datasets  # noqa: F401


def load_m1(ckpt_path: str, device: str):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    m.config.use_cache = True

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    # ORDER MATTERS (matches benchmarks/capability_eval.py): MT v2 adapters
    # attach onto the UNWRAPPED model; LoRA (get_peft_model) wraps after.
    from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
    attach_mt_v2_adapters(
        m, every=int(cargs.get("mt_every", 4)),
        n_protofilaments=int(cargs.get("mt_proto", 13)),
        d_proto=int(cargs.get("v2_d_proto", 64)),
        n_time_scales=int(cargs.get("mt_scales", 5)),
        proj_rank=int(cargs.get("v2_rank", 128)),
        init_scale=float(cargs.get("mt_init_scale", 1e-3)),
        selective_decay=bool(cargs.get("v2_selective", False)),
        use_fast_weight=not bool(cargs.get("v2_no_fw", False)),
        fast_weight_dim=int(cargs.get("v2_fw_dim", 64)),
        fast_weight_heads=int(cargs.get("v2_fw_heads", 1)),
    )
    m = get_peft_model(m, LoraConfig(
        r=int(cargs.get("lora_r", 8)),
        lora_alpha=int(cargs.get("lora_alpha", 16)),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=str(cargs.get(
            "lora_targets", "q_proj,k_proj,v_proj,o_proj")).split(","),
    ))
    sd = ck.get("state_dict", {})
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, f"{len(unexpected)} tensors did not map"
    print(f"[probe] loaded {ckpt_path}: {len(sd)} tensors, "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    from mt_lnn.llama_adapter import set_adapter_streaming
    set_adapter_streaming(m, True)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/llama_mt_adapter/llama_mt_adapter_v2s_003000.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--shots", type=int, default=5)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok.pad_token = tok.eos_token

    model = load_m1(args.ckpt, args.device)

    results = {}
    for task in ["mmlu", "gsm8k"]:
        try:
            if task == "mmlu":
                ds = _datasets.load_dataset("cais/mmlu", "college_computer_science", split="test")
            else:
                ds = _datasets.load_dataset("gsm8k", "main", split="test")
        except Exception as e:
            print(f"[probe] {task} dataset error: {e}")
            results[task] = {"error": str(e)[:200]}
            continue

        rows = list(ds)[: args.limit]
        correct = 0
        total = 0
        t0 = time.time()
        for r in rows:
            if task == "mmlu":
                prompt = (r["question"] + "\nA. " + r["choices"][0] + "\nB. " +
                          r["choices"][1] + "\nC. " + r["choices"][2] + "\nD. " +
                          r["choices"][3] + "\nAnswer:")
                answer_letter = "ABCD"[r["answer"]]
            else:
                prompt = r["question"] + "\nAnswer:"
                answer_letter = None
            ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=32, do_sample=False)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if task == "mmlu":
                pred = gen[0].upper() if gen else ""
                ok = pred == answer_letter
            else:
                nums = re.findall(r"-?\d[\d,]*", gen)
                ans_nums = re.findall(r"-?\d[\d,]*", r["answer"].split("####")[-1])
                ok = bool(nums) and bool(ans_nums) and \
                    nums[0].replace(",", "") == ans_nums[0].replace(",", "")
            correct += int(ok)
            total += 1
            print(f"  [{task}] {total}/{args.limit} ok={ok} gen={gen[:40]!r}", flush=True)
        wall = time.time() - t0
        results[task] = {"acc": correct / total if total else 0.0,
                         "n": total, "correct": correct,
                         "wall_s": round(wall, 1),
                         "s_per_item": round(wall / total, 2) if total else 0}
        print(f"[probe] {task}: {correct}/{total} = {correct/total:.3f} "
              f"({wall:.0f}s, {wall/total:.1f}s/item)", flush=True)

    out = os.path.join("benchmarks", "results", "m1_evals_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "device": args.device,
                   "base": "TinyLlama-1.1B-Chat-v1.0",
                   "results": results}, f, indent=2)
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
