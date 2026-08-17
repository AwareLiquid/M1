"""MMLU 5-shot probe for M1 (TinyLlama-1.1B + MT v2s + LoRA), 20 items.
Determines the REAL capability number for the deck's cost-vs-capability chart.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DLL-order guard (Windows): datasets/pyarrow BEFORE transformers' tokenizers.
import datasets as _ds  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import datasets as _ds
from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters

def main():
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
    ck = torch.load("checkpoints/llama_mt_adapter/llama_mt_adapter_v2s_003000.pt",
                    map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
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
        fast_weight_heads=int(cargs.get("v2_fw_heads", 1)))
    m = get_peft_model(m, LoraConfig(
        r=int(cargs.get("lora_r", 8)), lora_alpha=int(cargs.get("lora_alpha", 16)),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=str(cargs.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")).split(",")))
    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
    assert len(unexpected) == 0, f"{len(unexpected)} did not map"
    from mt_lnn.llama_adapter import set_adapter_streaming
    set_adapter_streaming(m, True)
    m.eval()

    dev = list(_ds.load_dataset("cais/mmlu", "college_computer_science", split="dev"))[:5]
    test = list(_ds.load_dataset("cais/mmlu", "college_computer_science", split="test"))[:20]

    def fmt(r, answer=False):
        q = (r["question"] + "\nA. " + r["choices"][0] + "\nB. " + r["choices"][1]
             + "\nC. " + r["choices"][2] + "\nD. " + r["choices"][3] + "\nAnswer:")
        return q + (" " + "ABCD"[r["answer"]] + "\n\n" if answer else "")

    shots = "".join(fmt(d, True) for d in dev)
    correct = 0
    for i, r in enumerate(test):
        prompt = shots + fmt(r, False)
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=8, do_sample=False)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        pred = gen[0].upper() if gen else ""
        ans = "ABCD"[r["answer"]]
        ok = pred == ans
        correct += int(ok)
        print(f"[{i+1}/20] pred={pred!r} ans={ans} ok={ok} raw={gen[:30]!r}", flush=True)
    print(f"MMLU 5-shot 20题: {correct}/20 = {correct/20:.2f}")

if __name__ == "__main__":
    main()
