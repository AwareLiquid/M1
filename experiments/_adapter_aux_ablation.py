"""Adapter aux-loss ablation: does wiring predictive-coding + Hebbian into the
MT adapter route actually help, or is it just live-but-inert?

Design (fair A/B):
  * Same base (Qwen2.5-0.5B-Instruct, frozen), same adapter geometry, SAME seed
    so both arms start from byte-identical adapter init.
  * Both arms attach adapters with predictive-coding + Hebbian ENABLED (so the
    parameter count / graph is identical). The ONLY difference is whether the
    training objective adds collect_adapter_aux_losses()["aux_loss"].
  * After N steps we report held-out perplexity using PURE next-token CE
    (no aux term) -- the honest language-modelling metric.

HONEST SCOPE: CPU-only, tiny step budget -> this is an EARLY-SIGNAL probe, not a
benchmark. A null / within-noise result is the expected prior (Experiment 5 found
Hebbian inert at this scale). The point is to MEASURE, on the adapter route, with
the now-wired aux path -- not to claim a win.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def _local_path(model_id: str) -> str:
    """Resolve the cached snapshot dir so from_pretrained gets a LOCAL path.

    Passing a local dir makes transformers treat it as local and skip the
    network model-info probe (which fails / hangs offline). Falls back to the
    model id if no snapshot is cached.
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id

from mt_lnn.llama_adapter import (  # noqa: E402
    attach_mt_adapters,
    collect_adapter_aux_losses,
    count_trainable_parameters,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
SEQ_LEN = 128
N_TRAIN_STEPS = 40
N_EVAL_BATCHES = 16
LR = 2e-4
PRED_WEIGHT = 0.1
HEBB_LR = 1e-2  # use the largest sweep value from Exp5 so any effect is visible


MAX_CHARS = 160_000  # cap before tokenizing -- a single huge encode can segfault


def _load_corpus(tokenizer):
    """Return (train_ids, eval_ids): 1-D LongTensors of token ids.

    Offline-safe: build the corpus from local repo text (no datasets/network).
    Both arms share the same split, so this gives a valid RELATIVE comparison of
    whether the aux gradient helps; it is NOT a wikitext-comparable absolute PPL.
    """
    parts = []
    for p in ("serve/static/llms-full.txt",
              "serve/static/research.html",
              "mt_lnn_v2_reliable_long_pretraining_arxiv.tex"):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                parts.append(fh.read())
    text = ("\n".join(parts) * 6)[:MAX_CHARS]

    # Encode in chunks and concat -- avoids a single oversized tokenizer call.
    ids_chunks = []
    for i in range(0, len(text), 8000):
        chunk = text[i:i + 8000]
        ids_chunks.append(tokenizer(chunk, return_tensors="pt").input_ids[0])
    ids = torch.cat(ids_chunks)
    n = ids.numel()
    cut = int(n * 0.9)
    print(f"[data] source=local-repo  chars={len(text):,}  total_tokens={n:,}  "
          f"train={cut:,}  eval={n-cut:,}")
    return ids[:cut], ids[cut:]


def _batches(ids, seq_len, n):
    """Yield up to n contiguous (1, seq_len) blocks."""
    step = seq_len
    out = []
    for i in range(0, ids.numel() - seq_len - 1, step):
        out.append(ids[i:i + seq_len].unsqueeze(0))
        if len(out) >= n:
            break
    return out


def _build():
    """Fresh frozen base + freshly-initialised adapter under a fixed seed."""
    torch.manual_seed(SEED)
    path = _local_path(MODEL)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True
    )
    torch.manual_seed(SEED)  # re-seed so adapter init is identical across arms
    attach_mt_adapters(
        model, every=4, n_protofilaments=13, n_time_scales=5, map_hidden_dim=64,
        init_scale=1e-3, use_predictive_coding=True, use_hebbian=True,
    )
    return tok, model


@torch.no_grad()
def _eval_ppl(model, eval_batches):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for b in eval_batches:
        out = model(input_ids=b)
        logits = out.logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = b[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        tot_loss += loss.item()
        tot_tok += shift_labels.numel()
    return float(torch.tensor(tot_loss / tot_tok).exp())


def _train_arm(aux_on: bool, train_batches, eval_batches):
    tok, model = _build()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    model.train()

    t0 = time.time()
    aux_running = 0.0
    n_seen = 0
    for step in range(N_TRAIN_STEPS):
        b = train_batches[step % len(train_batches)]
        out = model(input_ids=b, labels=b)
        loss = out.loss  # HF next-token CE
        if aux_on:
            aux = collect_adapter_aux_losses(
                model, predictive_weight=PRED_WEIGHT, hebbian_lr=HEBB_LR
            )
            if "aux_loss" in aux:
                aux_running += float(aux["aux_loss"].detach())
                n_seen += 1
                loss = loss + aux["aux_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"    [{'aux-on ' if aux_on else 'aux-off'}] step {step:>3} "
                  f"ce_loss={float(out.loss.detach()):.4f}")

    dt = time.time() - t0
    ppl = _eval_ppl(model, eval_batches)
    mean_aux = (aux_running / n_seen) if n_seen else 0.0
    return ppl, dt, mean_aux


def main():
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    tok, _probe = _build()
    print(f"[setup] model={MODEL}  trainable={count_trainable_parameters(_probe):,}")
    del _probe

    train_ids, eval_ids = _load_corpus(tok)
    train_batches = _batches(train_ids, SEQ_LEN, N_TRAIN_STEPS)
    eval_batches = _batches(eval_ids, SEQ_LEN, N_EVAL_BATCHES)
    print(f"[setup] train_batches={len(train_batches)}  eval_batches={len(eval_batches)}  "
          f"seq_len={SEQ_LEN}  steps={N_TRAIN_STEPS}")

    print("\n=== ARM A: aux-off (CE only) ===")
    ppl_off, dt_off, _ = _train_arm(False, train_batches, eval_batches)
    print(f"  -> held-out PPL = {ppl_off:.3f}   ({dt_off:.1f}s)")

    print("\n=== ARM B: aux-on (CE + predictive + Hebbian) ===")
    ppl_on, dt_on, mean_aux = _train_arm(True, train_batches, eval_batches)
    print(f"  -> held-out PPL = {ppl_on:.3f}   ({dt_on:.1f}s)   mean_aux={mean_aux:.4e}")

    delta = ppl_on - ppl_off
    rel = 100.0 * delta / ppl_off
    print("\n==================== RESULT ====================")
    print(f"  aux-off PPL : {ppl_off:.3f}")
    print(f"  aux-on  PPL : {ppl_on:.3f}")
    print(f"  delta       : {delta:+.3f}  ({rel:+.2f}%)   (negative = aux helped)")
    print("  NOTE: CPU-only, 40 steps -> EARLY SIGNAL only. A within-noise")
    print("        delta is the expected prior; this measures, it does not prove.")
    print("===============================================")


if __name__ == "__main__":
    main()
