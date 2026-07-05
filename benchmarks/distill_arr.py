"""Distill a pretrained Llama into an attention-free recurrent model (ARR).

Two-stage MOHAWK-lite on WikiText-2 (or any text corpus):

  Stage A — hidden alignment: run TEACHER (frozen original) with
    output_hidden_states; train each converted layer so the student's
    hidden stream matches the teacher's layer-by-layer (MSE, normalised).
    This gives every mixer a dense, layer-local learning signal — far
    stronger than end-to-end CE through 22 layers.

  Stage B — knowledge distillation: KL(teacher logits || student logits)
    at temperature tau, mixed with plain CE. The student's frozen MLPs /
    embeddings are the teacher's own weights, so KD only has to teach the
    mixers to ROUTE like attention did.

Evaluates WikiText-2 test PPL (teacher vs student) at the end. Expectation
management: on a single P100 budget the student will NOT reach teacher
parity — the goal is a working pipeline and an honest first number that
scales with compute.

Usage:
    python benchmarks/distill_arr.py --steps_a 1000 --steps_b 2000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from benchmarks.attribution_ablation import build_chunks  # DLL-order safe

from mt_lnn.arr import (convert_to_arr, count_mixer_parameters,
                        iter_mixer_parameters, probe_return_convention)


class InputTap:
    """Forward PRE-hooks capturing each decoder layer's input hidden state
    (args[0] in Llama's layer call). Used to teacher-force stage A."""

    def __init__(self, layers, indices):
        self.acts = {}
        self.handles = [
            layers[i].register_forward_pre_hook(self._make(i)) for i in indices
        ]

    def _make(self, i):
        def hook(_mod, inp):
            self.acts[i] = inp[0]
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


class LayerTap:
    """Forward hooks that capture each decoder layer's output tensor.

    Version-proof replacement for output_hidden_states=True: newer
    transformers records hidden states by matching the stock decoder-layer
    CLASS, so custom ARRDecoderLayer outputs silently come back as None.
    Hooks read the true output object (tuple in old versions, tensor in
    new) and keep the tensor — gradients flow through unchanged.
    """

    def __init__(self, layers, indices):
        self.acts = {}
        self.handles = [
            layers[i].register_forward_hook(self._make(i)) for i in indices
        ]

    def _make(self, i):
        def hook(_mod, _inp, out):
            self.acts[i] = out[0] if isinstance(out, tuple) else out
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def eval_ppl(model, chunks, device, dtype, batch=1, max_chunks=0):
    model.eval()
    if max_chunks and max_chunks < len(chunks):
        chunks = chunks[:max_chunks]
    nll, tok = 0.0, 0
    for i in range(0, len(chunks), batch):
        ids = chunks[i: i + batch].to(device)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            out = model(input_ids=ids, labels=ids)
        n = ids.shape[0] * (ids.shape[1] - 1)
        nll += out.loss.float().item() * n
        tok += n
    return math.exp(nll / tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--steps_a", type=int, default=1000,
                    help="stage A: layerwise hidden-alignment steps")
    ap.add_argument("--steps_b", type=int, default=2000,
                    help="stage B: logit-KD steps")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr_a", type=float, default=1e-3)
    ap.add_argument("--lr_b", type=float, default=5e-4)
    ap.add_argument("--kd_tau", type=float, default=2.0)
    ap.add_argument("--kd_alpha", type=float, default=0.7,
                    help="weight of KL vs CE in stage B")
    ap.add_argument("--d_proto", type=int, default=96)
    ap.add_argument("--proj_rank", type=int, default=384)
    ap.add_argument("--fw_dim", type=int, default=96)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="benchmarks/arr_out")
    ap.add_argument("--save_ckpt", action="store_true")
    # Round-2 lessons: 'free' stage A (round 1) ran the full student forward,
    # so layer l trained against its own drifting layer-(l-1) output —
    # compounding error destabilised alignment (norm-MSE spiked to 467).
    # 'teacher_forced' feeds every student layer the TEACHER's input for that
    # layer (MOHAWK stage 2), decoupling all layers.
    ap.add_argument("--align_mode", choices=["teacher_forced", "free"],
                    default="teacher_forced")
    ap.add_argument("--resume", default="",
                    help="mixer checkpoint (.pt) to load before training")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    torch.manual_seed(args.seed)
    print(f"device={device} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_chunks = build_chunks(tok, "train", args.seq_len)
    test_chunks = build_chunks(tok, "test", args.seq_len)
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_chunks), generator=g)
    print(f"train {len(train_chunks)} chunks, test {len(test_chunks)}", flush=True)

    # --- teacher (frozen, untouched) and student (attention -> mixers) ---
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(device).eval()

    student = copy.deepcopy(teacher)
    converted = convert_to_arr(
        student, d_proto=args.d_proto, proj_rank=args.proj_rank,
        fast_weight_dim=args.fw_dim,
    )
    student.to(device)
    rt = probe_return_convention(student)
    print(f"layer return convention: {'tuple' if rt else 'tensor'}", flush=True)
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu",
                        weights_only=False)["state_dict"]
        missing, unexpected = student.load_state_dict(sd, strict=False)
        print(f"resumed mixers: {len(sd) - len(unexpected)}/{len(sd)} tensors",
              flush=True)
    n_mix = count_mixer_parameters(student)
    total = sum(p.numel() for p in student.parameters())
    print(f"converted {len(converted)} layers | mixer params {n_mix:,} "
          f"({100 * n_mix / total:.2f}% of {total:,})", flush=True)

    teacher_ppl = eval_ppl(teacher, test_chunks, device, dtype,
                           args.batch, args.eval_chunks)
    print(f"TEACHER test PPL: {teacher_ppl:.3f}", flush=True)

    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)

    def opt_step(opt, loss):
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def opt_apply(opt):
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

    # ---------------- Stage A: layerwise hidden alignment ----------------
    if args.steps_a > 0:
        from mt_lnn.llama_adapter import find_decoder_layers
        t_layers = find_decoder_layers(teacher)
        s_layers = find_decoder_layers(student)
        t_in = InputTap(t_layers, converted)
        t_tap = LayerTap(t_layers, converted)
        s_tap = (LayerTap(s_layers, converted)
                 if args.align_mode == "free" else None)
        opt = torch.optim.AdamW(iter_mixer_parameters(student), lr=args.lr_a)
        student.train()
        step, t0 = 0, time.time()
        opt.zero_grad(set_to_none=True)
        while step < args.steps_a:
            for idx in order:
                if step >= args.steps_a:
                    break
                ids = train_chunks[int(idx): int(idx) + 1].to(device)
                with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=dtype, enabled=device == "cuda"):
                    teacher(input_ids=ids)
                with torch.amp.autocast("cuda", dtype=dtype,
                                        enabled=device == "cuda"):
                    loss = 0.0
                    if args.align_mode == "teacher_forced":
                        # Each student layer sees the TEACHER's input for that
                        # layer: layers train decoupled, no compounding drift.
                        for l in converted:
                            ti = t_in.acts[l].detach()
                            to = t_tap.acts[l].detach().float()
                            so = s_layers[l](ti)
                            so = (so[0] if isinstance(so, tuple) else so).float()
                            loss = loss + F.mse_loss(so, to) / to.pow(2).mean().clamp_min(1e-6)
                    else:
                        student(input_ids=ids)
                        for l in converted:
                            t = t_tap.acts[l].detach().float()
                            s = s_tap.acts[l].float()
                            loss = loss + F.mse_loss(s, t) / t.pow(2).mean().clamp_min(1e-6)
                    loss = loss / len(converted) / args.grad_accum
                opt_step(opt, loss)
                if (step + 1) % args.grad_accum == 0:
                    opt_apply(opt)
                step += 1
                if step % args.log_every == 0:
                    dt = max(time.time() - t0, 1e-3)
                    print(f"[A] {step}/{args.steps_a} | norm-MSE "
                          f"{loss.item() * args.grad_accum:.4f} | "
                          f"{args.log_every / dt:.2f} it/s", flush=True)
                    t0 = time.time()
        del opt
        t_in.close()
        t_tap.close()
        if s_tap is not None:
            s_tap.close()
        ppl_a = eval_ppl(student, test_chunks, device, dtype,
                         args.batch, args.eval_chunks)
        print(f"after stage A: student test PPL {ppl_a:.3f}", flush=True)
    else:
        ppl_a = None

    # ---------------- Stage B: logit KD + CE ----------------
    if args.steps_b > 0:
        opt = torch.optim.AdamW(iter_mixer_parameters(student), lr=args.lr_b)
        student.train()
        step, t0 = 0, time.time()
        opt.zero_grad(set_to_none=True)
        tau = args.kd_tau
        while step < args.steps_b:
            for idx in order:
                if step >= args.steps_b:
                    break
                ids = train_chunks[int(idx): int(idx) + 1].to(device)
                with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=dtype, enabled=device == "cuda"):
                    t_logits = teacher(input_ids=ids).logits
                with torch.amp.autocast("cuda", dtype=dtype,
                                        enabled=device == "cuda"):
                    out = student(input_ids=ids, labels=ids)
                    s_logits = out.logits
                    kl = F.kl_div(
                        F.log_softmax(s_logits.float() / tau, dim=-1),
                        F.log_softmax(t_logits.float() / tau, dim=-1),
                        log_target=True, reduction="batchmean",
                    ) * tau * tau / s_logits.shape[1]
                    loss = (args.kd_alpha * kl
                            + (1 - args.kd_alpha) * out.loss) / args.grad_accum
                opt_step(opt, loss)
                if (step + 1) % args.grad_accum == 0:
                    opt_apply(opt)
                step += 1
                if step % args.log_every == 0:
                    dt = max(time.time() - t0, 1e-3)
                    print(f"[B] {step}/{args.steps_b} | KD "
                          f"{loss.item() * args.grad_accum:.4f} (kl {kl.item():.4f} "
                          f"ce {out.loss.item():.4f}) | "
                          f"{args.log_every / dt:.2f} it/s", flush=True)
                    t0 = time.time()
        del opt

    student_ppl = eval_ppl(student, test_chunks, device, dtype,
                           args.batch, args.eval_chunks)
    print(f"\nARR DISTILLATION | teacher PPL {teacher_ppl:.3f} | "
          f"student(after A) {ppl_a if ppl_a is None else round(ppl_a, 3)} | "
          f"student(final) {student_ppl:.3f} | mixers {n_mix:,}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "model": args.model, "teacher_ppl": teacher_ppl,
        "student_ppl_after_a": ppl_a, "student_ppl_final": student_ppl,
        "mixer_params": n_mix, "converted_layers": converted,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "arr_result.json"), "w") as f:
        json.dump(payload, f, indent=2)
    if args.save_ckpt:
        torch.save({
            "arr": True, "model": args.model, "args": vars(args),
            "state_dict": {k: v.cpu() for k, v in student.state_dict().items()
                           if "mixer" in k},
        }, os.path.join(args.out_dir, "arr_mixers.pt"))
        print("saved mixer checkpoint", flush=True)


if __name__ == "__main__":
    main()
