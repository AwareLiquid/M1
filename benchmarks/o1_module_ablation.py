"""O1 native-model module ablation — the switch-matrix, leave-one-in form.

7 configs, identical data/steps/optimizer/seed (TinyStories, 48M-class O1):
  core      all optional modules OFF (incl. predictive coding)
  pc        predictive coding only (the config default)
  gwtb      competitive GWTB only
  wm        world model only
  rhythm    LAVI rhythm gate only
  hebbian   Hebbian regularizer only
  full      everything ON

Reports final val PPL + throughput per config: which brain-inspired modules
pay rent, which get archived as negative results. Resume-safe: finished
configs (per-config .log present with a final val line) are skipped, so the
sweep can span multiple free-GPU sessions.

Platform-neutral: no Kaggle/Colab paths. Typical Colab cell:
    !git clone --depth 1 https://github.com/everest-an/M1.git /content/M1
    %cd /content/M1
    !python benchmarks/o1_module_ablation.py --out /content/drive/MyDrive/ablate_out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = {
    "core":    ["--no_predictive_coding"],
    "pc":      [],
    "gwtb":    ["--no_predictive_coding", "--competitive_gwtb", "--n_bids", "3"],
    "wm":      ["--no_predictive_coding", "--world_model"],
    "rhythm":  ["--no_predictive_coding", "--rhythm"],
    "hebbian": ["--no_predictive_coding", "--hebbian"],
    "full":    ["--competitive_gwtb", "--n_bids", "3", "--world_model",
                "--rhythm", "--hebbian"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmarks/o1_ablate_out")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--max_tokens", type=int, default=40_000_000)
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs(args.out, exist_ok=True)
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false")

    if not os.path.exists("data/meta.json"):
        subprocess.check_call([sys.executable, "prepare_data.py",
            "--dataset", "roneneldan/TinyStories", "--config", "none",
            "--tokenizer", "gpt2", "--max_tokens", str(args.max_tokens),
            "--out_dir", "data"], env=env)

    wanted = (list(CONFIGS) if args.configs == "all"
              else [c for c in args.configs.split(",") if c in CONFIGS])
    results = {}
    for name in wanted:
        log_path = os.path.join(args.out, f"{name}.log")
        if os.path.exists(log_path) and "val PPL" in open(log_path).read():
            print(f"[skip] {name} (finished log exists)", flush=True)
        else:
            cmd = [sys.executable, "train.py",
                   "--d_model", "416", "--n_layers", "6", "--n_heads", "13",
                   "--n_kv_heads", "1", "--seq_len", str(args.seq_len),
                   "--batch", str(args.batch),
                   "--grad_accum", str(args.grad_accum),
                   "--lr", "6e-4", "--warmup_steps", "200",
                   "--steps", str(args.steps),
                   "--save_every", "1000000",
                   "--eval_every", str(args.steps // 3),
                   "--eval_batches", "50",
                   "--ckpt_dir", os.path.join(args.out, f"ck_{name}"),
                   "--data_dir", "data"] + CONFIGS[name]
            print(f"=== {name} === {' '.join(CONFIGS[name])}", flush=True)
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    lf.write(line)
                    lf.flush()
                    if "val PPL" in line or "Parameters:" in line:
                        print(f"[{name}] {line.rstrip()}", flush=True)
                if p.wait() != 0:
                    raise SystemExit(f"{name} failed — see {log_path}")

        text = open(log_path).read()
        ppls = re.findall(r"val PPL: ([0-9.]+)", text)
        toks = re.findall(r"([0-9]+) tok/s", text)
        results[name] = {
            "val_ppl": float(ppls[-1]) if ppls else None,
            "val_loss": round(math.log(float(ppls[-1])), 4) if ppls else None,
            "tok_s": int(toks[-1]) if toks else None,
        }

    with open(os.path.join(args.out, "ablation_summary.json"), "w") as f:
        json.dump({"steps": args.steps, "results": results}, f, indent=2)
    print("\n" + "=" * 58, flush=True)
    print(f"{'config':<10} {'val_ppl':>9} {'val_loss':>9} {'tok/s':>8}", flush=True)
    for k, v in results.items():
        print(f"{k:<10} {str(v['val_ppl']):>9} {str(v['val_loss']):>9} "
              f"{str(v['tok_s']):>8}", flush=True)


if __name__ == "__main__":
    main()
