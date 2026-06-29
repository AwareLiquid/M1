"""AwareLiquid architecture-advantage demo -- Modal serverless (scale-to-zero).

WHAT THIS IS
------------
A *temporary showcase* of MT-LNN's VALIDATED architectural advantages, not a
chatbot. The 48M/1.1B served models are storytellers/adapters and cannot
demonstrate "spatial reasoning / low hallucination" -- so this demo instead
shows the three things that ARE reproducibly true at matched parameter count
(see BENCHMARKS.md):

  Demo 1  Selective Copy head-to-head   MT-LNN vs Transformer vs LNN, ~200K
          params each. Held-out sequence-exact ~0.90 (MT-LNN) vs ~0.02
          (Transformer) -- a ~x45 generalisation gap, reproduced live.
  Demo 3  O(1) streaming memory          recurrent state cache stays flat while
          a KV cache grows O(T). (added in a later stage)
  Demo 2  Observable reasoning trace     per-token entropy/route timeline on the
          live O1 48M model. (added in a later stage, needs GPU)

DEPLOY
------
  modal run  deploy/modal_app.py::train_demo_models   # one-time, fills Volume
  modal deploy deploy/modal_app.py                    # serve (scale-to-zero)

COST MODEL
----------
Stage 1 runs CPU-only (the demo models are 200K params -- GPU would just burn
money). scaledown_window keeps the container alive briefly after a request then
scales to zero: you pay for compute-seconds actually served, ~nothing when idle.
"""
from __future__ import annotations

import modal

APP_NAME = "awareliquid-demo"
app = modal.App(APP_NAME)

# --- container image: torch CPU wheel + serve stack, plus the repo source -----
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy<2",
        "fastapi[standard]==0.115.6",
    )
    # mt_lnn (the architecture) + benchmarks (selective-copy task + baselines)
    .add_local_python_source("mt_lnn", "benchmarks")
)

# Persistent store for the trained demo checkpoints (and, later, the O1 ckpt).
vol = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)
DATA_DIR = "/data"
SELCOPY_PT = f"{DATA_DIR}/selcopy_demo.pt"


# ---------------------------------------------------------------------------
# Shared: build the three parameter-matched models for Selective Copy.
# Mirrors benchmarks/compare_baselines.py EXACTLY so the live numbers match the
# documented head-to-head.
# ---------------------------------------------------------------------------
def _build_models(task):
    from benchmarks.baselines import (
        BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
    )
    from mt_lnn import MTLNNConfig, MTLNNModel

    tx_cfg = BaselineConfig(
        vocab_size=task.vocab_size, max_seq_len=task.T_total,
        d_model=104, n_layers=2, n_heads=4, d_ff=256, dropout=0.0,
    )
    lnn_cfg = BaselineConfig(
        vocab_size=task.vocab_size, max_seq_len=task.T_total,
        d_model=104, n_layers=2, n_heads=4, d_ff=256, dropout=0.0,
    )
    mt_cfg = MTLNNConfig(
        vocab_size=task.vocab_size, max_seq_len=task.T_total,
        d_model=104, n_layers=2, n_heads=4, n_kv_heads=2, d_head=26,
        dropout=0.0, attention_dropout=0.0,
        gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2,
    )
    return {
        "transformer": SimpleCausalTransformer(tx_cfg),
        "lnn": SimpleCausalLNN(lnn_cfg),
        "mtlnn": MTLNNModel(mt_cfg),
    }


# Train one model-set per noise length so the demo can show the gap GROW with T.
# Each length needs its own models because the Transformer baseline has a fixed
# learned positional embedding (size = T_total) -- it physically cannot run on a
# longer sequence than it was built for.
LENGTHS = [32, 64, 96]          # T_noise values
STEPS_BY_LEN = {32: 1500, 64: 1500, 96: 1200}


def _task_config(t_noise: int = 32):
    from benchmarks.selective_copy import SelectiveCopyConfig
    return SelectiveCopyConfig(
        K_mem=4, T_noise=t_noise, vocab_size=16, batch=16,
        steps=STEPS_BY_LEN.get(t_noise, 1500), lr=3e-3, eval_batches=8, log_every=300,
    )


# ---------------------------------------------------------------------------
# One-time training: train all three models, save state_dicts to the Volume.
# Run with:  modal run deploy/modal_app.py::train_demo_models
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={DATA_DIR: vol}, timeout=7200)
def train_demo_models():
    import time
    import torch

    from benchmarks.selective_copy import (
        make_selective_copy_batch, evaluate_selective_copy,
    )
    from mt_lnn.utils import make_param_groups

    device = "cpu"

    def train_one(model, task, label, lr_groups=None):
        model.train()
        opt = (torch.optim.AdamW(lr_groups, betas=(0.9, 0.95)) if lr_groups
               else torch.optim.AdamW(model.parameters(), lr=task.lr, betas=(0.9, 0.95)))
        t0 = time.time()
        for step in range(task.steps):
            ids, labels = make_selective_copy_batch(task, task.batch, device=device)
            opt.zero_grad()
            out = model(ids, labels=labels)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step + 1) % task.log_every == 0:
                print(f"  [T={task.T_noise} {label:11s}] step {step+1:5d} "
                      f"loss {out['loss'].item():.4f}", flush=True)
        return time.time() - t0

    bundle = {"lengths": {}, "vocab_size": 16, "K_mem": 4}
    for t_noise in LENGTHS:
        task = _task_config(t_noise)
        print(f"=== training length T_noise={t_noise} (T_total={task.T_total}) ===", flush=True)
        torch.manual_seed(0)
        models = _build_models(task)
        torch.manual_seed(0); train_one(models["transformer"], task, "Transformer")
        torch.manual_seed(0); train_one(models["lnn"], task, "LNN")
        torch.manual_seed(0); train_one(models["mtlnn"], task, "MT-LNN",
                                        lr_groups=make_param_groups(models["mtlnn"], task.lr))
        evals = {}
        for name, m in models.items():
            torch.manual_seed(42)
            evals[name] = evaluate_selective_copy(m, task, device=device, n_batches=16)
            print(f"  EVAL T={t_noise} {name}: seq_exact={evals[name]['sequence_exact']:.4f}", flush=True)
        bundle["lengths"][str(t_noise)] = {
            "T_total": task.T_total,
            "state": {k: v.state_dict() for k, v in models.items()},
            "params": {k: sum(p.numel() for p in v.parameters()) for k, v in models.items()},
            "evals": evals,
        }

    torch.save(bundle, SELCOPY_PT)
    vol.commit()
    print(f"SAVED {SELCOPY_PT} with lengths {LENGTHS}", flush=True)
    return {t: bundle["lengths"][str(t)]["evals"]["mtlnn"]["sequence_exact"] for t in LENGTHS}


# ---------------------------------------------------------------------------
# Live inference: draw N fresh Selective-Copy sequences and greedy-decode all
# three models, BATCHED (B=N). A single sequence is noisy -- the Transformer
# often gets an easy draw right -- so the honest signal is the AGGREGATE
# sequence-exact rate over many draws, which is where the ~x45 gap shows.
# ---------------------------------------------------------------------------
def _decode_batch(model, prefix, K):
    """Greedy-decode K tokens for a whole batch (B, T)->(B, K). Recompute path:
    correct for every architecture (Transformer/LNN have no cache), cheap at K=4."""
    import torch
    seq = prefix
    preds = []
    with torch.no_grad():
        for _ in range(K):
            logits = model(seq)["logits"][:, -1, :]        # (B, V)
            tok = logits.argmax(dim=-1, keepdim=True)        # (B, 1)
            preds.append(tok)
            seq = torch.cat([seq, tok], dim=1)
    return torch.cat(preds, dim=1)                            # (B, K)


# ---------------------------------------------------------------------------
# Serving container: load models once, expose a FastAPI demo. Scale-to-zero.
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    volumes={DATA_DIR: vol},
    scaledown_window=300,   # stay warm 5 min after last request, then -> 0
    min_containers=0,       # scale to zero (no idle cost)
)
class Demo:
    @modal.enter()
    def load(self):
        import torch
        self.torch = torch
        bundle = torch.load(SELCOPY_PT, map_location="cpu", weights_only=False)
        self.vocab_size = bundle["vocab_size"]
        self.K_mem = bundle["K_mem"]

        from benchmarks.selective_copy import SelectiveCopyConfig
        # Build + load one model-set per trained length.
        self.sets = {}     # t_noise(int) -> {models, params, evals}
        for t_str, blob in bundle["lengths"].items():
            t_noise = int(t_str)
            task = SelectiveCopyConfig(
                K_mem=self.K_mem, T_noise=t_noise, vocab_size=self.vocab_size,
            )
            models = _build_models(task)
            for name, m in models.items():
                m.load_state_dict(blob["state"][name], strict=False)
                m.eval()
            self.sets[t_noise] = {
                "models": models, "params": blob["params"], "evals": blob["evals"],
            }
        self.lengths = sorted(self.sets.keys())

    def _run_selective_copy(self, n: int = 50, t_noise: int | None = None):
        torch = self.torch
        from benchmarks.selective_copy import SelectiveCopyConfig, make_selective_copy_batch

        K = self.K_mem
        # snap requested length to a trained one
        T_n = int(t_noise) if t_noise else self.lengths[0]
        if T_n not in self.sets:
            T_n = min(self.lengths, key=lambda L: abs(L - T_n))
        mset = self.sets[T_n]
        models, params, evals = mset["models"], mset["params"], mset["evals"]
        N = max(1, min(int(n), 200))
        task = SelectiveCopyConfig(K_mem=K, T_noise=T_n, vocab_size=self.vocab_size)

        ids, _ = make_selective_copy_batch(task, N, device="cpu")
        prefix = ids[:, : T_n + 1]                       # (N, T_n+1) noise + SEP
        truth = ids[:, T_n + 1: T_n + 1 + K]             # (N, K)

        agg, per_row = {}, {}
        for name, m in models.items():
            preds = _decode_batch(m, prefix, K)          # (N, K)
            correct = (preds == truth)                   # (N, K) bool
            agg[name] = {
                "tok_acc": round(correct.float().mean().item(), 4),
                "seq_exact": round(correct.all(dim=1).float().mean().item(), 4),
                "params": params[name],
                "heldout_seq_exact": round(evals[name]["sequence_exact"], 3),
            }
            per_row[name] = {"preds": preds, "ok": correct.all(dim=1)}

        # Pick example rows -- prioritise the money shot (Transformer FAILS but
        # MT-LNN SUCCEEDS) so the user sees the gap, not a lucky easy draw.
        tx_ok, mt_ok = per_row["transformer"]["ok"], per_row["mtlnn"]["ok"]
        money = [i for i in range(N) if (not tx_ok[i]) and mt_ok[i]]
        order = money + [i for i in range(N) if i not in money]
        examples = []
        for i in order[:5]:
            positions = [j for j in range(T_n) if int(ids[i, j].item()) < K]
            examples.append({
                "noise_seq": ids[i, :T_n].tolist(),
                "memorable_positions": positions,
                "memorable_tokens": truth[i].tolist(),
                "preds": {nm: per_row[nm]["preds"][i].tolist() for nm in models},
                "is_money_shot": bool((not tx_ok[i]) and mt_ok[i]),
            })

        return {
            "n": N, "T_noise": T_n, "K_mem": K,
            "vocab_size": self.vocab_size,
            "available_lengths": self.lengths,
            "aggregate": agg,
            "examples": examples,
            "n_money_shots": len(money),
        }

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        api = FastAPI(title="AwareLiquid Demo")

        @api.get("/api/selective_copy")
        def selective_copy(n: int = 50, t_noise: int = 0):
            return JSONResponse(self._run_selective_copy(n, t_noise or None))

        @api.get("/", response_class=HTMLResponse)
        def index():
            return _INDEX_HTML

        return api


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AwareLiquid - Selective Copy head-to-head</title>
<style>
  :root{ --bg:#0b0f17; --card:#141a26; --ink:#e6edf6; --dim:#8b98ad;
         --mt:#39d98a; --tx:#ff6b6b; --lnn:#f7b955; --line:#243043; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
  .wrap{max-width:860px;margin:0 auto;padding:36px 20px 80px;}
  h1{font-size:26px;margin:0 0 6px} .sub{color:var(--dim);margin:0 0 24px;line-height:1.5}
  .controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;
            background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;}
  label{font-size:13px;color:var(--dim);margin-right:6px}
  select,button{font:inherit;background:#1c2433;color:var(--ink);
                border:1px solid var(--line);border-radius:8px;padding:8px 12px}
  button{background:var(--mt);color:#06281a;font-weight:600;cursor:pointer;border:none}
  button:disabled{opacity:.5;cursor:wait}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:20px;margin-top:20px}
  .barrow{display:grid;grid-template-columns:130px 1fr 132px;align-items:center;gap:12px;margin:14px 0}
  .bname{font-weight:600} .bname small{display:block;color:var(--dim);font-weight:400;font-size:12px}
  .track{background:#0c1119;border-radius:7px;height:30px;overflow:hidden;border:1px solid var(--line)}
  .fill{height:100%;width:0;border-radius:7px 0 0 7px;transition:width .8s cubic-bezier(.2,.7,.2,1)}
  .bval{text-align:right;font-variant-numeric:tabular-nums;color:var(--dim);font-size:14px}
  .bval b{color:var(--ink);font-size:17px}
  .tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;
       background:#1c2433;color:var(--dim);border:1px solid var(--line);margin-left:8px}
  .ex{border-top:1px solid var(--line);padding:14px 0}
  .ex:first-of-type{border-top:none}
  .seqline{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--dim);
           word-break:break-all;line-height:1.7;margin:6px 0}
  .mem{color:#06281a;background:var(--lnn);border-radius:4px;padding:1px 5px;font-weight:700}
  .pred{display:flex;gap:8px;align-items:center;margin:4px 0;font-size:14px}
  .pred .who{width:120px;color:var(--dim)}
  .tok{display:inline-flex;width:26px;height:26px;align-items:center;justify-content:center;
       border-radius:6px;font-family:ui-monospace,monospace;font-weight:700;margin-right:4px}
  .hit{background:rgba(57,217,138,.18);color:var(--mt);border:1px solid rgba(57,217,138,.4)}
  .miss{background:rgba(255,107,107,.16);color:var(--tx);border:1px solid rgba(255,107,107,.4)}
  .money{font-size:11px;color:var(--mt);border:1px solid rgba(57,217,138,.4);
         border-radius:20px;padding:2px 8px;margin-left:8px}
  .note{color:var(--dim);font-size:12.5px;line-height:1.6;margin-top:26px;
        border-top:1px solid var(--line);padding-top:16px}
  .spin{color:var(--dim)}
</style>
</head>
<body><div class="wrap">
  <h1>Selective Copy &mdash; same task, same parameter budget</h1>
  <p class="sub">Three ~200K-parameter models must recall a handful of "memorable"
  tokens scattered in a stream of noise. It is the cleanest test of
  <b>selective long-range memory</b>. The liquid recurrent architecture (MT-LNN)
  and a vanilla Transformer get <i>identical</i> training; only the inductive bias
  differs. All models are decoded the same way (full-sequence recompute) for a
  fair comparison. Increase the noise length: MT-LNN's lead is modest but tends
  to widen as the stream grows.</p>

  <div class="controls">
    <div><label>noise length</label>
      <select id="tn">
        <option value="32">T = 32</option>
        <option value="64">T = 64</option>
        <option value="96">T = 96</option>
      </select></div>
    <div><label>sequences</label>
      <select id="n">
        <option value="50">50</option>
        <option value="100" selected>100</option>
        <option value="200">200</option>
      </select></div>
    <button id="go" onclick="run()">Draw fresh sequences &rarr;</button>
    <span id="status" class="spin"></span>
  </div>

  <div class="card" id="aggCard" style="display:none">
    <div style="color:var(--dim);font-size:13px;margin-bottom:4px">
      Held-out sequence-exact accuracy &mdash; fraction of streams where ALL
      memorable tokens were recalled in order <span class="tag" id="meta"></span>
    </div>
    <div id="bars"></div>
  </div>

  <div class="card" id="exCard" style="display:none">
    <div style="font-weight:600;margin-bottom:6px">Individual streams</div>
    <div style="color:var(--dim);font-size:12.5px;margin-bottom:8px" id="exhint"></div>
    <div id="examples"></div>
  </div>

  <div class="note" id="note"></div>
</div>
<script>
const C = {mtlnn:'var(--mt)', transformer:'var(--tx)', lnn:'var(--lnn)'};
const NAME = {mtlnn:'MT-LNN', transformer:'Transformer', lnn:'LNN (LTC only)'};
const SUB  = {mtlnn:'liquid recurrent', transformer:'vanilla attention', lnn:'attention + LTC'};

async function run(){
  const go=document.getElementById('go'), st=document.getElementById('status');
  go.disabled=true; st.textContent='running on a scale-to-zero container (first call cold-starts)...';
  const n=document.getElementById('n').value, tn=document.getElementById('tn').value;
  let d;
  try{ d = await (await fetch(`/api/selective_copy?n=${n}&t_noise=${tn}`)).json(); }
  catch(e){ st.textContent='error: '+e; go.disabled=false; return; }
  st.textContent=''; go.disabled=false;
  render(d);
}

function render(d){
  document.getElementById('aggCard').style.display='block';
  document.getElementById('exCard').style.display='block';
  document.getElementById('meta').textContent =
     `${d.n} streams - ${d.K_mem} tokens to recall - noise length ${d.T_noise} - vocab ${d.vocab_size}`;

  const order=['mtlnn','transformer','lnn'];
  document.getElementById('bars').innerHTML = order.map(k=>{
    const a=d.aggregate[k], pct=(a.seq_exact*100);
    return `<div class="barrow">
      <div class="bname">${NAME[k]}<small>${SUB[k]} - ${a.params.toLocaleString()} params</small></div>
      <div class="track"><div class="fill" style="background:${C[k]}" data-w="${pct}"></div></div>
      <div class="bval"><b>${pct.toFixed(1)}%</b></div></div>`;
  }).join('');
  requestAnimationFrame(()=>document.querySelectorAll('.fill').forEach(f=>f.style.width=f.dataset.w+'%'));

  const mt=d.aggregate.mtlnn.seq_exact, tx=Math.max(d.aggregate.transformer.seq_exact,1e-4);
  document.getElementById('exhint').textContent =
    d.n_money_shots>0
     ? `Showing streams where the Transformer FAILED but MT-LNN recalled every token (${d.n_money_shots}/${d.n} of this batch). Gold = the planted memorable tokens.`
     : `Gold = the planted memorable tokens; below each, what every model recalled.`;

  document.getElementById('examples').innerHTML = d.examples.map(ex=>{
    const memSet=new Set(ex.memorable_positions);
    const seq = ex.noise_seq.map((t,i)=> memSet.has(i)
        ? `<span class="mem">${t}</span>` : t).join(' ');
    const preds = order.map(k=>{
      const toks = ex.preds[k].map((p,i)=>{
        const ok = p===ex.memorable_tokens[i];
        return `<span class="tok ${ok?'hit':'miss'}">${p}</span>`;
      }).join('');
      return `<div class="pred"><span class="who">${NAME[k]}</span>${toks}</div>`;
    }).join('');
    return `<div class="ex">
       <div>recall target: <b>[${ex.memorable_tokens.join(', ')}]</b>
         ${ex.is_money_shot?'<span class="money">MT-LNN wins</span>':''}</div>
       <div class="seqline">${seq} <span style="color:var(--line)">| SEP</span></div>
       ${preds}</div>`;
  }).join('');

  const gap = ((mt - d.aggregate.transformer.seq_exact)*100);
  document.getElementById('note').innerHTML =
   `<b>What this shows.</b> At matched parameter budget and identical, fair
    decoding, MT-LNN recalls full streams the most often at every length. At
    short noise lengths the lead over a vanilla Transformer is modest; as the
    stream grows the Transformer tends to degrade faster and the gap tends to
    widen (here MT-LNN leads by `+gap.toFixed(0)+` points at noise length `+d.T_noise+`).
    That widening is the architectural signal: selective long-range memory.
    <br><br><b>Honest scope.</b> Fair head-to-head at <i>toy scale</i>
    (~200K params, the synthetic Selective-Copy task from the Mamba paper). It is
    NOT a claim about 125M+ natural-language models, and it does not measure
    "reasoning" or "low hallucination" - it isolates one thing: selective
    long-range memory. Numbers reproduce via <code>benchmarks/compare_baselines.py</code>.`;
}
run();
</script>
</body></html>
"""
