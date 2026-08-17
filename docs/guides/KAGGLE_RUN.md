# Kaggle Run — Phase 5 Backbone Training

Notebook: [`kaggle/awareliquid_train_p100.ipynb`](../../kaggle/awareliquid_train_p100.ipynb)

Targets **TinyLlama-1.1B + MT-LNN adapter + LoRA on WikiText-2**, single P100 16GB, ~3–4 h.

## 1. Create the Kaggle notebook

1. <https://www.kaggle.com> → **Code** → **New Notebook**.
2. **File → Import Notebook** → upload `kaggle/awareliquid_train_p100.ipynb` from this repo.
3. Right sidebar → **Settings**:
   - **Accelerator**: `GPU P100`
   - **Internet**: `On` (required for HuggingFace + dataset)
   - **Persistence**: default (no need)

> If `Internet` is greyed out, Kaggle requires phone verification on your account once. Free.

## 2. Run

Click **Run All**. Cells in order:

| Cell | Purpose | Time |
|---|---|---|
| 1 | env sanity (verifies GPU is P100) | <5 s |
| 2 | `git clone` the repo into `/tmp/M1` | ~10 s |
| 3 | `pip install -r requirements.txt` + accelerate, safetensors, peft, datasets | ~2 min |
| 4 | adapter smoke test (`pytest tests/test_llama_adapter.py`) | ~30 s |
| 5 | **train + PPL ablation + needle benchmark** | ~3–4 h |
| 6 | bundle checkpoints + result JSONs into `awareliquid_phase5.zip` | <30 s |
| 7 | print head of `ppl_ablation.json` / `needle.json` | <1 s |

All long-running output streams live to the cell.

## 3. Get the trained adapter

After cell 6, the right sidebar **Output** panel shows:

- `awareliquid_phase5.zip` — adapter checkpoints + benchmark JSONs + train logs (typically <100 MB)
- `checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt` — raw adapter only (no base weights)

Download the zip, then in your local repo:

```bash
mkdir -p checkpoints/llama_mt_adapter
cp <download>/awareliquid_phase5/llama_mt_adapter_001000.pt checkpoints/llama_mt_adapter/
```

## 4. Pass/fail criteria

Same as `CLOUD_RUN.md`:

| Metric | Pass |
|---|---|
| Training loss | Monotonically decreasing, no NaN |
| `ppl_ablation.json` | Adapter PPL ≤ base PPL × 1.10 (no big regression) |
| `needle.json` | Adapter beats base on at least one (context, depth) cell at ctx ≥ 2048 |

If PPL regresses badly *and* needle does not improve, the adapter is not helping at this scale — file a bug rather than scaling up.

## 5. Re-using the adapter locally

Once the checkpoint is downloaded, `demo_llama_mt_adapter.py` can load it:

```bash
python demo_llama_mt_adapter.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt
```

This is the path that eventually replaces the `vocab=200` sandbox in `demo_mvp_loop.py` (see `AWARELIQUID_SYSTEM_MVP.md` Phase 5 acceptance).

## 6. Notes & gotchas

- **Don't change the working dir mid-cell** — Kaggle saves outputs from `/kaggle/working` only.
- **`%env` lines only set vars for subsequent `!`/`%%bash` cells**, which is what we want.
- The bash script auto-times the result dir with `date +%Y%m%d_%H%M%S`; we override `RESULT_DIR` to a stable path so cell 6 finds the JSONs deterministically.
- Kaggle's free GPU quota is **30 h/week**; one run = ~4 h, so you have headroom for a second pass with different `MT_EVERY` or `SEQ_LEN`.
- If you want to switch to Qwen-1.5B (the script's default model), set `%env MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct` in cell 5 — tighter on 16GB; may need `SEQ_LEN=768`.
