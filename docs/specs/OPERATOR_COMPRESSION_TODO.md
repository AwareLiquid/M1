# MT-LNN Operator Compression TODO

This roadmap continues after Day1 direct target extraction.

## Status Snapshot

- Done: direct target extraction head (`target_logits`, `target_loss`).
- Done: Selective Copy direct-head training/evaluation helpers.
- Existing before this plan: recurrent `h_prev` persistence through `SessionMemory`.
- Missing: channel gating, compute skipping, benchmark report, and README integration.

## Day2 - State-Only Streaming Inference

- [x] Add a state-only streaming helper that accepts exactly one new token.
- [x] Preserve recurrent `h_prev` while dropping attention/GWTB/coherence KV history.
- [x] Track `token_count` separately from token history.
- [x] Add a smoke test proving the state-only cache does not grow with sequence length.
- [x] Add a demo flag for state-only generation.
- [x] Add demo flags for session loading/saving.

Acceptance:
- One-token calls work with `use_lnn_recurrence=True`.
- Returned cache contains recurrent states and no historical KV tensors in state-only mode.
- A 100-token smoke run keeps cache memory approximately constant for the state-only path.

## Day3 - Persistent Session Workflow

- [x] Wrap `model.save_state()` / `model.load_state()` in a higher-level demo workflow.
- [x] Add CLI options: `--session_id`, `--state_db`, `--state_only`.
- [x] Persist only `h_prev` plus metadata (`token_count`, timestamps).
- [x] Add resume smoke test: save state, reload model/session, continue inference.

Acceptance:
- A session can be resumed without replaying historical tokens.
- Existing KV-cached generation remains available as the default compatibility path.

## Day4 - Long-Run Stress Benchmark

- [x] Add `benchmarks/state_only_streaming.py`.
- [x] Compare three modes: full replay, KV cache, state-only recurrent cache.
- [x] Report latency, peak cache size, and output divergence metrics.
- [x] Run at 100, 1k, and configurable N steps.

Acceptance:
- State-only cache size is O(layers * h_prev), not O(tokens).
- Benchmark writes JSON results for investor/report use.

## Day5 - Multi-Scale Channel Gating Prototype

- [x] Add config flags for dynamic scale gates.
- [x] Start with a non-skipping multiplicative gate over tau scales.
- [x] Initialize gates near open so old checkpoints remain stable.
- [x] Expose diagnostics: mean gate per scale, active-scale ratio.

Acceptance:
- Existing tests pass with dynamic gates enabled and disabled.
- Gate diagnostics show different activations across input batches.

## Day6 - Compute Skipping and Demo Integration

- [x] Add thresholded scale masking behind a config flag.
- [x] Keep dense fallback for parity and training stability.
- [x] Update `demo.py` to expose direct target and state-only modes.
- [x] Add examples for direct target and interactive stateful sessions.

Deferred:
- [x] Add an optional sparse resonance path that computes only top-k tau scales.
- [x] Add dense vs sparse resonance ablation benchmark.
- [ ] Replace the PyTorch top-k sparse path with a custom low-level sparse kernel if profiling justifies it.

Acceptance:
- Simple benchmark shows fewer active tau scales on low-information inputs.
- Demo can show direct extraction and state-only continuation in one script.

## Day7 - Benchmark Report

- [x] Add benchmark runner for Day1-Day6 features.
- [x] Generate a Markdown report with speed/cache/accuracy tables.
- [x] Include caveats: state-only mode trades exact attention recall for compressed recurrent memory.
- [x] Update command docs with reproducible commands.
- [x] Add JSONL metric export for benchmark observability.
- [x] Add a pinned environment lock file for reproducible smoke benchmark reruns.
- [x] Document SQLite session persistence as local-first rather than high-QPS distributed storage.

Acceptance:
- One command regenerates the summary report.
- Claims distinguish measured results from architectural hypotheses.

## Production Hardening Backlog

- [ ] Replace ad-hoc `print` logging in demos/training with project-level logging.
- [ ] Add a service storage adapter for recurrent state if M1 is deployed behind concurrent API traffic.
- [ ] Add CI jobs that install from `requirements.lock` and run smoke benchmarks on CPU.
- [ ] Profile sparse resonance on GPU before considering a Triton/CUDA fused kernel.
